import os
from dataclasses import dataclass
from typing import Dict, Sequence

import datasets
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import PreTrainedTokenizerBase

IGNORE_INDEX = -100


def format_prompt(tokenizer: PreTrainedTokenizerBase, instruction: str, input: str):
    if input:
        content = f"{instruction}\n{input}"
    else:
        content = instruction
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _tokenize_fn(strings: Sequence[str], tokenizer: PreTrainedTokenizerBase) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            max_length=2048,
            truncation=True,
        )
        for text in strings
    ]

    input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        input_ids_lens=input_ids_lens,
    )


def preprocess(
    samples: Dict[str, Sequence[str]],
    tokenizer: PreTrainedTokenizerBase,
):
    sources = [
        format_prompt(tokenizer, instruction, input)
        for input, instruction in zip(samples["input"], samples["instruction"])
    ]
    return dict(text=sources)


def process_dataset(
    input_dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int,
):
    workers = 24
    dataset = input_dataset.map(
        lambda samples: preprocess(samples, tokenizer),
        batched=True,
        batch_size=3000,
        num_proc=workers,
    )

    def filter_text(text):
        return tokenizer(text, return_tensors="pt")["input_ids"].size(1) < max_seq_len

    return dataset.filter(lambda samples: filter_text(samples["text"]), batched=False)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, dataset):
        super(SupervisedDataset, self).__init__()
        self.text = dataset["text"]
        self.index = 0

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        text = self.text[index]
        self.index += 1
        return dict(
            text=text,
        )


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: PreTrainedTokenizerBase

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        text = [instance["text"] for instance in instances]

        """
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        """
        return self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)


IGNORE_INDEX = -100


def load_dataset(
    tokenizer,
    train_batch_size,
    test_batch_size,
    local_rank,
    world_size,
    max_input_seq_len,
    seed=42,
):
    should_reprocess = False

    if os.path.exists("data/openhermes-processed") and not should_reprocess:
        train_dataset = datasets.load_from_disk("data/openhermes-processed/train")
        test_dataset = datasets.load_from_disk("data/openhermes-processed/test")
    else:
        ds = datasets.load_dataset("teknium/openhermes")
        split = ds["train"].train_test_split(0.01, seed=seed)
        train, test = split["train"], split["test"]
        train_dataset = process_dataset(train, tokenizer, max_input_seq_len)
        test_dataset = process_dataset(test, tokenizer, max_input_seq_len)
        if local_rank == 0:
            train_dataset.save_to_disk("data/openhermes-processed/train")
            test_dataset.save_to_disk("data/openhermes-processed/test")

    train_dataset = SupervisedDataset(train_dataset)
    test_dataset = SupervisedDataset(test_dataset)

    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=False,
        seed=seed,
    )
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True,
        seed=seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        num_workers=0,
        sampler=train_sampler,
        pin_memory=True,
        collate_fn=DataCollatorForSupervisedDataset(tokenizer),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        sampler=test_sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=DataCollatorForSupervisedDataset(tokenizer),
    )
    return train_loader, test_loader
