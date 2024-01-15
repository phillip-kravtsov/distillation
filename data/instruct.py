import copy
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

import datasets
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import PreTrainedTokenizerBase

IGNORE_INDEX = -100


def format_mixtral_prompt(tokenizer, instruction: str, input: str):
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

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _filter_tokenize_fn(
    strings: Sequence[str], tokenizer: PreTrainedTokenizerBase
) -> List:
    samples = []
    for text in strings:
        tokens = tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            max_length=2048,
            truncation=True,
        )

        if tokens.input_ids.squeeze().numel() < tokenizer.model_max_length:
            samples.append(True)
        else:
            samples.append(False)

    return samples


def filter_long_samples(
    samples: Dict[str, Sequence[str]],
    tokenizer: PreTrainedTokenizerBase,
) -> List:
    inputs = samples["input"]
    instructions = samples["instruction"]
    sources = [
        format_mixtral_prompt(tokenizer, instruction, input)
        for input, instruction in zip(inputs, instructions)
    ]
    targets = samples["output"]
    examples = [s + t for s, t in zip(sources, targets)]

    return _filter_tokenize_fn(examples, tokenizer)


def preprocess(
    samples: Dict[str, Sequence[str]],
    tokenizer: PreTrainedTokenizerBase,
    train_on_inputs: bool = False,
):
    inputs = samples["input"]
    instructions = samples["instruction"]
    sources = [
        format_mixtral_prompt(tokenizer, instruction, input)
        for input, instruction in zip(inputs, instructions)
    ]
    targets = samples["output"]
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        if not train_on_inputs:
            label[:source_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)


def process_dataset(
    input_dataset,
    tokenizer: PreTrainedTokenizerBase,
    train_on_inputs: bool,
):
    workers = 12
    filtered_dataset = input_dataset.filter(
        lambda samples: filter_long_samples(samples, tokenizer),
        batched=True,
        batch_size=3000,
        num_proc=workers,
    )
    dataset = filtered_dataset.map(
        lambda samples: preprocess(samples, tokenizer, train_on_inputs),
        batched=True,
        batch_size=3000,
        num_proc=workers,
    )
    return dataset


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, dataset):
        super(SupervisedDataset, self).__init__()
        self.input_ids = dataset["input_ids"]
        self.labels = dataset["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=torch.tensor(self.input_ids[i]),
            labels=torch.tensor(self.labels[i]),
        )


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: PreTrainedTokenizerBase

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def load_dataset(
    tokenizer, train_batch_size, test_batch_size, local_rank, world_size, seed=42
):
    should_reprocess = False
    if os.path.exists("data/openhermes-processed-labels") and not should_reprocess:
        train_dataset = datasets.load_from_disk(
            "data/openhermes-processed-labels/train"
        )
        test_dataset = datasets.load_from_disk("data/openhermes-processed-labels/test")
    else:
        ds = datasets.load_dataset("teknium/openhermes")
        split = ds["train"].train_test_split(0.01, seed=seed)
        train, test = split["train"], split["test"]
        train_dataset = process_dataset(train, tokenizer, True)
        test_dataset = process_dataset(test, tokenizer, False)
        if local_rank == 0:
            train_dataset.save_to_disk("data/openhermes-processed-labels/train")
            test_dataset.save_to_disk("data/openhermes-processed-labels/test")

    train_dataset = SupervisedDataset(train_dataset)
    test_dataset = SupervisedDataset(test_dataset)

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        sampler=train_sampler,
        num_workers=12,
        pin_memory=True,
        collate_fn=DataCollatorForSupervisedDataset(tokenizer),
    )

    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=False,
        seed=seed,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        num_workers=12,
        sampler=test_sampler,
        pin_memory=True,
        collate_fn=DataCollatorForSupervisedDataset(tokenizer),
    )
    return train_loader, test_loader
