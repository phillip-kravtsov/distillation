import argparse
import functools
import sys
import glob
import json
import os
import pprint
import shutil
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict
from typing import *

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.distributed.fsdp.api import (
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
from torch.profiler import profile
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

import lora
import utils
import wandb
from config import TrainingConfig, parse_training_args
from data import distill, instruct
from utils import get_checkpoint_dir, get_run_dir

OUTPUT_DIR = "/workspace/monorepo/output"
SAVE_SIGNAL_FILE = "/root/should_save"
CACHE_DIR = "/workspace/.cache"
MAX_STEPS_TO_KEEP = 3


def get_offline_log_pq(student, batch: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
    logit_mask = batch["attention_mask"][:, 1:].bool()
    teacher_logits = batch.pop("teacher_logits")[:, :-1][logit_mask]
    student_logits = student(**batch).logits[:, :-1][logit_mask]
    log_p = torch.log_softmax(teacher_logits, dim=-1)
    log_q = torch.log_softmax(student_logits, dim=-1)
    return log_p, log_q


def get_online_log_pq(
    batch: Dict[str, Tensor],
    local_rank: int,
) -> Tuple[Tensor, Tensor]:
    logit_mask = batch["attention_mask"][:, 1:].bool()
    student_logits = batch["student_logits"][logit_mask]
    teacher_logits = batch["teacher_logits"].to(f"cuda:{local_rank}")[:, :-1][
        logit_mask
    ]
    log_p = torch.log_softmax(teacher_logits, dim=-1)
    log_q = torch.log_softmax(student_logits, dim=-1)
    return log_p, log_q


@torch.inference_mode()
def distillation_eval(
    student,
    teacher,
    tokenizer,
    prompt_dataloader,
    local_rank,
    online: bool,
    max_eval_batches: Optional[int] = None,
):
    batch_losses: Dict[str, List[float]] = defaultdict(list)
    for i, batch in tqdm(enumerate(prompt_dataloader)):
        if max_eval_batches is not None and i > max_eval_batches:
            break
        if online:
            batch = get_online_batch(
                student,
                teacher,
                tokenizer,
                batch,
                local_rank=local_rank,
                include_student_logits=True,
            )
            log_p, log_q = get_online_log_pq(batch, local_rank)
        else:
            batch = get_offline_batch(teacher, batch, local_rank=local_rank)
            log_p, log_q = get_offline_log_pq(
                student,
                batch,
            )
        reverse_kl = F.kl_div(log_q, log_p, reduction="batchmean", log_target=True)
        forward_kl = F.kl_div(log_p, log_q, reduction="batchmean", log_target=True)
        batch_losses["reverse_kl"].append(reverse_kl.cpu().item())
        batch_losses["forward_kl"].append(forward_kl.cpu().item())
    return {k: sum(v) / len(v) for k, v in batch_losses.items()}


class OnlineDistillationDataset(Dataset):
    def __init__(self, examples):
        self.input_ids = examples["input_ids"]
        self.attention_mask = [torch.ones_like(ids) for ids in examples["input_ids"]]
        self.teacher_logits = examples["teacher_logits"]
        self.student_logits = examples["student_logits"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "teacher_logits": self.teacher_logits[idx],
            "student_logits": self.student_logits[idx],
        }


class ResumableSampler(Sampler):
    def __init__(self, data_source: Dataset):
        self.data_source = data_source
        self.current_index = 0

    def __iter__(self):
        self.current_index = max(0, self.current_index - 1)
        while self.current_index < len(self.data_source):
            yield self.current_index
            self.current_index += 1

    def __len__(self):
        return len(self.data_source)


def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    teacher_logits = [item["teacher_logits"] for item in batch]
    student_logits = [item["student_logits"] for item in batch]

    # Pad sequences to the maximum length in the batch
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    teacher_logits = pad_sequence(teacher_logits, batch_first=True, padding_value=0)
    student_logits = pad_sequence(student_logits, batch_first=True, padding_value=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "teacher_logits": teacher_logits,
        "student_logits": student_logits,
    }


@torch.inference_mode()
def get_online_batch(
    student,
    teacher,
    tokenizer,
    batch: Dict[str, Tensor],
    local_rank,
    include_student_logits=False,
) -> Dict[str, Tensor]:
    for k, v in batch.items():
        batch[k] = v.to(f"cuda:{local_rank}")
    student.use_cache = True
    if hasattr(student, "module"):
        student.module.gradient_checkpointing_disable()
    else:
        student.gradient_checkpointing_disable()
    student.eval()
    teacher.eval()
    max_tokens = 1024
    student_logits = []

    student_tokens, input_ids = batch["input_ids"], batch["input_ids"]
    current_tokens = student_tokens.size(1)
    max_new_tokens = max_tokens - current_tokens

    attention_mask = batch["attention_mask"]
    past_key_values = None
    completed = torch.zeros(
        (input_ids.size(0)), dtype=torch.bool, device=input_ids.device
    )

    # Produces at least one new token.
    # Every logit computed is used to sample a new token.
    new_tokens = 0
    start = time.perf_counter()
    while new_tokens < max_new_tokens:
        # forward
        outputs = student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        if include_student_logits:
            student_logits.append(outputs.logits)
        next_token = torch.multinomial(
            torch.softmax(outputs.logits.detach()[:, -1], dim=-1), num_samples=1
        )
        student_tokens = torch.cat([student_tokens, next_token], dim=1)
        new_tokens += 1
        completed = completed | (next_token.view(-1) == tokenizer.eos_token_id)
        next_token[completed] = 0
        if torch.all(completed):
            break

        input_ids = next_token
        past_key_values = outputs.past_key_values
        attention_mask = None

    if include_student_logits:
        student_logits = torch.cat(student_logits, dim=1)
    student.use_cache = False
    if hasattr(student, "module"):
        student.module.gradient_checkpointing_enable()
    else:
        student.gradient_checkpointing_enable()
    student.train()

    tam = batch["attention_mask"]
    teacher_attention_mask = torch.cat(
        [
            tam,
            torch.ones((tam.size(0), new_tokens), dtype=tam.dtype, device=tam.device),
        ],
        dim=1,
    )
    start_teacher = time.perf_counter()
    teacher_logits = teacher(
        input_ids=student_tokens, attention_mask=teacher_attention_mask
    ).logits.cpu()
    # TODO: Log these.
    completed_mean = completed.float().mean().cpu().item()
    gen_tok_s = new_tokens * student_tokens.size(0) / (time.perf_counter() - start)
    teacher_tok_s = (
        teacher_logits.size(0)
        * teacher_logits.size(1)
        / (time.perf_counter() - start_teacher)
    )
    if len(student_logits) > 0 and isinstance(student_logits, Tensor):
        assert teacher_logits.size(0) == student_logits.size(0)
        assert (
            teacher_logits.size(1) == student_logits.size(1) + 1
        ), f"tl {teacher_logits.size()} dl {student_logits.size()}"
    return_dict = {
        "input_ids": student_tokens,
        "attention_mask": teacher_attention_mask & input_ids.ne(0),
        "teacher_logits": teacher_logits.cpu(),  # Save GPU memory as B * S * V can get large
    }
    if include_student_logits:
        return_dict["student_logits"] = student_logits
    return return_dict


@torch.inference_mode()
def get_offline_batch(
    teacher,
    batch,
    local_rank,
) -> Dict[str, Tensor]:
    for k, v in batch.items():
        batch[k] = v.to(f"cuda:{local_rank}")
    teacher_logits = teacher(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    ).logits
    return {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "teacher_logits": teacher_logits,
    }


def save_fsdp_model(step, model, tokenizer, local_rank):
    dist.barrier()
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = model.state_dict()
    if local_rank == 0:
        print("Saving model.")
        model.save_pretrained(get_checkpoint_dir(step), state_dict=cpu_state)


def save_model(step, model, tokenizer, is_lora, local_rank, max_checkpoints=2):
    checkpoint_dir = get_checkpoint_dir(step)
    if local_rank == 0 and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if local_rank == 0 and is_lora:
        lora.save_lora_model(step, model, tokenizer)
    elif isinstance(model, FSDP):
        save_fsdp_model(step, model, tokenizer, local_rank)
    elif local_rank == 0 and isinstance(model, DDP):
        model.module.save_pretrained(checkpoint_dir)
    else:
        model.save_pretrained(checkpoint_dir)

    if local_rank == 0:
        tokenizer.save_pretrained(checkpoint_dir)
        if os.path.exists(SAVE_SIGNAL_FILE):
            os.remove(SAVE_SIGNAL_FILE)

        checkpoints = glob.glob(os.path.join(get_run_dir(), "step-*"))
        if len(checkpoints) > max_checkpoints and step != "final":
            steps = sorted([int(c.split("-")[-1]) for c in checkpoints])
            for step_to_delete in steps[:-MAX_STEPS_TO_KEEP]:
                print(f"Deleting checkpoint {step_to_delete}")
                shutil.rmtree(get_checkpoint_dir(step_to_delete))


def load_teacher(config: TrainingConfig, local_rank: int):
    print("Loading teacher model.")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        config.teacher_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if config.use_flash_attn else "eager",
        device_map=f"cuda:{local_rank}",
    ).eval()
    teacher_model.config.use_cache = False
    if not config.teacher_no_fsdp:
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={teacher_model.model.layers[0].__class__},
        )
        teacher_model = FSDP(
            teacher_model,
            cpu_offload=None,
            backward_prefetch=None,
            param_init_fn=None,
            auto_wrap_policy=auto_wrap_policy,
            use_orig_params=True,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=torch.cuda.current_device(),
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            ),
        ).eval()
    return teacher_model


def load_model(
    config: TrainingConfig,
    training: bool,
    local_rank=None,
):
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    model_path = config.model_name_or_path

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if config.use_flash_attn else "eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=False,
        cache_dir=CACHE_DIR,
    )
    if tokenizer.pad_token or tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    vocab = tokenizer.get_vocab()
    tokenizer.get_vocab = lambda: vocab

    # Avoid conflicts with gradient checkpointing during training.
    model.config.use_cache = False
    if config.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
        except TypeError:
            model._set_gradient_checkpointing(model, value=True)

    if config.lora:
        model = lora.get_lora_model(model, model_path, config, training)
    elif config.ddp:
        model = model.to(local_rank)
        model = DDP(model, device_ids=[local_rank])
    return model, tokenizer


def load_config(run_name: str) -> TrainingConfig:
    with open(f"{OUTPUT_DIR}/{run_name}/config.json", "r") as f:
        config = json.loads(f.read())
    return TrainingConfig(**config)


def save_config(config: TrainingConfig, run_name: str):
    run_dir = get_run_dir(run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            f.write(json.dumps(asdict(config)))


def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]

    result += list(model._parameters.keys())
    return result


def get_optimizer(model, max_train_steps: int, config: TrainingConfig):
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    if config.lora:
        opt_params = [
            {
                "params": [p for n, p in model.named_parameters() if "lora" in n],
                "learning_rate": config.learning_rate,
            },
        ]
    else:
        opt_params = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]
    utils.print_trainable_parameters(model)

    optimizer = torch.optim.AdamW(
        opt_params,
        foreach=False,
        weight_decay=config.weight_decay,
        lr=config.learning_rate,
    )
    lr_scheduler = get_scheduler(
        name=config.scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=max_train_steps,
    )
    return optimizer, lr_scheduler


def promote_scalar(x: torch.Tensor) -> torch.Tensor:
    return x.view(1) if len(x.size()) == 0 else x


def rkl(log_p, log_q):
    return F.kl_div(
        log_q,
        log_p,
        reduction="batchmean",
        log_target=True,
    )


def fkl(log_p, log_q):
    return F.kl_div(
        log_p,
        log_q,
        reduction="batchmean",
        log_target=True,
    )


class Trainer:
    def __init__(
        self,
        model,
        teacher_model,
        tokenizer,
        optimizer,
        lr_scheduler,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: TrainingConfig,
        local_rank: int,
        world_size: int,
        max_train_steps: Optional[int] = None,
    ):
        print("Initializing trainer.")
        self.model = model
        self.teacher_model = teacher_model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        if max_train_steps is not None:
            self.max_train_steps = max_train_steps
        elif config.max_train_steps is not None:
            self.max_train_steps = config.max_train_steps
        else:
            assert False, "Must specify max train steps."
        if config.loss == "rkl":
            self.compute_loss = rkl
        elif config.loss == "fkl":
            self.compute_loss = fkl
        elif config.loss == "jsd":
            raise NotImplementedError("JSD not implemented")
            # self.compute_loss = jsd
        else:
            raise ValueError(f"Inappropriate loss {config.loss}")

    def _init_tracking(self):
        if self.local_rank == 0:
            wandb.init(project=f"{self.config.task}", config=self.config.__dict__)
            if wandb.run is None:
                raise Exception
            save_config(self.config, wandb.run.name)

    def _profile_train_loop(self):
        self.step = utils.python_profile_function(self.step)
        with profile(profile_memory=True, record_shapes=True, with_stack=True) as prof:
            for batch in tqdm(self.train_dataloader, disable=(self.local_rank != 0)):
                self.step(batch)
                if self.completed_steps >= 2:
                    break
        utils.export_profile(prof, self.local_rank)

    def train(self):
        config = self.config
        self._init_tracking()
        self.completed_steps: int = 0
        self.throughput_counter = 0
        self.throughput_start_time = time.time()
        dist.barrier()
        self.model = self.model.to(self.local_rank)

        self.forward_context = (
            torch.autocast("cuda")
            if not isinstance(self.model, FSDP)
            else nullcontext()
        )
        if config.profile:
            self._profile_train_loop()
            return

        if self.local_rank == 0:
            print("Beginning train loop.")

        print(self.local_rank)
        for epoch in range(config.num_epochs):
            for i, batch in enumerate(
                tqdm(self.train_dataloader, disable=(self.local_rank != 0))
            ):
                if self.config.skip_steps is not None and i < self.config.skip_steps:
                    self.completed_steps += 1
                    continue
                self.step(batch)

                if (
                    self.completed_steps % config.save_every_steps == 0
                    or os.path.exists(SAVE_SIGNAL_FILE)
                ):
                    self.save_model()

                if self.completed_steps % config.eval_every_steps == 0:
                    eval_results = self.eval("val")
                    if self.local_rank == 0:
                        wandb.log(
                            eval_results,
                            step=self.completed_steps,
                        )
                if self.completed_steps >= self.max_train_steps:
                    break
            self.save_model()
            if self.completed_steps >= self.max_train_steps:
                break

        eval_log = self.eval("final")

        if self.local_rank == 0:
            wandb.log(eval_log)
        self.save_model("final")

    def step(self, batch):
        model, optimizer, lr_scheduler = self.model, self.optimizer, self.lr_scheduler
        model.train()
        dist.barrier()
        log = {}
        for k, v in batch.items():
            batch[k] = v.to(self.local_rank)
        try:
            with self.forward_context:
                if self.config.task == "online-distillation":
                    batch = get_online_batch(
                        model,
                        self.teacher_model,
                        self.tokenizer,
                        batch,
                        local_rank=self.local_rank,
                        include_student_logits=False,
                    )
                elif self.config.task == "offline-distillation":
                    batch = get_offline_batch(
                        self.teacher_model,
                        batch,
                        self.local_rank,
                    )
                else:
                    raise
                model.train()
                teacher_logits = batch.pop("teacher_logits")
                utils.clear_mem()
                # since the batch getter is in inference mode
                for k, v in batch.items():
                    batch[k] = v.clone()
                draft_logits = model(**batch).logits
                utils.clear_mem()

                logit_mask = batch["attention_mask"][:, 1:].bool()
                teacher_logits = teacher_logits.to(self.local_rank)[:, :-1][logit_mask]
                draft_logits = draft_logits[:, :-1][logit_mask]

                loss = self.compute_loss(
                    torch.log_softmax(teacher_logits, dim=-1),
                    torch.log_softmax(draft_logits, dim=-1),
                )
                log[self.config.loss] = promote_scalar(loss.detach())
            loss.backward()
            dist.barrier()
            for key in sorted(log.keys()):
                result = utils.get_all_reduce_mean(log[key])
                log[key] = result.cpu().item()
            if hasattr(model, "clip_grad_norm_"):
                model.clip_grad_norm_(1.0)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
        except torch.cuda.OutOfMemoryError as e:
            print("OOM on inputs with shape", batch["input_ids"].shape)
            raise e

        to_log = {f"train/{k}": v for k, v in log.items()}

        self.completed_steps += 1
        if self.local_rank == 0 and wandb.run is not None:
            wandb.log(to_log, step=self.completed_steps)

    def save_model(self, step=None):
        if step is None:
            step = self.completed_steps
        save_model(step, self.model, self.tokenizer, self.config.lora, self.local_rank)

    def eval(self, suffix: str) -> Dict[str, Any]:
        results = {}
        assert self.config.max_eval_batches is not None
        print("Evaluating.")
        eval_results = distillation_eval(
            student=self.model,
            teacher=self.teacher_model,
            tokenizer=self.tokenizer,
            prompt_dataloader=self.val_dataloader,
            local_rank=self.local_rank,
            online=self.config.task == "online-distillation",
            max_eval_batches=self.config.max_eval_batches,
        )
        for k, v in eval_results.items():
            results[f"{suffix}/{k}"] = v

        return results


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    parser = argparse.ArgumentParser(description="Train models.")
    config = parse_training_args(parser)
    utils.enforce_reproducibility(config.seed)

    print(config)
    if local_rank == 0:
        pprint.pprint(config)
        print(f"Using seed {config.seed}")

    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        local_rank = 0
        world_size = 1
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(local_rank)
    torch.cuda.set_per_process_memory_fraction(50 / 80)

    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    model, tokenizer = load_model(
        config,
        training=True,
        local_rank=local_rank,
    )

    if config.task == "online-distillation":
        train_dataloader, val_dataloader = distill.load_dataset(
            tokenizer,
            config.per_device_batch_size,
            config.eval_batch_size,
            local_rank,
            world_size,
            seed=42,
            max_input_seq_len=config.max_input_seq_len,
        )
    elif config.task == "offline-distillation":
        train_dataloader, val_dataloader = instruct.load_dataset(
            tokenizer,
            config.per_device_batch_size,
            config.eval_batch_size,
            local_rank,
            world_size,
            seed=42,
        )
    else:
        raise ValueError(f"unexpected task {config.task}")

    print("Loaded dataset.")
    optimizer, lr_scheduler = get_optimizer(
        model, config.max_train_steps or len(train_dataloader), config
    )

    teacher_model = load_teacher(config, local_rank=local_rank)
    trainer = Trainer(
        model=model,
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        local_rank=local_rank,
        world_size=world_size,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
    )
    trainer.train()


class Tee:
    """
    A helper class to tee print's output into a file.
    Usage:
    sys.stdout = Tee(filename)
    """

    def __init__(self, filename):
        self.stdout = sys.stdout
        self.file = open(filename, "a")

    def __getattr__(self, attr):
        return getattr(self.stdout, attr)

    def write(self, msg):
        self.stdout.write(msg)
        self.file.write(msg)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


if __name__ == "__main__":

    def guarded_main():
        import socket
        import sys
        import trace

        # enable the trace
        if 0:
            cwd = os.path.realpath(".")
            pid = os.getpid()
            hostname = socket.gethostname()
            local_rank = int(os.environ["LOCAL_RANK"])
            trace_output_file = f"{cwd}/trace-{hostname}-{local_rank}-{pid}.txt"

            # create a Trace object, telling it what to ignore, and whether to
            # do tracing or line-counting or both.
            tracer = trace.Trace(
                ignoredirs=[sys.prefix, sys.exec_prefix],
                trace=1,
                count=1,
                timing=True,
            )

            # run the new command using the given tracer
            sys.stdout = Tee(trace_output_file)
            tracer.run("main()")
        else:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
            main()

    guarded_main()