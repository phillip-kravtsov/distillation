import cProfile
import gc
import glob
import json
import os
import random
from dataclasses import asdict

import numpy as np
import torch
import torch.distributed as dist

import wandb
from config import TrainingConfig

OUTPUT_DIR = "/workspace/monorepo/output"


def get_latest_checkpoint_dir(run_name: str) -> str:
    run_dir = get_run_dir(run_name)
    paths = glob.glob(os.path.join(run_dir, "step-*"))
    steps = [p.split("-")[-1] for p in paths]
    if "final" in steps:
        checkpoint_dir = os.path.join(run_dir, "step-final")
    else:
        step = max([int(s) for s in steps])
        checkpoint_dir = os.path.join(run_dir, f"step-{step}")
    return checkpoint_dir


def clear_mem():
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()


def print_parameters(model):
    for name, _ in model.named_parameters():
        if any(str(i) in name for i in range(1, 10)):
            continue
        if "0" in name:
            print(name.replace("0", "%d"))
        else:
            print(name)


def python_profile_function(func):
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        profiler.dump_stats(f"{func.__name__}.prof")
        return result

    return wrapper


# from Stas Bekman https://github.com/stas00/ml-engineering/tree/master/reproducibility
def enforce_reproducibility(use_seed=None):
    seed = use_seed if use_seed is not None else random.randint(1, 1000000)

    random.seed(seed)  # python RNG
    np.random.seed(seed)  # numpy RNG

    # pytorch RNGs
    torch.manual_seed(seed)  # cpu + cuda
    torch.cuda.manual_seed_all(seed)  # multi-gpu - can be called without gpus
    return seed


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_all_reduce_mean(tensor):
    if hasattr(dist.ReduceOp, "AVG"):
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    else:
        dist.all_reduce(tensor, op=dist.reduce_op.SUM)
        tensor = tensor / dist.get_world_size()
    return tensor


# copied from artidoro/qlora
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param:2f}%"
    )


def get_checkpoint_dir(step, run_name=None):
    if run_name is None:
        assert wandb.run is not None
        run_name = wandb.run.name
    return os.path.join(OUTPUT_DIR, f"{run_name}/step-{step}")


def get_run_dir(run_name=None):
    if run_name is None:
        assert wandb.run is not None
        run_name = wandb.run.name
    return os.path.join(OUTPUT_DIR, run_name)


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


def export_profile(prof, local_rank):
    trace_path = f"traces/trace_{local_rank}.json"
    timeline_path = f"timelines/memory_timeline_{local_rank}.html"
    print(f"Exporting chrome trace to {trace_path}")
    if not os.path.exists("traces"):
        os.makedirs("traces")
    prof.export_chrome_trace(f"traces/trace_{local_rank}.json")

    print(f"Exporting memory timeline to {timeline_path}")
    if not os.path.exists("timelines"):
        os.makedirs("timelines")
    prof.export_memory_timeline(
        f"timelines/memory_timeline_{local_rank}.html", f"cuda:{local_rank}"
    )
