import glob
import os
import shutil

import torch
from peft import PeftModel, get_peft_model
from peft.tuners.lora import LoraConfig, LoraLayer

from utils import get_checkpoint_dir, get_run_dir


def save_lora_model(step, model, tokenizer):
    model_path = os.path.join(get_checkpoint_dir(step), "adapter_model")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(get_checkpoint_dir(step))
    checkpoints = glob.glob(os.path.join(get_run_dir(), "step-*"))
    if len(checkpoints) > 2:
        steps = sorted([int(c.split("-")[-1]) for c in checkpoints])
        for step_to_delete in steps[:-2]:
            print(f"Deleting checkpoint {step_to_delete}")
            shutil.rmtree(get_checkpoint_dir(step_to_delete))


def get_lora_model(model, checkpoint_dir: str, config, training: bool):
    maybe_checkpoint_path = os.path.join(checkpoint_dir, "adapter_model")
    if os.path.exists(maybe_checkpoint_path):
        model = PeftModel.from_pretrained(
            model, maybe_checkpoint_path, is_trainable=training
        )
    else:
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                names = name.split(".")
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        lora_module_names.remove("lm_head")
        if not config.lora_vision:
            lora_module_names.remove("vision_embed_tokens")
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=list(lora_module_names),
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    if training:
        for name, module in model.named_modules():
            if hasattr(module, "weight") and module.weight.dtype != torch.bfloat16:
                print(name)
                module.to(torch.bfloat16)
            if isinstance(module, LoraLayer):
                module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_tokens" in name:
                module = module.to(torch.bfloat16)

    return model
