#!/usr/bin/env python
"""
Multi-GPU QLoRA fine-tuning of Llama-3.x on math pitfalls prediction.

Designed for multi-GPU setups (e.g., 6x RTX 6000 Ada, 48 GB each):
- Loads the base model in 4-bit (QLoRA) to keep per-GPU memory manageable.
- Uses torchrun + DDP (one process per GPU).
- Gradient checkpointing + bf16 compute to further cut activation memory.
- Optional DeepSpeed config hook if you want ZeRO optimizer states.

Example launch (6 GPUs):
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --nproc_per_node 6 math/LoRA_Llama3_Math_multigpu.py \
  --dataset_path benchmarks/math_finetune_dataset/math_finetune_dataset.json \
  --output_dir ./lora-llama3-8b-math \
  --base_model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --per_device_batch_size 2 \
  --grad_accum_steps 8 \
  --num_epochs 3 \
  --lr 3e-5 \
  --max_seq_len 2048

Dependencies
------------
pip install "transformers>=4.42.0" "datasets>=2.19.0" \
            peft bitsandbytes==0.43.2 accelerate
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from peft.utils import prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# ----------------------------- Data helpers ----------------------------- #

def load_math_json(path: str | Path) -> List[Dict[str, str]]:
    """
    Load a JSON list where each element contains:
      - "problem": The math problem statement (input)
      - "pitfalls": Common mistakes description (output/label)
    Returns a list of {"problem", "pitfalls"}.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Top-level JSON must be a list")

    processed: List[Dict[str, str]] = []
    for idx, ex in enumerate(data):
        if "problem" not in ex:
            raise KeyError(f"Entry {idx} missing 'problem' field")
        if "pitfalls" not in ex:
            raise KeyError(f"Entry {idx} missing 'pitfalls' field")

        processed.append({
            "problem": ex["problem"],
            "pitfalls": ex["pitfalls"]
        })
    return processed


# ----------------------------- Prompt helpers --------------------------- #

SYSTEM_PROMPT = (
    "You are an AI assistant specialized in mathematics education. "
    "Given a math problem, identify and explain the common mistakes and pitfalls "
    "students might encounter when solving it."
)


def format_prompt(problem: str, pitfalls: str | None = None) -> str:
    """
    Format a math problem with instruction-output structure.

    Args:
        problem: The math problem statement
        pitfalls: Optional pitfalls text (for training, None for inference)

    Returns:
        Formatted prompt string with proper markers
    """
    user_block = (
        "[INST] <<SYS>> "
        f"{SYSTEM_PROMPT} <</SYS>>\n\n"
        f"PROBLEM:\n{problem.strip()}\n\n"
        "[/INST]"
    )
    return f"<s>{user_block}" if pitfalls is None else f"<s>{user_block}\n{pitfalls.strip()}</s>"


# ----------------------------- Arg parsing ------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Multi-GPU QLoRA fine-tuner for math pitfalls prediction")
    p.add_argument("--dataset_path", required=True, help="Path to math dataset JSON file")
    p.add_argument("--output_dir", required=True, help="Where to save adapters/checkpoints")
    p.add_argument("--base_model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--per_device_batch_size", type=int, default=1,
                   help="Micro-batch size per GPU (keep small for large models).")
    p.add_argument("--grad_accum_steps", type=int, default=16,
                   help="Gradient accumulation to reach desired global batch.")
    p.add_argument("--lr", type=float, default=1.5e-5)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--deepspeed_config", type=str, default=None,
                   help="Optional DeepSpeed JSON (e.g., deepspeed.json).")
    return p.parse_args()


# ----------------------------- Main routine ----------------------------- #

def main() -> None:
    args = parse_args()

    # DDP: set per-process device
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if local_rank == 0:
        print(f"Launching QLoRA on {world_size} GPUs — base model: {args.base_model}")

    # 1) Dataset prep: load + shuffle + split
    all_examples = load_math_json(args.dataset_path)
    full_ds = Dataset.from_list(
        [{"full_text": format_prompt(ex["problem"], ex["pitfalls"])} for ex in all_examples]
    ).shuffle(seed=args.seed)
    split = full_ds.train_test_split(test_size=0.15, seed=args.seed)
    train_ds, valid_ds = split["train"], split["test"]

    # 2) Tokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tok.pad_token = tok.eos_token

    # 3) Base model in 4-bit (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map={"": local_rank},  # one GPU per process via torchrun
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # 4) Attach LoRA adapters
    lora_cfg = LoraConfig(
        r=64,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    if local_rank == 0:
        model.print_trainable_parameters()

    # 5) Tokenise on-the-fly (only calculate loss on output, not instruction)
    def tok_fn(batch: Dict[str, Any]) -> Dict[str, Any]:
        # First tokenize everything normally
        t = tok(
            batch["full_text"],
            max_length=args.max_seq_len,
            truncation=True,
            padding="max_length",
        )

        # Create labels, masking instruction tokens with -100
        labels = []
        for full_text in batch["full_text"]:
            # Split at [/INST] to separate instruction from output
            if "[/INST]" in full_text:
                instruction_part, output_part = full_text.split("[/INST]", 1)
                instruction_part += "[/INST]"  # Include the marker in instruction

                # Tokenize each part separately
                inst_tokens = tok(instruction_part, add_special_tokens=False)["input_ids"]
                output_tokens = tok(output_part, add_special_tokens=False)["input_ids"]

                # Create labels: -100 for instruction, actual IDs for output
                example_labels = [-100] * len(inst_tokens) + output_tokens

                # Pad or truncate to max_seq_len
                if len(example_labels) < args.max_seq_len:
                    example_labels += [-100] * (args.max_seq_len - len(example_labels))
                else:
                    example_labels = example_labels[:args.max_seq_len]
            else:
                # Fallback: if no [/INST] found, mask everything
                example_labels = [-100] * args.max_seq_len

            labels.append(example_labels)

        t["labels"] = labels
        return t

    train_tok = train_ds.map(tok_fn, batched=True, remove_columns=["full_text"])
    valid_tok = valid_ds.map(tok_fn, batched=True, remove_columns=["full_text"])

    # 6) Collator
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    # 7) Training arguments (DDP aware via torchrun)
    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        tf32=True,
        logging_steps=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=4,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        report_to="none",
        ddp_find_unused_parameters=False,
        deepspeed=args.deepspeed_config,
        remove_unused_columns=False,
    )

    # 8) Trainer
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=valid_tok,
        tokenizer=tok,
        data_collator=collator,
    )

    # 9) Train & save
    trainer.train()
    if local_rank == 0:
        trainer.save_model(args.output_dir)
        with open(Path(args.output_dir) / "loss_history.json", "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)
        print(f"✅ Fine-tuning complete — artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
