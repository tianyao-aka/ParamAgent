#!/usr/bin/env python
"""
Multi-GPU QLoRA fine-tuning of Llama-3.x on MIXED math + code pitfalls.

Combines both math and code datasets for comprehensive pitfalls prediction training.
- Math dataset: problem -> pitfalls mapping
- Code dataset: func_sign -> pitfalls/implementation mapping
- Auto-detects dataset type and applies appropriate formatting
- Single --limit flag applies to both datasets (N from each)

Example launch (2 GPUs with 10 total samples = 5 math + 5 code):
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 LoRA_Llama3_Mixed_multigpu.py \
  --dataset1_path benchmarks/math_finetune_dataset/math_finetune_dataset.json \
  --dataset2_path benchmarks/synthetic_code_datasets.json \
  --limit 5 \
  --output_dir tmp/test-mixed-llama3-8b \
  --base_model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --num_epochs 2

Example with only one dataset:
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 LoRA_Llama3_Mixed_multigpu.py \
  --dataset1_path benchmarks/math_finetune_dataset/math_finetune_dataset.json \
  --dataset2_path none \
  --output_dir ./lora-math-only

Dependencies
------------
pip install "transformers>=4.42.0" "datasets>=2.19.0" \
            peft bitsandbytes==0.43.2 accelerate
"""

from __future__ import annotations
import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

def detect_dataset_type(data: List[Dict]) -> str:
    """
    Auto-detect dataset type by inspecting first entry's keys.

    Args:
        data: List of dataset examples

    Returns:
        "math" if dataset has "problem" + "pitfalls"
        "code" if dataset has "func_sign" + ("raw_text"/"raw_texts"/"pitfalls")

    Raises:
        ValueError: If dataset type cannot be determined
    """
    if not data:
        raise ValueError("Dataset is empty")

    first = data[0]

    # Math dataset: has "problem" and "pitfalls"
    if "problem" in first and "pitfalls" in first:
        return "math"

    # Code dataset: has "func_sign" and some form of output
    if "func_sign" in first:
        if "raw_text" in first or "raw_texts" in first or "pitfalls" in first:
            return "code"

    raise ValueError(
        f"Cannot detect dataset type. First entry has keys: {list(first.keys())}\n"
        f"Expected either: ['problem', 'pitfalls'] for math "
        f"or ['func_sign', 'raw_text'/'raw_texts'/'pitfalls'] for code"
    )


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


def load_code_json(path: str | Path) -> List[Dict[str, str]]:
    """
    Load a JSON list where each element can be:
      1. A string "func_sign: ...\\ndocstring: ..." (func_sign parsed from prefix)
      2. A dict with "func_sign" and either "raw_text"/"raw_texts"
      3. A dict with "func_sign" + "pitfalls" (extracts only pitfalls)
    Returns a list of {"func_sign", "raw_text"}.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Top-level JSON must be a list")
    processed: List[Dict[str, str]] = []
    for ex in data:
        if isinstance(ex, str):
            lines = ex.split("\n")
            func_sign = None
            for line in lines:
                if line.startswith("func_sign:"):
                    func_sign = line.split("func_sign:", 1)[1].strip()
                    break
            if not func_sign:
                raise KeyError(f"String entry missing func_sign: prefix: {ex[:80]}")
            processed.append({"func_sign": func_sign, "raw_text": ex})
            continue

        if "func_sign" not in ex:
            raise KeyError("Missing 'func_sign' field")

        if "raw_text" in ex or "raw_texts" in ex:
            k = "raw_texts" if "raw_texts" in ex else "raw_text"
            processed.append({"func_sign": ex["func_sign"], "raw_text": ex[k]})
        elif "pitfalls" in ex:
            # Only use pitfalls (no flawed implementations)
            processed.append({"func_sign": ex["func_sign"], "raw_text": ex["pitfalls"]})
        else:
            raise KeyError(
                "Entry must have 'raw_text', 'raw_texts', or 'pitfalls'"
            )
    return processed


def load_mixed_datasets(
    dataset1_path: str | None,
    dataset2_path: str | None,
    limit: int | None,
    seed: int,
) -> Tuple[List[Dict], str, str | None]:
    """
    Load and combine up to 2 datasets with optional sampling.

    Args:
        dataset1_path: Path to first dataset (required)
        dataset2_path: Path to second dataset (optional, "none" to skip)
        limit: If specified, sample this many examples from EACH dataset
        seed: Random seed for sampling

    Returns:
        Tuple of (all_examples, dataset1_type, dataset2_type)
        dataset2_type is None if dataset2 is not loaded
    """
    all_examples = []
    dataset1_type = None
    dataset2_type = None

    # Validate at least one dataset provided
    if not dataset1_path:
        raise ValueError("dataset1_path is required")

    # Load dataset 1 (always required)
    print(f"Loading dataset1 from: {dataset1_path}")
    with open(dataset1_path, "r", encoding="utf-8") as f:
        dataset1_raw = json.load(f)

    dataset1_type = detect_dataset_type(dataset1_raw)
    print(f"  Detected type: {dataset1_type}")

    # Parse based on type
    if dataset1_type == "math":
        dataset1 = load_math_json(dataset1_path)
    else:
        dataset1 = load_code_json(dataset1_path)

    # Apply limit to dataset1
    if limit is not None and limit < len(dataset1):
        random.seed(seed)
        dataset1 = random.sample(dataset1, limit)
        print(f"  Sampled {len(dataset1)} examples")
    else:
        print(f"  Loaded {len(dataset1)} examples")

    # Tag with dataset type
    for ex in dataset1:
        ex['dataset_type'] = dataset1_type
    all_examples.extend(dataset1)

    # Load dataset 2 (optional)
    if dataset2_path is not None and dataset2_path.lower() != "none":
        print(f"Loading dataset2 from: {dataset2_path}")
        with open(dataset2_path, "r", encoding="utf-8") as f:
            dataset2_raw = json.load(f)

        dataset2_type = detect_dataset_type(dataset2_raw)
        print(f"  Detected type: {dataset2_type}")

        # Parse based on type
        if dataset2_type == "math":
            dataset2 = load_math_json(dataset2_path)
        else:
            dataset2 = load_code_json(dataset2_path)

        # Apply limit to dataset2
        if limit is not None and limit < len(dataset2):
            random.seed(seed)
            dataset2 = random.sample(dataset2, limit)
            print(f"  Sampled {len(dataset2)} examples")
        else:
            print(f"  Loaded {len(dataset2)} examples")

        # Tag with dataset type
        for ex in dataset2:
            ex['dataset_type'] = dataset2_type
        all_examples.extend(dataset2)

    return all_examples, dataset1_type, dataset2_type


# ----------------------------- Prompt helpers --------------------------- #

MATH_SYSTEM_PROMPT = (
    "You are an AI assistant specialized in mathematics education. "
    "Given a math problem, identify and explain the common mistakes and pitfalls "
    "students might encounter when solving it."
)

CODE_SYSTEM_PROMPT = (
    "You are an AI assistant in coding. "
    "Given a Python function signature and docstring, list potential pitfalls."
)


def format_prompt_mixed(example: Dict[str, Any], dataset_type: str) -> str:
    """
    Format prompt based on dataset type.

    Args:
        example: Dataset example with keys depending on type
        dataset_type: "math" or "code"

    Returns:
        Formatted prompt with [INST]...[/INST] structure
    """
    if dataset_type == "math":
        problem = example["problem"]
        pitfalls = example.get("pitfalls", "")
        user_block = (
            "[INST] <<SYS>> "
            f"{MATH_SYSTEM_PROMPT} <</SYS>>\n\n"
            f"PROBLEM:\n{problem.strip()}\n\n"
            "[/INST]"
        )
        return f"<s>{user_block}\n{pitfalls.strip()}</s>"

    elif dataset_type == "code":
        func_sign = example["func_sign"]
        raw_text = example.get("raw_text", "")
        user_block = (
            "[INST] <<SYS>> "
            f"{CODE_SYSTEM_PROMPT} <</SYS>>\n\n"
            f"FUNC_SIGNATURE:\n{func_sign.strip()}\n\n"
            "[/INST]"
        )
        return f"<s>{user_block}\n{raw_text.strip()}</s>"

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


# ----------------------------- Arg parsing ------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Multi-GPU QLoRA fine-tuner for mixed math+code pitfalls")
    p.add_argument("--dataset1_path", required=True, help="First JSON file (math or code)")
    p.add_argument("--dataset2_path", default=None,
                   help="Second JSON file (optional, use 'none' to skip)")
    p.add_argument("--limit", type=int, default=None,
                   help="Sample this many examples from EACH dataset. Total = 2*limit if both datasets used.")
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

    # 1) Dataset prep: load mixed datasets with auto-detection
    all_examples, type1, type2 = load_mixed_datasets(
        args.dataset1_path,
        args.dataset2_path,
        args.limit,
        args.seed
    )

    if local_rank == 0:
        print(f"\nLoaded {len(all_examples)} total examples:")
        count1 = sum(1 for ex in all_examples if ex['dataset_type'] == type1)
        print(f"  - Dataset 1 ({type1}): {count1}")
        if type2:
            count2 = sum(1 for ex in all_examples if ex['dataset_type'] == type2)
            print(f"  - Dataset 2 ({type2}): {count2}")

    # 2) Create HuggingFace Dataset and format prompts
    formatted_examples = []
    for ex in all_examples:
        full_text = format_prompt_mixed(ex, ex['dataset_type'])
        formatted_examples.append({"full_text": full_text})

    full_ds = Dataset.from_list(formatted_examples).shuffle(seed=args.seed)
    split = full_ds.train_test_split(test_size=0.15, seed=args.seed)
    train_ds, valid_ds = split["train"], split["test"]

    if local_rank == 0:
        print(f"Split into train: {len(train_ds)}, validation: {len(valid_ds)}")

    # 3) Tokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tok.pad_token = tok.eos_token

    # 4) Base model in 4-bit (QLoRA)
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

    # 5) Attach LoRA adapters
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

    # 6) Tokenise on-the-fly (only calculate loss on output, not instruction)
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

    # 7) Collator
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    # 8) Training arguments (DDP aware via torchrun)
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

    # 9) Trainer
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=valid_tok,
        tokenizer=tok,
        data_collator=collator,
    )

    # 10) Train & save
    trainer.train()
    if local_rank == 0:
        trainer.save_model(args.output_dir)
        with open(Path(args.output_dir) / "loss_history.json", "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)
        print(f"✅ Fine-tuning complete — artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
