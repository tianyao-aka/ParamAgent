#!/usr/bin/env python
"""
Iterative Multi-Round QLoRA fine-tuning with progressive model refinement.

Each iteration:
1. Trains LoRA on current dataset (DDP across all GPUs)
2. Merges LoRA with base model (rank 0)
3. Generates new pitfalls dataset for next iteration (rank 0)

Progressive approach: Each iteration builds on the previous merged model.

Example launch (6 GPUs, 3 iterations):
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --nproc_per_node 6 \
  LoRA_Llama3_Code_multigpu_multiIter.py \
  --original_dataset benchmarks/augmented_coding_datasets_llama3_8b.json \
  --output_dir ./lora-llama3-8b-iterative \
  --base_model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --num_iterations 3 \
  --num_epochs 2 \
  --generation_temperature 0.8 \
  --generation_max_tokens 1024

Dependencies:
pip install "transformers>=4.42.0" "datasets>=2.19.0" \
            peft bitsandbytes==0.43.2 accelerate
"""

from __future__ import annotations
import argparse
import gc
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
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

def load_code_json(path: str | Path) -> List[Dict[str, str]]:
    """
    Load a JSON list where each element can be:
      1. A string "func_sign: ...\\ndocstring: ..." (func_sign parsed from prefix)
      2. A dict with "func_sign" and either "raw_text"/"raw_texts"
      3. A dict with "func_sign" + both "pitfalls" and "flawed_impl"
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
            # Use pitfalls (and optionally flawed_impl)
            raw = ex["pitfalls"]
            if "flawed_impl" in ex:
                raw = raw + "\n\n[Flawed Implementations]" + ex["flawed_impl"]
            processed.append({"func_sign": ex["func_sign"], "raw_text": raw})
        else:
            raise KeyError(
                "Entry must have 'raw_text', 'raw_texts', or 'pitfalls'"
            )
    return processed


def load_code_json_full(path: str | Path) -> List[Dict[str, str]]:
    """
    Load a JSON list and preserve all original fields.
    Used for generation where we need to maintain the original structure.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Top-level JSON must be a list")
    return data


# ----------------------------- Prompt helpers --------------------------- #

SYSTEM_PROMPT = (
    "You are an AI assistant in coding. "
    "Given a Python function signature and docstring, list potential pitfalls. "
)


def format_prompt(func_sign: str, raw_text: str | None = None) -> str:
    """Format prompt for training or inference."""
    user_block = (
        "[INST] <<SYS>> "
        f"{SYSTEM_PROMPT} <</SYS>>\n\n"
        f"FUNC_SIGNATURE:\n{func_sign.strip()}\n\n"
        "[/INST]"
    )
    return f"<s>{user_block}" if raw_text is None else f"<s>{user_block}\n{raw_text.strip()}</s>"


def format_inference_prompt(func_sign: str) -> str:
    """Format prompt for inference (no completion)."""
    return format_prompt(func_sign, raw_text=None)


# ----------------------------- Path helpers ----------------------------- #

def get_dataset_path(args: argparse.Namespace, iteration: int) -> str:
    """Get the dataset path for a given iteration."""
    if iteration == 1:
        return args.original_dataset
    else:
        base_name = os.path.basename(args.original_dataset).replace('.json', '')
        dataset_dir = os.path.join(args.output_dir, "datasets")
        os.makedirs(dataset_dir, exist_ok=True)
        return os.path.join(dataset_dir, f"{base_name}_iter{iteration}.json")


def get_base_model_path(args: argparse.Namespace, iteration: int) -> str:
    """
    Get the base model path for a given iteration.
    Iteration 1: Use original base model from HuggingFace
    Iteration 2+: Use previous iteration's merged model (progressive)
    """
    if iteration == 1:
        return args.base_model
    else:
        prev_iter = iteration - 1
        return os.path.join(args.output_dir, f"iteration_{prev_iter}", "merged_model")


def get_lora_output_path(args: argparse.Namespace, iteration: int) -> str:
    """Get the LoRA adapter output path for a given iteration."""
    return os.path.join(args.output_dir, f"iteration_{iteration}", "lora_adapter")


def get_merged_output_path(args: argparse.Namespace, iteration: int) -> str:
    """Get the merged model output path for a given iteration."""
    return os.path.join(args.output_dir, f"iteration_{iteration}", "merged_model")


# ----------------------------- Output parsing --------------------------- #

def parse_generated_output(text: str) -> Tuple[str, str]:
    """
    Parse generated text to extract pitfalls and flawed implementations.

    Expected format:
    <Pitfalls>:
    1. **Title**: description
    ...

    [Flawed Implementations]:
    ```python
    def func_v1(...):
        ...
    ```
    """
    pitfalls = ""
    flawed_impl = ""

    # Try to find pitfalls section
    pitfalls_patterns = [
        r'(<Pitfalls>:.*?)(?=\[Flawed|\n\n```python|$)',
        r'(\[Pitfalls\].*?)(?=\[Flawed|\n\n```python|$)',
        r'(Pitfalls:.*?)(?=\[Flawed|\n\n```python|$)',
    ]

    for pattern in pitfalls_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            pitfalls = match.group(1).strip()
            break

    # Try to find flawed implementations
    flawed_patterns = [
        r'(\[Flawed Implementation.*?\].*)',
        r'(\[Flawed Implementations\].*)',
        r'(Flawed Implementation.*?:.*```python.*)',
    ]

    for pattern in flawed_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            flawed_impl = ":\n\n" + match.group(1).strip()
            break

    # Fallback: if no pitfalls found, use entire text
    if not pitfalls:
        pitfalls = text.strip()

    return pitfalls, flawed_impl


# ----------------------------- Memory management ------------------------ #

def cleanup_memory():
    """Clean up GPU memory and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


# ----------------------------- Training function ------------------------ #

def train_iteration(
    args: argparse.Namespace,
    iteration: int,
    dataset_path: str,
    lora_output_dir: str,
    local_rank: int,
    world_size: int,
) -> bool:
    """
    Train LoRA for one iteration.

    Returns True if training completed, False if skipped (already exists).
    """
    # Check if already trained
    if os.path.exists(os.path.join(lora_output_dir, "adapter_config.json")):
        if local_rank == 0:
            print(f"  LoRA adapter already exists at {lora_output_dir}, skipping training...")
        return False

    # Determine base model path
    base_model_path = get_base_model_path(args, iteration)

    if local_rank == 0:
        print(f"\n  Phase 1: Training LoRA")
        print(f"  Base model: {base_model_path}")
        print(f"  Dataset: {dataset_path}")
        print(f"  Output: {lora_output_dir}")

    # Load dataset
    all_examples = load_code_json(dataset_path)
    if args.limit is not None and args.limit < len(all_examples):
        random.seed(args.seed)
        all_examples = random.sample(all_examples, args.limit)

    if local_rank == 0:
        print(f"  Loaded {len(all_examples)} training examples")

    # Create HuggingFace Dataset
    full_ds = Dataset.from_list(
        [{"full_text": format_prompt(ex["func_sign"], ex["raw_text"])} for ex in all_examples]
    ).shuffle(seed=args.seed)

    split = full_ds.train_test_split(test_size=0.15, seed=args.seed)
    train_ds, valid_ds = split["train"], split["test"]

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    tok.pad_token = tok.eos_token

    # 4-bit quantization config (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map={"": local_rank},
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = prepare_model_for_kbit_training(model)

    # Attach LoRA adapters
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

    # Tokenization function with label masking
    def tok_fn(batch: Dict[str, Any]) -> Dict[str, Any]:
        t = tok(
            batch["full_text"],
            max_length=args.max_seq_len,
            truncation=True,
            padding="max_length",
        )

        labels = []
        for full_text in batch["full_text"]:
            if "[/INST]" in full_text:
                instruction_part, output_part = full_text.split("[/INST]", 1)
                instruction_part += "[/INST]"

                inst_tokens = tok(instruction_part, add_special_tokens=False)["input_ids"]
                output_tokens = tok(output_part, add_special_tokens=False)["input_ids"]

                example_labels = [-100] * len(inst_tokens) + output_tokens

                if len(example_labels) < args.max_seq_len:
                    example_labels += [-100] * (args.max_seq_len - len(example_labels))
                else:
                    example_labels = example_labels[:args.max_seq_len]
            else:
                example_labels = [-100] * args.max_seq_len

            labels.append(example_labels)

        t["labels"] = labels
        return t

    train_tok = train_ds.map(tok_fn, batched=True, remove_columns=["full_text"])
    valid_tok = valid_ds.map(tok_fn, batched=True, remove_columns=["full_text"])

    # Data collator
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    # Training arguments
    targs = TrainingArguments(
        output_dir=lora_output_dir,
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
        eval_strategy="no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        report_to="none",
        ddp_find_unused_parameters=False,
        deepspeed=args.deepspeed_config,
        remove_unused_columns=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=valid_tok,
        tokenizer=tok,
        data_collator=collator,
    )

    # Train
    trainer.train()

    # Save LoRA adapter (rank 0 only)
    if local_rank == 0:
        trainer.save_model(lora_output_dir)

        # Save training log
        iter_dir = os.path.dirname(lora_output_dir)
        log_path = os.path.join(iter_dir, "training_log.json")
        with open(log_path, "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)

        print(f"  LoRA adapter saved to {lora_output_dir}")

    # Cleanup
    del model, trainer, train_tok, valid_tok, train_ds, valid_ds, full_ds
    cleanup_memory()

    return True


# ----------------------------- Merging function ------------------------- #

def merge_lora_weights(
    base_model_path: str,
    lora_path: str,
    output_path: str,
) -> bool:
    """
    Load base model, apply LoRA adapter, merge, and save.

    Returns True if merge completed, False if skipped (already exists).
    """
    # Check if already merged
    if os.path.exists(os.path.join(output_path, "config.json")):
        print(f"  Merged model already exists at {output_path}, skipping merge...")
        return False

    print(f"\n  Phase 2: Merging LoRA weights")
    print(f"  Base model: {base_model_path}")
    print(f"  LoRA adapter: {lora_path}")
    print(f"  Output: {output_path}")

    # Load base model in full precision (BF16) - NOT quantized for merging
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        torch_dtype=torch.bfloat16,
    )

    # Merge LoRA weights into base model
    print("  Merging weights...")
    model = model.merge_and_unload()

    # Save merged model
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)

    print(f"  Merged model saved to {output_path}")

    # Cleanup
    del model, base_model
    cleanup_memory()

    return True


# ----------------------------- Generation function ---------------------- #

def generate_new_dataset(
    model_path: str,
    original_dataset_path: str,
    output_path: str,
    args: argparse.Namespace,
) -> bool:
    """
    Generate new pitfalls dataset using the merged model.

    Returns True if generation completed, False if skipped (already exists).
    """
    # Check if already generated
    if os.path.exists(output_path):
        print(f"  Dataset already exists at {output_path}, skipping generation...")
        return False

    print(f"\n  Phase 3: Generating new dataset")
    print(f"  Model: {model_path}")
    print(f"  Original dataset: {original_dataset_path}")
    print(f"  Output: {output_path}")

    # Load model for inference
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Important for batch generation

    # Load original dataset to extract function signatures
    original_data = load_code_json_full(original_dataset_path)

    # Respect the limit argument for quick testing
    if args.limit is not None and args.limit < len(original_data):
        random.seed(args.seed)
        original_data = random.sample(original_data, args.limit)

    print(f"  Generating pitfalls for {len(original_data)} functions...")

    # Generate new pitfalls
    new_dataset = []
    batch_size = args.generation_batch_size

    for i in range(0, len(original_data), batch_size):
        batch = original_data[i:i + batch_size]

        # Create prompts (without completions)
        prompts = [format_inference_prompt(ex["func_sign"]) for ex in batch]

        # Tokenize
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.generation_max_tokens,
                do_sample=True,
                temperature=args.generation_temperature,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode outputs
        for j, output_ids in enumerate(outputs):
            # Get the prompt length for this specific input
            prompt_len = (inputs['input_ids'][j] != tokenizer.pad_token_id).sum().item()
            generated_ids = output_ids[prompt_len:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Parse generated text
            pitfalls, flawed_impl = parse_generated_output(generated_text)

            # Create new entry with same structure as original
            new_entry = {
                "func_sign": batch[j]["func_sign"],
                "pitfalls": pitfalls,
                "flawed_impl": flawed_impl if flawed_impl else batch[j].get("flawed_impl", ""),
                "raw_text": generated_text.strip()
            }
            new_dataset.append(new_entry)

        # Progress update
        if (i // batch_size + 1) % 10 == 0 or i + batch_size >= len(original_data):
            print(f"  Progress: {min(i + batch_size, len(original_data))}/{len(original_data)}")

    # Save new dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_dataset, f, indent=2, ensure_ascii=False)

    print(f"  New dataset saved: {output_path} ({len(new_dataset)} examples)")

    # Cleanup
    del model
    cleanup_memory()

    return True


# ----------------------------- Arg parsing ------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Iterative Multi-Round QLoRA Fine-Tuning with Progressive Refinement"
    )

    # Iteration settings
    p.add_argument("--num_iterations", type=int, default=3,
                   help="Number of iterative training rounds (default: 3)")
    p.add_argument("--original_dataset", required=True,
                   help="Path to initial dataset (iteration 1)")

    # Output settings
    p.add_argument("--output_dir", required=True,
                   help="Base directory for all outputs")

    # Model settings
    p.add_argument("--base_model", default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                   help="Base model to fine-tune (HF model ID or path)")

    # Training settings
    p.add_argument("--num_epochs", type=int, default=2,
                   help="Training epochs per iteration")
    p.add_argument("--per_device_batch_size", type=int, default=1,
                   help="Micro-batch size per GPU")
    p.add_argument("--grad_accum_steps", type=int, default=16,
                   help="Gradient accumulation steps")
    p.add_argument("--lr", type=float, default=1.5e-5,
                   help="Learning rate")
    p.add_argument("--max_seq_len", type=int, default=2048,
                   help="Maximum sequence length")
    p.add_argument("--seed", type=int, default=1,
                   help="Random seed")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of training examples (for testing)")
    p.add_argument("--deepspeed_config", type=str, default=None,
                   help="Optional DeepSpeed config JSON")

    # Generation settings
    p.add_argument("--generation_batch_size", type=int, default=4,
                   help="Batch size for pitfall generation")
    p.add_argument("--generation_max_tokens", type=int, default=1024,
                   help="Max tokens to generate per sample")
    p.add_argument("--generation_temperature", type=float, default=0.8,
                   help="Temperature for generation sampling")

    return p.parse_args()


# ----------------------------- Main routine ----------------------------- #

def main() -> None:
    args = parse_args()

    # DDP setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)

    # Initialize process group for DDP
    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    if local_rank == 0:
        print("=" * 70)
        print("Iterative Multi-Round QLoRA Fine-Tuning")
        print("=" * 70)
        print(f"Base model: {args.base_model}")
        print(f"Original dataset: {args.original_dataset}")
        print(f"Output directory: {args.output_dir}")
        print(f"Number of iterations: {args.num_iterations}")
        print(f"World size (GPUs): {world_size}")
        print("=" * 70)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Main iteration loop
    for iteration in range(1, args.num_iterations + 1):
        if local_rank == 0:
            print(f"\n{'#' * 70}")
            print(f"# ITERATION {iteration}/{args.num_iterations}")
            print(f"{'#' * 70}")

        # Get paths for this iteration
        current_dataset = get_dataset_path(args, iteration)
        lora_output = get_lora_output_path(args, iteration)
        merged_output = get_merged_output_path(args, iteration)

        # Create iteration directory
        iter_dir = os.path.dirname(lora_output)
        os.makedirs(iter_dir, exist_ok=True)

        # =====================================================================
        # Phase 1: Training (All GPUs via DDP)
        # =====================================================================
        train_iteration(
            args=args,
            iteration=iteration,
            dataset_path=current_dataset,
            lora_output_dir=lora_output,
            local_rank=local_rank,
            world_size=world_size,
        )

        # Barrier: Wait for all processes to finish training
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # =====================================================================
        # Phase 2: Merge LoRA with base model (Rank 0 only)
        # =====================================================================
        if local_rank == 0:
            base_model_path = get_base_model_path(args, iteration)
            merge_lora_weights(base_model_path, lora_output, merged_output)

        # Barrier: Wait for merge to complete
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # =====================================================================
        # Phase 3: Generate new dataset (Rank 0 only, not last iteration)
        # =====================================================================
        if local_rank == 0 and iteration < args.num_iterations:
            next_dataset = get_dataset_path(args, iteration + 1)
            generate_new_dataset(
                model_path=merged_output,
                original_dataset_path=args.original_dataset,
                output_path=next_dataset,
                args=args,
            )

        # Barrier: Wait for generation to complete before next iteration
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if local_rank == 0:
            print(f"\n  Iteration {iteration} complete!")

    # Final summary
    if local_rank == 0:
        print("\n" + "=" * 70)
        print("ALL ITERATIONS COMPLETE!")
        print("=" * 70)
        print(f"\nOutput structure:")
        for i in range(1, args.num_iterations + 1):
            print(f"  iteration_{i}/")
            print(f"    lora_adapter/  - LoRA weights")
            print(f"    merged_model/  - Full merged model")
        print(f"  datasets/")
        for i in range(2, args.num_iterations + 1):
            base_name = os.path.basename(args.original_dataset).replace('.json', '')
            print(f"    {base_name}_iter{i}.json")
        print("=" * 70)


if __name__ == "__main__":
    main()
