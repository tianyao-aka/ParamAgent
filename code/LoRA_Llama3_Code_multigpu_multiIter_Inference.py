#!/usr/bin/env python
"""
Inference script for iterative LoRA fine-tuned models.

Generates pitfalls using models from each training iteration.
Auto-detects iterations by scanning for iteration_*/merged_model directories.

Example usage:
python LoRA_Llama3_Code_multigpu_multiIter_Inference.py \
    --file_path benchmarks/humaneval.jsonl \
    --out_path ./outputs/pitfalls \
    --model_dir ./lora-llama3-8b-iterative \
    --num_samples 8 \
    --batch_size 4

Dependencies:
pip install "transformers>=4.42.0" bitsandbytes==0.43.2 accelerate tqdm
"""

from __future__ import annotations
import argparse
import gc
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# ----------------------------- Constants ----------------------------------- #

SYSTEM_PROMPT = (
    "You are an AI assistant in coding. "
    "Given a Python function signature and docstring, list potential pitfalls. "
)


# ----------------------------- Data Loading -------------------------------- #

def load_input_data(path: str) -> List[Dict[str, Any]]:
    """
    Load input data from JSON or JSONL file.
    Auto-detects format based on file extension.
    """
    path = Path(path)

    if path.suffix == ".jsonl":
        # JSONL format: one JSON object per line
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    else:
        # JSON format: single array
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list")
        return data


def save_jsonl(data: List[Dict[str, Any]], path: str) -> None:
    """Save data as JSONL file (one JSON object per line)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ----------------------------- Prompt Formatting --------------------------- #

def format_inference_prompt(func_signature: str) -> str:
    """
    Format prompt for inference.
    Uses the same format as training to ensure consistency.
    """
    user_block = (
        "[INST] <<SYS>> "
        f"{SYSTEM_PROMPT} <</SYS>>\n\n"
        f"FUNC_SIGNATURE:\n{func_signature.strip()}\n\n"
        "[/INST]"
    )
    return f"<s>{user_block}"


def get_func_signature(item: Dict[str, Any]) -> str:
    """
    Extract function signature from input item.
    Handles different field names: 'prompt', 'func_sign', 'signature'.
    """
    for key in ["prompt", "func_sign", "signature", "function"]:
        if key in item:
            return item[key]
    raise KeyError(f"Could not find function signature in item keys: {list(item.keys())}")


# ----------------------------- Model Loading ------------------------------- #

def load_model_for_inference(model_path: str) -> Tuple[Any, Any]:
    """
    Load model with 4-bit quantization for memory-efficient inference.
    """
    print(f"  Loading model from: {model_path}")

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Important for batch generation

    return model, tokenizer


def cleanup_memory() -> None:
    """Clean up GPU memory and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


# ----------------------------- Iteration Detection ------------------------- #

def detect_iterations(model_dir: str) -> List[Tuple[int, str]]:
    """
    Scan model_dir for iteration_*/merged_model directories.

    Returns:
        List of (iteration_number, model_path) tuples, sorted by iteration number
    """
    iterations = []

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    for d in os.listdir(model_dir):
        if d.startswith("iteration_"):
            merged_path = os.path.join(model_dir, d, "merged_model")
            if os.path.exists(merged_path) and os.path.isdir(merged_path):
                try:
                    iter_num = int(d.split("_")[1])
                    iterations.append((iter_num, merged_path))
                except (ValueError, IndexError):
                    print(f"  Warning: Could not parse iteration number from {d}")

    # Sort by iteration number
    iterations.sort(key=lambda x: x[0])

    if not iterations:
        raise ValueError(f"No iteration_*/merged_model directories found in {model_dir}")

    return iterations


# ----------------------------- Argument Parsing ---------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate pitfalls using iteratively fine-tuned models"
    )

    # Input/Output
    p.add_argument("--file_path", required=True,
                   help="Input JSON or JSONL file with function signatures")
    p.add_argument("--out_path", required=True,
                   help="Output directory for JSONL files")
    p.add_argument("--model_dir", required=True,
                   help="Directory containing iteration_*/merged_model subdirs")

    # Generation settings
    p.add_argument("--num_samples", type=int, default=8,
                   help="Number of pitfall samples per function (default: 8)")
    p.add_argument("--max_tokens", type=int, default=1024,
                   help="Max tokens per generation (default: 1024)")
    p.add_argument("--temperature", type=float, default=0.8,
                   help="Sampling temperature (default: 0.8)")

    # Other
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of samples: None or <=0 uses full data, >0 randomly samples N items")

    return p.parse_args()


# ----------------------------- Main ---------------------------------------- #

def main() -> None:
    args = parse_args()

    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 70)
    print("Iterative LoRA Model Inference")
    print("=" * 70)
    print(f"Input file: {args.file_path}")
    print(f"Output directory: {args.out_path}")
    print(f"Model directory: {args.model_dir}")
    print(f"Samples per function: {args.num_samples}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print("=" * 70)

    # Load input data
    print("\nLoading input data...")
    input_data = load_input_data(args.file_path)

    # Apply limit if specified (>0 means randomly sample N items)
    if args.limit is not None and args.limit > 0:
        if args.limit < len(input_data):
            random.shuffle(input_data)
            input_data = input_data[:args.limit]
        print(f"  Using {len(input_data)} samples (limited from original)")
    else:
        print(f"  Loaded {len(input_data)} samples (full dataset)")

    # Detect iterations
    print("\nDetecting iterations...")
    iterations = detect_iterations(args.model_dir)
    print(f"  Found {len(iterations)} iteration(s): {[i[0] for i in iterations]}")

    # Create output directory
    os.makedirs(args.out_path, exist_ok=True)

    # Process each iteration
    for iter_num, model_path in iterations:
        print(f"\n{'#' * 70}")
        print(f"# ITERATION {iter_num}")
        print(f"{'#' * 70}")

        # Load model
        model, tokenizer = load_model_for_inference(model_path)

        # Prepare prompts
        prompts = [format_inference_prompt(get_func_signature(item)) for item in input_data]

        # Generate pitfalls with progress bar
        print(f"\n  Generating pitfalls for {len(input_data)} functions...")
        print(f"  ({args.num_samples} samples each)")

        results = []
        for idx, item in enumerate(tqdm(input_data, desc=f"Iter {iter_num}")):
            prompt = prompts[idx]

            # Generate all samples at once using num_return_sequences
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    num_return_sequences=args.num_samples,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Decode all generated sequences
            prompt_len = inputs["input_ids"].shape[1]
            samples = []
            for i in range(args.num_samples):
                generated_ids = outputs[i][prompt_len:]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                samples.append(generated_text.strip())

            # Create output entry preserving all original fields
            result = {**item, "high_temp_pitfall": samples}
            results.append(result)

        # Save results
        output_file = os.path.join(args.out_path, f"iter_{iter_num:03d}.jsonl")
        save_jsonl(results, output_file)
        print(f"\n  Saved: {output_file}")

        # Cleanup
        del model, tokenizer
        cleanup_memory()

    # Summary
    print("\n" + "=" * 70)
    print("INFERENCE COMPLETE!")
    print("=" * 70)
    print(f"\nOutput files:")
    for iter_num, _ in iterations:
        print(f"  {args.out_path}/iter_{iter_num:03d}.jsonl")
    print("=" * 70)


if __name__ == "__main__":
    main()
