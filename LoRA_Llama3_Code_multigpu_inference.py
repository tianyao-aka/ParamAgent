#!/usr/bin/env python
"""
Multi-GPU inference for the LoRA-fine-tuned Meta-Llama-3.x-32B coding model.

- Single-GPU default: 4-bit load fits on one 48 GB RTX 6000 Ada.
- Multi-GPU (--num_gpus > 1): replicate the quantized model per GPU, split the
  JSONL dataset evenly, run workers in parallel, then merge their outputs.

Example (4 GPUs data-parallel):
  CUDA_VISIBLE_DEVICES=4,5,6,7 python LoRA_Llama3_Code_multigpu_inference.py \
    --base_model ./DeepSeek-R1-Distill-Qwen-32B \
    --lora_path ./lora-dsQwen32b-code/checkpoint-156 \
    --input_jsonl benchmarks/humaneval_full.jsonl \
    --output_jsonl benchmarks/code_pitfalls/HE_pitfalls_ds_32b_v2.jsonl \
    --num_gpus 4 --batch_size 1 --num_versions 1 --test

Notes
-----
- 4-bit load keeps memory per GPU low; a single 48 GB card can serve batches of
  1–2 prompts. Multi-GPU here uses data parallelism (one full model copy/GPU).
- Prompt format matches the fine-tuning script (system + FUNC_SIGNATURE block).
"""

from __future__ import annotations
import argparse
import os
import re
from typing import Iterable, List, Tuple

import torch
import jsonlines
import torch.multiprocessing as mp
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SYSTEM_PROMPT = (
    "You are an AI assistant in coding. "
    "Given a Python function signature and docstring, list potential pitfalls.\n\n"
    "[Example]\n"
    "def longest_subarray_with_sum_limit(nums: List[int], target: int) -> List[int]:\n"
    "    \"\"\"\n"
    "    Return the longest **contiguous** subarray of `nums` whose elements sum to at most `target`.\n"
    "\n"
    "    • If several subarrays tie for maximum length, return the **left‑most**.\n"
    "    • If no valid subarray exists, return the empty list `[]`.\n"
    "    • The input may contain negative as well as positive integers.\n"
    "\n"
    "    Complexity requirements: time O(n), auxiliary space O(1).\n"
    "    \"\"\"\n"
    "\n"
    "[Pitfalls]:\n"
    "1. **No‑solution case** — must return `[]`, not `[x]` or `None`.\n"
    "2. **Length update rule** — use strictly greater (`>`); otherwise, a later equal‑length window overwrites the earlier left‑most one.\n"
    "3. **Negatives in the window** — shrinking only while `current_sum > target` can leave an over‑target sum if later negatives cancel it.\n"
    "\n"
    "Now, list potential pitfalls for the following question:"
)


# ============================================================================
# Checkpoint/Resume Helper Functions
# ============================================================================

def get_sample_id(item: dict, prompt_key: str) -> str:
    """
    Generate unique identifier for a sample.

    Priority order:
    1. task_id field (HumanEval, MBPP datasets)
    2. id field (generic datasets)
    3. SHA256 hash of prompt content (fallback)

    Returns 16-character identifier.
    """
    import hashlib

    # Try task_id first (HumanEval, MBPP)
    if "task_id" in item:
        return str(item["task_id"])

    # Try generic id field
    if "id" in item:
        return str(item["id"])

    # Fallback: hash the prompt content
    if prompt_key in item:
        prompt_content = item[prompt_key]
    elif "question_title" in item and "question_content" in item:
        prompt_content = f"{item['question_title']}\n\n{item['question_content']}"
    else:
        raise KeyError(
            f"Cannot generate sample ID: missing '{prompt_key}', 'task_id', 'id', "
            f"or 'question_title'+'question_content' in item: {list(item.keys())}"
        )

    return hashlib.sha256(prompt_content.encode('utf-8')).hexdigest()[:16]


def load_completed_samples(temp_file: str, prompt_key: str) -> set:
    """
    Read temp file and extract IDs of all completed samples.
    Handles corrupted lines gracefully.

    Returns set of completed sample IDs.
    """
    completed = set()

    if not os.path.exists(temp_file):
        return completed

    try:
        with jsonlines.open(temp_file, mode='r') as reader:
            for item in reader:
                try:
                    sample_id = get_sample_id(item, prompt_key)
                    completed.add(sample_id)
                except (KeyError, jsonlines.InvalidLineError) as e:
                    print(f"Warning: Skipping corrupted line in {temp_file}: {e}")
                    continue
    except Exception as e:
        print(f"Warning: Error reading {temp_file}: {e}. Treating as empty checkpoint.")
        return set()

    return completed


def filter_pending_samples(data: List[dict], completed: set, prompt_key: str) -> List[dict]:
    """
    Filter out already-processed samples from dataset.
    Returns only samples that need processing.
    """
    pending = []
    for item in data:
        try:
            sample_id = get_sample_id(item, prompt_key)
            if sample_id not in completed:
                pending.append(item)
        except KeyError as e:
            print(f"Warning: Cannot get sample ID, including in pending: {e}")
            pending.append(item)

    return pending


def format_prompt(func_sign: str) -> str:
    user_block = (
        "[INST] <<SYS>> "
        f"{SYSTEM_PROMPT} <</SYS>>\n\n"
        f"FUNC_SIGNATURE:\n{func_sign.strip()}\n\n"
        "[/INST]"
    )
    return f"<s>{user_block}"


def read_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File `{path}` does not exist.")
    if not path.endswith(".jsonl"):
        raise ValueError(f"File `{path}` is not a jsonl file.")
    items: List[dict] = []
    with jsonlines.open(path) as reader:
        for item in reader:
            items.append(item)
    return items


def chunked(seq: List[dict], n: int) -> Iterable[List[dict]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def split_dataset(data: List[dict], num_parts: int) -> List[List[dict]]:
    if num_parts <= 1:
        return [data]
    chunk_size = (len(data) + num_parts - 1) // num_parts
    return [data[i * chunk_size : (i + 1) * chunk_size] for i in range(num_parts)]


def infer_input_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "hf_device_map"):
        for dev in model.hf_device_map.values():
            if isinstance(dev, str):
                return torch.device(dev)
            if isinstance(dev, int):
                return torch.device(f"cuda:{dev}")
    if hasattr(model, "device"):
        return torch.device(model.device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(
    base_model: str,
    lora_path: str,
    num_gpus: int,
    max_gpu_mem: str,
    device_override: int | None = None,
) -> Tuple[AutoTokenizer, PeftModel, torch.device]:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if device_override is not None:
        device_map = {"": device_override}
        max_memory = None
    elif num_gpus > 1:
        device_map = "auto"
        max_memory = {f"cuda:{i}": max_gpu_mem for i in range(num_gpus)}
    else:
        device_map = {"": 0}
        max_memory = None

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        max_memory=max_memory,
    )

    model = PeftModel.from_pretrained(base, lora_path, torch_dtype=torch.bfloat16)
    model.eval()
    input_device = infer_input_device(model)
    model.config.use_cache = True
    return tokenizer, model, input_device


def generate_batch(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    input_device: torch.device,
    max_prompt_len: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> List[str]:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_len,
    )
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    cleaned: List[str] = []
    for text in decoded:
        # Extract reply after instruction marker
        reply = text.split("[/INST]", 1)[-1].strip()

        # Remove special tokens (ALWAYS do this first)
        reply = re.sub(r"</s>|<s>|\[/?INST\]", "", reply).strip()

        # Then cut at last code fence if present
        fence_pos = reply.rfind("```")
        if fence_pos != -1:
            reply = reply[: fence_pos + 3]

        cleaned.append(reply.strip())
    return cleaned


def process_data(
    data: List[dict],
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
    model: PeftModel,
    input_device: torch.device,
    output_path: str,
    tqdm_desc: str,
    resume_mode: bool = False,
) -> None:
    num_batches = (len(data) + args.batch_size - 1) // args.batch_size

    # Use append mode if resuming, write mode otherwise
    file_mode = "a" if resume_mode else "w"

    with jsonlines.open(output_path, mode=file_mode) as writer:
        for chunk in tqdm(
            chunked(data, args.batch_size),
            total=num_batches,
            desc=tqdm_desc,
        ):
            prompts: List[str] = []
            owners: List[int] = []
            for idx, item in enumerate(chunk):
                # Try to get prompt from prompt_key first (backward compatibility)
                if args.prompt_key in item:
                    prompt_content = item[args.prompt_key]
                # Otherwise, concatenate question_title and question_content
                elif "question_title" in item and "question_content" in item:
                    prompt_content = f"{item['question_title']}\n\n{item['question_content']}"
                else:
                    raise KeyError(
                        f"Missing key '{args.prompt_key}' and could not find 'question_title' + "
                        f"'question_content' in sample: {item}"
                    )
                prompt_text = format_prompt(prompt_content)
                for _ in range(args.num_versions):
                    prompts.append(prompt_text)
                    owners.append(idx)

            outputs = generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                input_device=input_device,
                max_prompt_len=args.max_prompt_len,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
            )

            grouped: List[List[str]] = [[] for _ in chunk]
            for owner_idx, text in zip(owners, outputs):
                grouped[owner_idx].append(text)

            for item, gens in zip(chunk, grouped):
                item[args.output_key] = gens if args.num_versions > 1 else gens[0]
                writer.write(item)

            # Flush and fsync after each batch for atomic checkpointing
            writer._fp.flush()
            os.fsync(writer._fp.fileno())


def worker_entry(
    gpu_id: int,
    data_subset: List[dict],
    args: argparse.Namespace,
    temp_output: str,
) -> None:
    # Check for existing progress
    completed_ids = load_completed_samples(temp_output, args.prompt_key)

    if completed_ids:
        original_count = len(data_subset)
        data_subset = filter_pending_samples(data_subset, completed_ids, args.prompt_key)
        print(
            f"GPU {gpu_id}: Resuming from checkpoint. "
            f"Already completed: {len(completed_ids)}/{original_count}. "
            f"Remaining: {len(data_subset)}"
        )

    # Skip if all done
    if not data_subset:
        print(f"GPU {gpu_id}: All samples already completed. Skipping.")
        return

    # Load model
    tokenizer, model, input_device = load_model(
        base_model=args.base_model,
        lora_path=args.lora_path,
        num_gpus=1,
        max_gpu_mem=args.max_gpu_mem,
        device_override=gpu_id,
    )

    # Process remaining data
    process_data(
        data=data_subset,
        args=args,
        tokenizer=tokenizer,
        model=model,
        input_device=input_device,
        output_path=temp_output,
        tqdm_desc=f"gpu{gpu_id}",
        resume_mode=bool(completed_ids),
    )


def merge_outputs(temp_files: List[str], final_output: str, prompt_key: str = "prompt") -> None:
    """
    Merge temp files with deduplication.
    Uses sample IDs to ensure each sample appears exactly once.
    """
    seen_ids = set()
    merged: List[dict] = []
    duplicate_count = 0

    for temp_file in temp_files:
        if not os.path.exists(temp_file):
            print(f"Warning: Expected temp file {temp_file} not found. Skipping.")
            continue

        try:
            items = read_jsonl(temp_file)
            for item in items:
                try:
                    sample_id = get_sample_id(item, prompt_key)

                    if sample_id in seen_ids:
                        duplicate_count += 1
                        print(f"Warning: Duplicate sample {sample_id} found in {temp_file}. Skipping.")
                        continue

                    seen_ids.add(sample_id)
                    merged.append(item)
                except KeyError as e:
                    print(f"Warning: Cannot get sample ID from item in {temp_file}: {e}. Including anyway.")
                    merged.append(item)
        except Exception as e:
            print(f"Error reading {temp_file}: {e}")
            raise RuntimeError(f"Failed to merge outputs from {temp_file}")

    if duplicate_count > 0:
        print(f"Removed {duplicate_count} duplicate samples during merge.")

    print(f"Merged {len(merged)} unique samples from {len(temp_files)} temp files.")

    # Write final output
    with jsonlines.open(final_output, mode="w") as writer:
        for item in merged:
            writer.write(item)


def check_existing_progress(temp_files: List[str], prompt_key: str) -> dict:
    """
    Check for existing temp files and report progress.
    Returns dict mapping temp_file -> (completed_count, exists).
    """
    progress = {}
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            completed = load_completed_samples(temp_file, prompt_key)
            progress[temp_file] = (len(completed), True)
        else:
            progress[temp_file] = (0, False)
    return progress


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("LoRA Llama3-32B multi-GPU inference")
    parser.add_argument("--base_model", default="meta-llama/Meta-Llama-3.1-32B-Instruct")
    parser.add_argument("--lora_path", required=True, help="Path to LoRA adapter weights")
    parser.add_argument("--input_jsonl", required=True, help="JSONL with a `prompt` field")
    parser.add_argument("--output_jsonl", required=True, help="Where to write generations")
    parser.add_argument("--prompt_key", default="prompt", help="Key in JSONL containing the prompt text. If not found, will use question_title + question_content")
    parser.add_argument("--output_key", default="pitfall", help="Key to store model output")
    parser.add_argument("--batch_size", type=int, default=1, help="#unique prompts per forward pass")
    parser.add_argument("--num_versions", type=int, default=1, help="Samples per prompt (uses multiple draws)")
    parser.add_argument("--max_prompt_len", type=int, default=1536)
    parser.add_argument("--max_new_tokens", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--num_gpus", type=int, default=1, help="Replicate across this many GPUs")
    parser.add_argument("--max_gpu_mem", default="46GiB", help="Per-GPU budget if using HF sharding")
    parser.add_argument("--test", action="store_true", help="Test mode: only run on 4 samples")

    # Checkpoint/resume arguments
    parser.add_argument(
        "--force_restart",
        action="store_true",
        help="Ignore existing checkpoints and restart from scratch. Will overwrite temp files."
    )
    parser.add_argument(
        "--validate_input",
        action="store_true",
        help="Validate that input file hasn't changed since checkpoint creation. "
             "Raises error if input is newer than checkpoint."
    )
    parser.add_argument(
        "--keep_temp_files",
        action="store_true",
        help="Keep temporary GPU output files after merge (for debugging)."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_gpus > torch.cuda.device_count():
        raise ValueError(
            f"Requested {args.num_gpus} GPUs but only {torch.cuda.device_count()} visible"
        )

    data = read_jsonl(args.input_jsonl)
    if args.test:
        data = data[:4]
        print(f"Test mode enabled — using first {len(data)} samples")
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)

    if args.num_gpus <= 1:
        # Check for existing progress (single GPU)
        completed_ids = set()
        if not args.force_restart and os.path.exists(args.output_jsonl):
            completed_ids = load_completed_samples(args.output_jsonl, args.prompt_key)
            if completed_ids:
                original_count = len(data)
                data = filter_pending_samples(data, completed_ids, args.prompt_key)
                print(
                    f"Resuming from checkpoint. "
                    f"Already completed: {len(completed_ids)}/{original_count}. "
                    f"Remaining: {len(data)}"
                )

        if not data:
            print("All samples already completed. Nothing to do.")
            return

        tokenizer, model, input_device = load_model(
            base_model=args.base_model,
            lora_path=args.lora_path,
            num_gpus=1,
            max_gpu_mem=args.max_gpu_mem,
        )

        print(
            f"Loaded {len(data)} samples | batch_size={args.batch_size}, "
            f"num_versions={args.num_versions}, device={input_device}"
        )
        process_data(
            data=data,
            args=args,
            tokenizer=tokenizer,
            model=model,
            input_device=input_device,
            output_path=args.output_jsonl,
            tqdm_desc="batches",
            resume_mode=bool(completed_ids),
        )
        print(f"Done. Wrote results to {args.output_jsonl}")
        return

    # Multi-GPU data-parallel path
    mp.set_start_method("spawn", force=True)
    splits = split_dataset(data, args.num_gpus)
    temp_files: List[str] = []
    processes: List[mp.Process] = []

    # Generate temp file paths
    base, ext = os.path.splitext(args.output_jsonl)
    for gpu_id in range(args.num_gpus):
        temp_path = f"{base}_gpu{gpu_id}{ext or '.jsonl'}"
        temp_files.append(temp_path)

    # Check for existing progress and display resume info
    if not args.force_restart:
        progress = check_existing_progress(temp_files, args.prompt_key)
        total_completed = sum(count for count, _ in progress.values())
        any_existing = any(exists for _, exists in progress.values())

        if any_existing:
            print(f"\n{'='*60}")
            print(f"RESUME MODE DETECTED")
            print(f"{'='*60}")
            print(f"Found existing checkpoint files:")
            for temp_file, (count, exists) in progress.items():
                if exists:
                    print(f"  {temp_file}: {count} samples completed")
            print(f"Total completed: {total_completed}/{len(data)}")
            print(f"Resuming processing...\n")
    else:
        # Force restart: delete existing temp files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"Removed checkpoint: {temp_file}")

    # Validate input if requested
    if args.validate_input:
        input_mtime = os.path.getmtime(args.input_jsonl)
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                temp_mtime = os.path.getmtime(temp_file)
                if input_mtime > temp_mtime:
                    raise ValueError(
                        f"Input file {args.input_jsonl} has been modified after checkpoint "
                        f"{temp_file} was created. Use --force_restart to ignore checkpoints."
                    )

    print(
        f"Running data-parallel across {args.num_gpus} GPUs — "
        f"total samples {len(data)}, batch_size={args.batch_size}, num_versions={args.num_versions}"
    )

    # Spawn workers
    for gpu_id, subset in enumerate(splits):
        if not subset:
            continue
        temp_path = temp_files[gpu_id]
        p = mp.Process(
            target=worker_entry,
            args=(gpu_id, subset, args, temp_path),
        )
        p.start()
        processes.append(p)

    # Wait for completion
    for p in processes:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker process failed with exit code {p.exitcode}")

    # Merge outputs with deduplication
    merge_outputs(temp_files, args.output_jsonl, args.prompt_key)

    # Cleanup temp files unless --keep_temp_files
    if not args.keep_temp_files:
        for path in temp_files:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"Warning: Could not delete temp file {path}: {e}")
    else:
        print(f"Kept temp files for inspection: {', '.join(temp_files)}")

    print(f"Done. Wrote merged results to {args.output_jsonl}")


if __name__ == "__main__":
    main()
