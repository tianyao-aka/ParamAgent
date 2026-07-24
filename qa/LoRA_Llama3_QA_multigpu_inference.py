#!/usr/bin/env python
"""
Multi-GPU inference for the LoRA-fine-tuned Meta-Llama-3.1-8B QA decomposition model.

- Single-GPU default: 4-bit load fits on one 48 GB RTX 6000 Ada.
- Multi-GPU (--num_gpus > 1): replicate the quantized model per GPU, split the
  JSON dataset evenly, run workers in parallel, then merge their outputs.

Example (4 GPUs data-parallel):
  CUDA_VISIBLE_DEVICES=0,1,2,3 python qa/LoRA_Llama3_QA_multigpu_inference.py \
    --base_model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --lora_path ./lora-llama3-8b-qa \
    --input_json benchmarks/light_hotpotqa.json \
    --output_json generated_reflections/qa/hotpotqa_insights_llama3_8b.jsonl \
    --num_gpus 4 --batch_size 2 --num_versions 10 --test

Notes
-----
- Prompt format matches the fine-tuning script (system + few-shots + question block).
- Multi-GPU mode uses data parallelism (one full model copy per GPU).
"""

from __future__ import annotations
import argparse
import os
import re
from typing import Iterable, List, Tuple

import torch
import json
import jsonlines
import torch.multiprocessing as mp
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SYS_PROMPT = (
    "You will be given a question.  "
    "Use your knowledge to extract the key point, the underlying intent, "
    "and possible inference patterns needed to answer the question."
)

PRE_INSIGHT_FEWSHOT = """
<Example 1>
q: Anatoly Maltsev and Valentin Turchin were both from Russia, which of the two is known for his work as a mathematician?

Question Parsing and Intent Extraction  
Intent:  
--------------------------------------------------------------------------------  
🔍 Key Components  
1. Entity A:  
- **Anatoly Maltsev** — mathematician and logician known for contributions in mathematical logic and abstract algebra  
2. Entity B:  
- **Valentin Turchin** — computer scientist and philosopher known for work in cybernetics and philosophy of science  
3. Implied Relationship:  
- Comparative inquiry: which individual is more closely associated with the domain of mathematics  
4. Answer Type Expected:  
- Person name (e.g., "Anatoly Maltsev")  
5. Reasoning Type:  
- Comparative factual reasoning  
6. Required Background:  
- Biographical knowledge or retrieved professional profiles  
--------------------------------------------------------------------------------  
🧠 Inference Trace  
1. Retrieve factual data about Maltsev and Turchin’s academic domains.  
2. Classify Maltsev as a mathematician based on core contributions to mathematical logic.  
3. Classify Turchin as mainly working in cybernetics and philosophy.  
4. Eliminate Turchin as primary mathematician.  
5. Conclude Maltsev is the individual known for mathematics.  
--------------------------------------------------------------------------------  
📝 Disambiguation Note  
- Nationality (Russia) does not help differentiate them.

<Example 2>
The Last Girl on Earth was the third concert tour by Barbadian recording artist Rihanna, the tour visited Europe, Asia, North America and Australia to support her fourth studio album, which 2009 , and fourth studio album by Barbadian singer Rihanna, and released on November 20, 2009 by Def Jam Recordings and SRP Records?

### Question Parsing and Intent Extraction

**Intent:**
--------------------------------------------------------------------------------
🔍 **Key Components**
1. **Entity A**:  
   - *The Last Girl on Earth* — Rihanna's third concert tour, associated with promoting a studio album  

2. **Entity B**:  
   - *Fourth studio album* by Rihanna — referenced multiple times, released in 2009  

3. **Key Relationship / Constraint**:  
   - Identify the name of Rihanna’s **fourth studio album**, which was released on **November 20, 2009**, and **supported by** her third concert tour, *The Last Girl on Earth*  

4. **Answer Type Expected**:  
   - Album title (e.g., *Rated R*)  

5. **Reasoning Type**:  
   - Factual entity retrieval based on event-album association and release date  

6. **Required Background**:  
   - Rihanna’s discography: album release dates and which albums were promoted during which concert tours  

--------------------------------------------------------------------------------

🧠 **Inference Trace**
- Determine the name of Rihanna’s **fourth studio album**  
- Confirm that this album was released on **November 20, 2009**  
- Verify that this album was the basis for the **"The Last Girl on Earth"** tour  
- Conclude that the album is **Rated R**

--------------------------------------------------------------------------------

📝 **Disambiguation Note**
- Although the question includes fragmented/redundant phrasing, the focus is clear: determine the album that matches both the **release date** and **tour association**
"""


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
    else:
        raise KeyError(
            f"Cannot generate sample ID: missing '{prompt_key}', 'task_id', or 'id' "
            f"in item: {list(item.keys())}"
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


def read_jsonl(path: str) -> List[dict]:
    """Read JSONL file and return list of items."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File `{path}` does not exist.")
    items: List[dict] = []
    with jsonlines.open(path) as reader:
        for item in reader:
            items.append(item)
    return items


def format_prompt(question: str) -> str:
    user_block = (
        "[INST] <<SYS>> "
        f"{SYS_PROMPT} <</SYS>>\n\n"
        f"Here are some examples:{PRE_INSIGHT_FEWSHOT}\n\n"
        f"[Question]: {question.strip()} [/INST]"
    )
    return f"<s>{user_block}"


def read_json_file(path: str) -> List[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File `{path}` does not exist.")
    if not path.endswith(".json"):
        raise ValueError(f"File `{path}` is not a json file.")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON list.")
    return data


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
        reply = text.split("[/INST]", 1)[-1].strip()
        reply = re.sub(r"\s*(?:</s>|\[\/INST\])\s*", "", reply)
        for marker in ("**Disambiguation Note**", "Disambiguation Note"):
            idx = reply.find(marker)
            if idx != -1:
                reply = reply[: idx - 4]
                break
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
    # Write as JSONL for temp files (allows incremental writing)
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
                if args.prompt_key not in item:
                    raise KeyError(f"Missing key '{args.prompt_key}' in sample: {item}")
                prompt_text = format_prompt(item[args.prompt_key])
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
                if args.output_key == "high_temp_insight" and "insight" not in item:
                    item["insight"] = gens[0] if gens else ""
                writer.write(item)  # Write immediately to JSONL

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


def merge_outputs(temp_files: List[str], final_output: str, prompt_key: str = "question") -> None:
    """
    Merge JSONL temp files with deduplication, write JSON final output.
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
            # Read JSONL temp files
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

    write_outputs(merged, final_output)


def write_outputs(items: List[dict], output_path: str) -> None:
    if output_path.endswith(".jsonl"):
        with jsonlines.open(output_path, mode="w") as writer:
            for item in items:
                writer.write(item)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)


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
    parser = argparse.ArgumentParser("LoRA Llama3-8B QA multi-GPU inference")
    parser.add_argument("--base_model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--lora_path", required=True, help="Path to LoRA adapter weights")
    parser.add_argument("--input_json", required=True, help="JSON file with a `question` field")
    parser.add_argument("--output_json", required=True, help="Where to write generations (.json or .jsonl)")
    parser.add_argument("--prompt_key", default="question", help="Key in JSON containing the question text")
    parser.add_argument("--output_key", default="insight", help="Key to store model output")
    parser.add_argument("--batch_size", type=int, default=1, help="#unique prompts per forward pass")
    parser.add_argument("--num_versions", type=int, default=1, help="Samples per prompt (uses multiple draws)")
    parser.add_argument("--max_prompt_len", type=int, default=1536)
    parser.add_argument("--max_new_tokens", type=int, default=800)
    parser.add_argument("--temperature", type=float, default=0.8)
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

    data = read_json_file(args.input_json)
    if args.test:
        data = data[:4]
        print(f"Test mode enabled — using first {len(data)} samples")
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)

    if args.num_gpus <= 1:
        # For single GPU, use temp JSONL file then convert to JSON
        base, _ = os.path.splitext(args.output_json)
        temp_jsonl = f"{base}_temp.jsonl"

        # Check for existing progress (single GPU)
        completed_ids = set()
        if not args.force_restart and os.path.exists(temp_jsonl):
            completed_ids = load_completed_samples(temp_jsonl, args.prompt_key)
            if completed_ids:
                original_count = len(data)
                data = filter_pending_samples(data, completed_ids, args.prompt_key)
                print(
                    f"Resuming from checkpoint. "
                    f"Already completed: {len(completed_ids)}/{original_count}. "
                    f"Remaining: {len(data)}"
                )

        if not data:
            # All done - convert temp to final if needed
            if os.path.exists(temp_jsonl):
                items = read_jsonl(temp_jsonl)
                write_outputs(items, args.output_json)
                os.remove(temp_jsonl)
                print(f"Converted checkpoint to final output: {args.output_json}")
            else:
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
            output_path=temp_jsonl,
            tqdm_desc="batches",
            resume_mode=bool(completed_ids),
        )

        # Convert JSONL checkpoint to requested final format.
        items = read_jsonl(temp_jsonl)
        write_outputs(items, args.output_json)

        # Clean up temp file
        if not args.keep_temp_files:
            os.remove(temp_jsonl)

        print(f"Done. Wrote results to {args.output_json}")
        return

    # Multi-GPU data-parallel path
    mp.set_start_method("spawn", force=True)

    # Generate temp file paths (use JSONL for checkpoint-friendly incremental writes)
    base, _ = os.path.splitext(args.output_json)
    temp_files: List[str] = [f"{base}_gpu{i}.jsonl" for i in range(args.num_gpus)]

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
        print(f"Validating input data with prompt_key='{args.prompt_key}'...")
        for i, item in enumerate(data):
            if args.prompt_key not in item:
                raise ValueError(
                    f"Sample {i} missing required key '{args.prompt_key}'. "
                    f"Available keys: {list(item.keys())}"
                )
        print("Input validation passed.")

    splits = split_dataset(data, args.num_gpus)
    processes: List[mp.Process] = []

    print(
        f"Running data-parallel across {args.num_gpus} GPUs — "
        f"total samples {len(data)}, batch_size={args.batch_size}, num_versions={args.num_versions}"
    )

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

    for p in processes:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker process failed with exit code {p.exitcode}")

    merge_outputs(temp_files, args.output_json, args.prompt_key)

    if not args.keep_temp_files:
        for path in temp_files:
            if os.path.exists(path):
                os.remove(path)
                print(f"Cleaned up temp file: {path}")

    print(f"Done. Wrote merged results to {args.output_json}")


if __name__ == "__main__":
    main()
