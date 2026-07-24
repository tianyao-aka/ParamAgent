"""
Generate pitfall analyses for HumanEval coding problems using Llama 3.1 8B.

This script processes all 164 HumanEval samples and generates 8 pitfall variations
per sample using TogetherAI's Llama 3.1 8B model with temperature 0.8.

Output: benchmarks/code_pitfalls/humaneval_full_pitfalls_llama3_untuned.jsonl
"""

import sys
from typing import List, Dict
from generators.model import Message, Llama3_1_8B
from utils import read_jsonl, write_jsonl
from gpt_usage import gpt_usage
from tqdm import tqdm


# System prompt for generating pitfalls (from LoRA_Llama3_Code_multigpu_inference.py)
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


def generate_pitfalls_for_sample(model: Llama3_1_8B, sample: Dict, num_variations: int = 8) -> Dict:
    """
    Generate pitfall analyses for a single HumanEval sample.

    Args:
        model: The Llama3_1_8B model instance
        sample: Dict with 'prompt' field and other HumanEval fields
        num_variations: Number of pitfall variations to generate (default: 8)

    Returns:
        Enhanced sample dict with 'high_temp_pitfall' field containing 8 variations
    """
    # Create messages with system prompt + user prompt
    messages = [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(role="user", content=sample["prompt"])
    ]

    # Generate 8 variations in ONE API call using num_comps=8
    # This is much more efficient than making 8 separate API calls
    pitfalls = model.generate_chat(
        messages=messages,
        temperature=0.8,
        max_tokens=1024,
        num_comps=num_variations
    )

    # Ensure pitfalls is a list (should already be from num_comps=8)
    if isinstance(pitfalls, str):
        pitfalls = [pitfalls]

    # Build result with all original fields plus new field
    result = {
        "task_id": sample["task_id"],
        "prompt": sample["prompt"],
        "entry_point": sample["entry_point"],
        "test": sample["test"],
        "canonical_solution": sample["canonical_solution"],
        "high_temp_pitfall": pitfalls  # List of 8 strings
    }

    return result


def main():
    """Main processing loop."""
    # Configuration
    INPUT_FILE = "benchmarks/humaneval_full.jsonl"
    OUTPUT_FILE = "benchmarks/code_pitfalls/humaneval_full_pitfalls_llama3_untuned.jsonl"
    NUM_VARIATIONS = 8

    print("="*70)
    print("HumanEval Pitfall Generation with Llama 3.1 8B")
    print("="*70)

    # Initialize model
    print("\nInitializing Llama 3.1 8B model...")
    model = Llama3_1_8B()
    print("Model initialized successfully")

    # Load dataset
    print(f"\nLoading dataset from {INPUT_FILE}...")
    dataset = read_jsonl(INPUT_FILE)
    print(f"Loaded {len(dataset)} samples")

    # Process samples with resume capability
    print(f"\nProcessing samples (generating {NUM_VARIATIONS} pitfall variations each)...")
    print(f"Output will be saved to: {OUTPUT_FILE}")
    print("Using upsert mode for automatic resume capability\n")

    processed_count = 0
    error_count = 0

    for idx, sample in enumerate(tqdm(dataset, desc="Generating pitfalls")):
        try:
            result = generate_pitfalls_for_sample(model, sample, NUM_VARIATIONS)

            # Incremental save with upsert for resume capability
            # If script crashes and restarts, already-processed samples will be updated, not duplicated
            write_jsonl(OUTPUT_FILE, [result], append=True, key="task_id")
            processed_count += 1

            # Progress logging every 10 samples
            if (idx + 1) % 10 == 0:
                usage = gpt_usage(backend="llama3_1_8b")
                print(f"\nProgress: {processed_count}/{len(dataset)} samples | Errors: {error_count}")
                print(f"API Usage: {usage}")

        except Exception as e:
            error_count += 1
            task_id = sample.get("task_id", f"index_{idx}")
            print(f"\nError processing sample {task_id}: {e}")
            # Continue processing remaining samples
            continue

    # Final summary
    print("\n" + "="*70)
    print("Processing Complete!")
    print("="*70)
    print(f"Successfully processed: {processed_count}/{len(dataset)} samples")
    print(f"Errors encountered: {error_count}")
    print(f"Output saved to: {OUTPUT_FILE}")

    final_usage = gpt_usage(backend="llama3_1_8b")
    print(f"\nFinal API Usage: {final_usage}")
    print("="*70)


if __name__ == "__main__":
    main()
