#!/usr/bin/env python3
"""
Generate HotpotQA question decompositions using Llama-3.1-8B via TogetherAI.

This script:
1. Loads existing hotpotQA_decomposition_dataset.json (30,000 entries)
2. Randomly selects 15,000 entries
3. Keeps the original question and level fields
4. Regenerates ONLY decomposition using Llama-3.1-8B-Instruct-Turbo
5. Saves to hotpotQA_decomposition_dataset_llama3.json
"""

import json
import random
import time
import argparse
import os
from typing import List, Dict
from tqdm import tqdm
from together import Together


# TogetherAI API configuration
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
LLAMA_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

_client = None


def get_together_client() -> Together:
    global _client
    if _client is None:
        api_key = TOGETHER_API_KEY or os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise RuntimeError("Set TOGETHER_API_KEY before running this script.")
        _client = Together(api_key=api_key)
    return _client


# ──────────────────────────── Prompts ────────────────────────────
# Keep the exact same prompts as gen_question_decomposition_data.py

PRE_INSIGHT_FEWSHOT = """
<Example 1>
q: Anatoly Maltsev and Valentin Turchin were both from Russia, which of the two is known for his work as a mathematician?

### Question Parsing and Intent Extraction

**Intent:**
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
1. Retrieve factual data about Maltsev and Turchin's academic domains.
2. Classify Maltsev as a mathematician based on core contributions to mathematical logic.
3. Classify Turchin as mainly working in cybernetics and philosophy.
4. Eliminate Turchin as primary mathematician.
5. Conclude Maltsev is the individual known for mathematics.
--------------------------------------------------------------------------------
📝 Disambiguation Note
- Nationality (Russia) does not help differentiate them.

<Example 2>
q: The Last Girl on Earth was the third concert tour by Barbadian recording artist Rihanna, the tour visited Europe, Asia, North America and Australia to support her fourth studio album, which was released on November 20, 2009, by Def Jam Recordings and SRP Records. What is the name of that fourth studio album?

### Question Parsing and Intent Extraction

**Intent:**
--------------------------------------------------------------------------------
🔍 **Key Components**
1. **Entity A**:
   - *The Last Girl on Earth* — Rihanna's third concert tour, associated with promoting a studio album

2. **Event**:
   - Release date **November 20, 2009** for the album

3. **Key Relationship**:
   - Identify Rihanna's **fourth studio album** released on that date and promoted by the tour

4. **Answer Type Expected**:
   - Album title (e.g., "Rated R")

5. **Reasoning Type**:
   - Factual retrieval from discography and tour association

6. **Required Background**:
   - Rihanna's discography and tour-album mapping
--------------------------------------------------------------------------------

🧠 **Inference Trace**
1. Locate Rihanna's albums around 2009.
2. Find the one released November 20, 2009.
3. Confirm it was promoted by "The Last Girl on Earth" tour.
4. Conclude the album is "Rated R".
--------------------------------------------------------------------------------
📝 **Disambiguation Note**
- Ignore redundant phrasing; focus on date & tour association.
"""

SYSTEM_PROMPT = (
    "You are an AI assistant for question parsing and intent extraction. "
    "Given a new question, produce a structured decomposition following "
    "the format shown in the examples."
)


# ──────────────────────────── LLM Call ────────────────────────────

def decompose_question_llama(question: str) -> str:
    """
    Call Llama-3.1-8B via TogetherAI to decompose `question` into Intent,
    Key Components, Inference Trace, etc., using the few-shot examples.

    Args:
        question: The HotpotQA question to decompose

    Returns:
        Structured decomposition string
    """
    user_prompt = (
        f"{PRE_INSIGHT_FEWSHOT}\n"
        f"q: {question}\n\n"
        "### Question Parsing and Intent Extraction"
    )

    try:
        resp = get_together_client().chat.completions.create(
            model=LLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,  # Increased from 0.1 for better Llama performance
            max_tokens=800    # Increased from 600 to prevent truncation
        )
        return resp.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error calling Llama API: {e}")
        raise


# ────────────────────────────── Main ──────────────────────────────

def process_hotpotqa_dataset(
    input_path: str,
    output_path: str,
    sample_size: int = 15000,
    limit: int = None,
    save_interval: int = 500,
    random_seed: int = 42
):
    """
    Load existing HotpotQA dataset, randomly sample entries, and regenerate
    decompositions using Llama-3.1-8B.

    Args:
        input_path: Path to existing hotpotQA_decomposition_dataset.json
        output_path: Path to save new dataset
        sample_size: Number of entries to randomly select (default: 15000)
        limit: Optional limit for testing (processes only first N from sample)
        save_interval: Save progress every N entries
        random_seed: Random seed for reproducibility
    """
    print(f"\n{'='*60}")
    print(f"HotpotQA Decomposition with Llama-3.1-8B")
    print(f"{'='*60}")
    print(f"Model: {LLAMA_MODEL}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Sample size: {sample_size}")
    if limit:
        print(f"Limit: {limit} entries (TEST MODE)")
    print(f"Save interval: Every {save_interval} entries")
    print(f"Random seed: {random_seed}")
    print(f"{'='*60}\n")

    # 1. Load existing dataset
    print("Loading existing dataset...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_entries = len(data)
    print(f"Loaded {total_entries} entries from {input_path}")

    # 2. Randomly sample entries
    random.seed(random_seed)
    if sample_size > total_entries:
        print(f"Warning: sample_size ({sample_size}) > total entries ({total_entries})")
        print(f"Using all {total_entries} entries")
        sampled_data = data
    else:
        print(f"Randomly sampling {sample_size} entries...")
        sampled_data = random.sample(data, sample_size)
        print(f"Sampled {len(sampled_data)} entries")

    # 3. Apply limit for testing
    if limit:
        sampled_data = sampled_data[:limit]
        print(f"Processing first {limit} entries for testing\n")

    # 4. Process entries
    results = []
    errors = 0

    print(f"Processing {len(sampled_data)} entries...\n")

    for i, entry in enumerate(tqdm(sampled_data, desc="Generating decompositions")):
        try:
            # Keep original fields
            question = entry["question"]
            level = entry["level"]

            # Regenerate decomposition using Llama
            decomposition = decompose_question_llama(question)

            # Create new entry
            results.append({
                "question": question,
                "decomposition": decomposition,
                "level": level
            })

            # Save intermediate results
            if (i + 1) % save_interval == 0:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"\n[Checkpoint] Saved {len(results)} entries to {output_path}")

            # Sleep to respect rate limits
            time.sleep(0.1)

        except Exception as e:
            print(f"\nError processing entry {i}: {e}")
            errors += 1
            time.sleep(1)
            continue

    # 5. Final save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Successfully processed: {len(results)}/{len(sampled_data)} entries")
    print(f"Errors: {errors}")
    print(f"Saved to: {output_path}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate HotpotQA decompositions using Llama-3.1-8B"
    )
    parser.add_argument(
        "--input_dataset",
        default="benchmarks/hotpotQA_decomposition_dataset.json",
        help="Path to existing HotpotQA dataset"
    )
    parser.add_argument(
        "--output_path",
        default="benchmarks/hotpotQA_decomposition_dataset_llama3.json",
        help="Output path for new dataset"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=15000,
        help="Number of entries to randomly select (default: 15000)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of entries to process (for testing)"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=500,
        help="Save progress every N entries (default: 500)"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    process_hotpotqa_dataset(
        input_path=args.input_dataset,
        output_path=args.output_path,
        sample_size=args.sample_size,
        limit=args.limit,
        save_interval=args.save_interval,
        random_seed=args.random_seed
    )


if __name__ == "__main__":
    main()
