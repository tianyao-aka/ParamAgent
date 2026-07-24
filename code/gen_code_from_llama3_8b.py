#!/usr/bin/env python3
"""
Generate pitfalls and flawed implementations using Llama-3.1-8B via TogetherAI.

This script:
1. Loads existing datasets (APP_code_datasets.json and augmented_coding_datasets.json)
2. Keeps the original func_sign and other fields
3. Regenerates ONLY pitfalls and flawed_impl using Llama-3.1-8B-Instruct-Turbo
4. Saves to new files with llama3_8b suffix
"""

import json
import time
import argparse
import os
from typing import List, Dict, Tuple
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


# System prompt for pitfall and flawed implementation generation
system_prompt = (
    "You are an AI assistant for Python coding. Given a function signature and docstring, "
    "use your knowledge to propose potential pitfalls for the implementation, "
    "and list the possible pitfalls, and generate up to 6 flawed implementations "
    "specific to the function signature that cover as many pitfalls as possible. "
    "Use <Pitfalls> and <Flawed Implementations> before pitfalls and implementations."
)


# Few-shot example
few_shot = """Example:
[Function Signature]:
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\"Check if any two numbers in the list are closer than the threshold.\"\"\"

<Pitfalls>:
1. **Empty or Single-Element Lists** must return `False`, not `True`.
2. **Duplicate Values** must be compared (difference 0), so never drop duplicates.
3. Always use **absolute difference** (`abs(a - b)`), not raw subtraction.
4. Use the correct **strictness** (`< threshold`, not `<=`).
5. Ensure you don't **exit too early**—check all distinct pairs.

[Flawed Implementations]:

```python
def has_close_elements_v1(numbers: List[float], threshold: float) -> bool:
    # BUG: returns True for empty or single-element lists
    if len(numbers) < 2:
        return True
    for i in range(len(numbers)-1):
        for j in range(i+1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False

def has_close_elements_v2(numbers: List[float], threshold: float) -> bool:
    # BUG: removes duplicates, so identical values never compared
    numbers = sorted(set(numbers))
    for i in range(len(numbers)-1):
        if abs(numbers[i+1] - numbers[i]) < threshold:
            return True
    return False

def has_close_elements_v3(numbers: List[float], threshold: float) -> bool:
    # BUG: uses raw subtraction instead of abs()
    for i in range(len(numbers)-1):
        for j in range(i+1, len(numbers)):
            if (numbers[i] - numbers[j]) < threshold:
                return True
    return False

def has_close_elements_v4(numbers: List[float], threshold: float) -> bool:
    # BUG: uses <= instead of <, misclassifies exactly-threshold pairs
    for i in range(len(numbers)-1):
        for j in range(i+1, len(numbers)):
            if abs(numbers[i] - numbers[j]) <= threshold:
                return True
    return False

def has_close_elements_v5(numbers: List[float], threshold: float) -> bool:
    # BUG: breaks out of outer loop too soon
    for i in range(len(numbers)-1):
        for j in range(i+1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
            break   # <-- this break prevents checking all j for each i
    return False
```"""


def extract_pitfalls_and_flawed_llama(func_block: str) -> Tuple[str, str, str]:
    """
    Generate pitfalls and flawed implementations using Llama-3.1-8B.

    Args:
        func_block: Function signature with docstring

    Returns:
        Tuple of (raw_text, pitfalls, flawed_impl)
    """
    user_prompt = f"{few_shot}\n\n{func_block}"

    try:
        resp = get_together_client().chat.completions.create(
            model=LLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            max_tokens=812
        )

        text = resp.choices[0].message.content

        # Parse the response to extract pitfalls and flawed implementations
        # Handle both <> and [] bracket styles
        split_markers = [
            "<Flawed Implementations>",
            "[Flawed Implementations]:",
            "[Flawed Implementations]"
        ]

        pits = text
        flawed = ""

        for marker in split_markers:
            if marker in text:
                pits, flawed = text.split(marker, 1)
                pits = pits.replace("<Pitfalls>:", "<Pitfalls>:").strip()
                flawed = flawed.strip()
                # Add back the marker prefix for consistency
                if flawed and not flawed.startswith(":"):
                    flawed = ":\n\n" + flawed
                break

        return text, pits, flawed

    except Exception as e:
        print(f"Error calling Llama API: {e}")
        raise


def process_app_code_datasets(input_path: str, output_path: str, limit: int = None):
    """
    Process APP_code_datasets.json - keep original fields, regenerate pitfalls/flawed_impl.

    Args:
        input_path: Path to original APP_code_datasets.json
        output_path: Path to save llama3_8b version
        limit: Optional limit on number of entries to process (for testing)
    """
    print(f"\n{'='*60}")
    print(f"Processing APP_code_datasets.json")
    print(f"{'='*60}\n")

    # Load existing dataset
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if limit:
        data = data[:limit]
        print(f"Processing first {limit} entries (test mode)\n")

    processed = []
    errors = 0

    for idx, entry in enumerate(tqdm(data, desc="Generating with Llama-3.1-8B")):
        try:
            # Keep original fields
            question = entry["question"]
            func_sign = entry["func_sign"]
            difficulty = entry["difficulty"]

            # Regenerate pitfalls and flawed_impl using Llama
            raw_text, pitfalls, flawed_impl = extract_pitfalls_and_flawed_llama(func_sign)

            # Create new entry
            processed.append({
                "question": question,
                "func_sign": func_sign,
                "difficulty": difficulty,
                "pitfalls": pitfalls,
                "flawed_impl": flawed_impl,
                "raw_text": raw_text
            })

            # Sleep to respect rate limits
            time.sleep(0.1)

        except Exception as e:
            print(f"\nError processing entry {idx}: {e}")
            errors += 1
            time.sleep(1)
            continue

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"APP_code_datasets processing complete!")
    print(f"Successfully processed: {len(processed)}/{len(data)} entries")
    print(f"Errors: {errors}")
    print(f"Saved to: {output_path}")
    print(f"{'='*60}\n")


def process_augmented_coding_datasets(input_path: str, output_path: str, limit: int = None):
    """
    Process augmented_coding_datasets.json - parse strings, generate pitfalls/flawed_impl.

    Args:
        input_path: Path to original augmented_coding_datasets.json
        output_path: Path to save llama3_8b version
        limit: Optional limit on number of entries to process (for testing)
    """
    print(f"\n{'='*60}")
    print(f"Processing augmented_coding_datasets.json")
    print(f"{'='*60}\n")

    # Load existing dataset
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if limit:
        data = data[:limit]
        print(f"Processing first {limit} entries (test mode)\n")

    processed = []
    errors = 0

    for idx, entry_str in enumerate(tqdm(data, desc="Generating with Llama-3.1-8B")):
        try:
            # Parse the string to extract func_sign and docstring
            # Format: "func_sign: <signature>\ndocstring: '<docstring>'"
            parts = entry_str.split("\ndocstring: ", 1)
            if len(parts) != 2:
                print(f"\nWarning: Could not parse entry {idx}, skipping")
                continue

            func_sign_part = parts[0].replace("func_sign: ", "").strip()
            docstring_part = parts[1].strip().strip("'\"")

            # Reconstruct function signature with docstring
            func_block = f"```python\n{func_sign_part}\n    \"\"\"{docstring_part}\"\"\"\n```"

            # Generate pitfalls and flawed_impl using Llama
            raw_text, pitfalls, flawed_impl = extract_pitfalls_and_flawed_llama(func_block)

            # Create new entry (similar structure to APP_code_datasets)
            processed.append({
                "func_sign": func_block,
                "pitfalls": pitfalls,
                "flawed_impl": flawed_impl,
                "raw_text": raw_text
            })

            # Sleep to respect rate limits
            time.sleep(0.1)

        except Exception as e:
            print(f"\nError processing entry {idx}: {e}")
            errors += 1
            time.sleep(1)
            continue

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"augmented_coding_datasets processing complete!")
    print(f"Successfully processed: {len(processed)}/{len(data)} entries")
    print(f"Errors: {errors}")
    print(f"Saved to: {output_path}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate pitfalls and flawed implementations using Llama-3.1-8B"
    )
    parser.add_argument(
        "--dataset",
        choices=["app", "augmented", "both"],
        default="both",
        help="Which dataset to process (default: both)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of entries to process (for testing)"
    )
    parser.add_argument(
        "--app_input",
        default="benchmarks/APP_code_datasets.json",
        help="Path to APP_code_datasets.json"
    )
    parser.add_argument(
        "--app_output",
        default="benchmarks/APP_code_datasets_llama3_8b.json",
        help="Output path for APP dataset"
    )
    parser.add_argument(
        "--aug_input",
        default="benchmarks/augmented_coding_datasets.json",
        help="Path to augmented_coding_datasets.json"
    )
    parser.add_argument(
        "--aug_output",
        default="benchmarks/augmented_coding_datasets_llama3_8b.json",
        help="Output path for augmented dataset"
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("Llama-3.1-8B Dataset Generation Script")
    print("="*60)
    print(f"Model: {LLAMA_MODEL}")
    print(f"Dataset(s): {args.dataset}")
    if args.limit:
        print(f"Limit: {args.limit} entries (TEST MODE)")
    print("="*60)

    # Process datasets based on selection
    if args.dataset in ["app", "both"]:
        process_app_code_datasets(args.app_input, args.app_output, args.limit)

    if args.dataset in ["augmented", "both"]:
        process_augmented_coding_datasets(args.aug_input, args.aug_output, args.limit)

    print("\n" + "="*60)
    print("All processing complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
