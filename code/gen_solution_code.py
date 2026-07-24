import json
import time
import openai
import argparse
import os
from typing import List, Dict
from tqdm import tqdm


def parse_sample_string(sample_str: str) -> dict:
    """
    Parse a sample string of the form:
      func_sign: <function signature>
      docstring: '<docstring text>'
    into a dict with keys "func_sign" and "docstring".
    """
    result = {}
    for line in sample_str.splitlines():
        line = line.strip()
        if line.startswith("func_sign:"):
            # extract everything after 'func_sign:'
            result["func_sign"] = line[len("func_sign:"):].strip()
        elif line.startswith("docstring:"):
            # extract after 'docstring:', then strip matching quotes
            ds = line[len("docstring:"):].strip()
            if (ds.startswith("'") and ds.endswith("'")) or (ds.startswith('"') and ds.endswith('"')):
                ds = ds[1:-1]
            result["docstring"] = ds
    return result


# System prompt for solution generation
system_prompt = (
    "You are an expert Python programmer. Given a function signature with its docstring, "
    "and optionally a problem description, provide a correct, efficient, and robust implementation. "
    "\n\n"
    "Your response should follow this structure:\n"
    "1. First, provide CONCISE step-by-step thinking about the approach (2-5 key points)\n"
    "2. Briefly mention any common pitfalls to avoid (1-2 sentences, no code examples)\n"
    "3. Use the marker '<Solution>' on its own line\n"
    "4. Provide the complete, correct Python function implementation\n"
    "\n"
    "Keep the thinking concise and to the point. Focus on the correct solution."
)


def generate_solution(question: str, func_sign: str, model: str = "gpt-4o-mini") -> str:
    """
    Call GPT-4o-mini to generate a step-by-step solution with thinking and correct implementation.

    Args:
        question: The problem description (can be empty for augmented_coding dataset)
        func_sign: The function signature with docstring
        model: The OpenAI model to use

    Returns:
        The complete response including thinking and solution
    """
    # Build the user prompt
    if question:
        user_prompt = f"Problem Description:\n{question}\n\n{func_sign}"
    else:
        user_prompt = func_sign

    try:
        # Call the LLM
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=1800,
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"Error generating solution: {e}")
        return f"Error: {str(e)}"


def process_app_dataset(
    json_path: str,
    model: str = "gpt-4o-mini",
    batch_save_every: int = 100,
    skip_existing: bool = True
):
    """
    Process APP_code_datasets.json and add 'gpt_solutions' key in-place.

    Args:
        json_path: Path to APP_code_datasets.json
        model: The OpenAI model to use
        batch_save_every: Save progress every N items
        skip_existing: Skip samples that already have 'gpt_solutions' key
    """
    print(f"Loading {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Found {len(data)} samples in APP dataset")

    processed_count = 0
    for i, sample in enumerate(tqdm(data, desc="Processing APP dataset")):
        # Skip if already has solution and skip_existing is True
        if skip_existing and "gpt_solutions" in sample and sample["gpt_solutions"]:
            continue

        # Extract question and func_sign
        question = sample.get("question", "")
        func_sign = sample.get("func_sign", "")

        if not func_sign:
            print(f"Warning: Sample {i} has no func_sign, skipping")
            continue

        # Generate solution
        solution = generate_solution(question, func_sign, model)
        sample["gpt_solutions"] = solution
        processed_count += 1

        # Periodic saving
        if processed_count > 0 and processed_count % batch_save_every == 0:
            print(f"\nProcessed {processed_count} new samples, saving intermediate file...")
            with open(json_path, "w", encoding="utf-8") as out_f:
                json.dump(data, out_f, ensure_ascii=False, indent=2)
            print(f"Intermediate file saved to {json_path}")

        # Rate limiting
        time.sleep(0.1)

    # Final save
    print(f"\nSaving final results...")
    with open(json_path, "w", encoding="utf-8") as out_f:
        json.dump(data, out_f, ensure_ascii=False, indent=2)
    print(f"Final output saved to {json_path}")
    print(f"Processed {processed_count} new samples (skipped {len(data) - processed_count} existing)")


def process_augmented_dataset(
    json_path: str,
    output_path: str,
    model: str = "gpt-4o-mini",
    batch_save_every: int = 100
):
    """
    Process augmented_coding_datasets.json and create a new output file with solutions.

    Args:
        json_path: Path to augmented_coding_datasets.json
        output_path: Path to save augmented_coding_datasets_solutions.json
        model: The OpenAI model to use
        batch_save_every: Save progress every N items
    """
    print(f"Loading {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Found {len(data)} samples in augmented dataset")

    # Check for existing output to resume processing
    if os.path.exists(output_path):
        print(f"Found existing output file: {output_path}")
        with open(output_path, "r", encoding="utf-8") as f:
            enriched = json.load(f)
        start_index = len(enriched)
        print(f"Resuming from sample {start_index} (found {start_index} existing samples)")
        print(f"Remaining samples to process: {len(data) - start_index}")
    else:
        enriched = []
        start_index = 0
        print(f"Starting fresh processing of all {len(data)} samples")

    for i in tqdm(range(start_index, len(data)), desc="Processing augmented dataset", initial=start_index, total=len(data)):
        sample_str = data[i]
        # Parse the sample string
        sample = parse_sample_string(sample_str)

        func_sign = sample.get("func_sign", "").strip()
        docstring = sample.get("docstring", "").strip()

        if not func_sign:
            print(f"Warning: Sample {i} has no func_sign, skipping")
            enriched.append(sample)
            continue

        # Build function signature block with docstring
        signature_block = f"{func_sign}\n    \"\"\"{docstring}\"\"\""

        # Generate solution (no question for augmented dataset)
        solution = generate_solution("", signature_block, model)

        # Add solution to sample
        sample["gpt_solutions"] = solution
        enriched.append(sample)

        # Periodic saving
        if (i + 1) % batch_save_every == 0:
            print(f"\nProcessed {i+1} samples, saving intermediate file...")
            with open(output_path, "w", encoding="utf-8") as out_f:
                json.dump(enriched, out_f, ensure_ascii=False, indent=2)
            print(f"Intermediate file saved to {output_path}")

        # Rate limiting
        time.sleep(0.1)

    # Final save
    print(f"\nSaving final results...")
    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(enriched, out_f, ensure_ascii=False, indent=2)
    print(f"Final output saved to {output_path}")
    print(f"Processed {len(enriched)} samples")


def main():
    parser = argparse.ArgumentParser(
        description="Generate step-by-step solutions using GPT-4o-mini for coding datasets"
    )
    parser.add_argument(
        "--dataset",
        choices=["app", "augmented", "both"],
        default="both",
        help="Which dataset to process: 'app', 'augmented', or 'both'"
    )
    parser.add_argument(
        "--app_path",
        default="benchmarks/APP_code_datasets.json",
        help="Path to APP_code_datasets.json"
    )
    parser.add_argument(
        "--augmented_path",
        default="benchmarks/augmented_coding_datasets.json",
        help="Path to augmented_coding_datasets.json"
    )
    parser.add_argument(
        "--augmented_output",
        default="benchmarks/augmented_coding_datasets_solutions.json",
        help="Output path for augmented dataset with solutions"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use"
    )
    parser.add_argument(
        "--batch_save",
        type=int,
        default=100,
        help="Save progress every N items"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip samples that already have solutions (APP dataset only)"
    )

    args = parser.parse_args()

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    # Process datasets based on choice
    if args.dataset in ["app", "both"]:
        print("\n" + "="*60)
        print("Processing APP_code_datasets.json")
        print("="*60)
        process_app_dataset(
            json_path=args.app_path,
            model=args.model,
            batch_save_every=args.batch_save,
            skip_existing=args.skip_existing
        )

    if args.dataset in ["augmented", "both"]:
        print("\n" + "="*60)
        print("Processing augmented_coding_datasets.json")
        print("="*60)
        process_augmented_dataset(
            json_path=args.augmented_path,
            output_path=args.augmented_output,
            model=args.model,
            batch_save_every=args.batch_save
        )

    print("\n" + "="*60)
    print("All processing complete!")
    print("="*60)


if __name__ == "__main__":
    main()
