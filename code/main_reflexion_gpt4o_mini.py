import os
import argparse
import sys
from reflexion_gpt4o_mini import run_reflexion_gpt4o_mini
from utils import read_jsonl, read_jsonl_gz
import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, help="The name of the run", default="reflexion_gpt4o_mini")
    parser.add_argument("--root_dir", type=str,
                        help="The root logging directory", default="root")
    parser.add_argument("--dataset_path", type=str,
                        help="The path to the HumanEval benchmark dataset",
                        default="benchmarks/humaneval_full.jsonl")
    parser.add_argument("--language", type=str, help="Language: `py`", default="py")
    parser.add_argument("--model", type=str,
                        help="Model name (default: llama3_1_8b)",
                        default="llama3_1_8b")
    parser.add_argument("--num_samples", type=int,
                        help="Number of samples to test (default: all)", default=None)
    parser.add_argument("--verbose", action='store_true',
                        help="To print live logs")
    parser.add_argument("--gpt_solutions_path", type=str,
                        help="Path to GPT solutions JSONL file",
                        default="benchmarks/code_solutions/humaneval_full_solutions.jsonl")
    args = parser.parse_args()
    return args


def main(args):
    # check if the root dir exists and create it if not
    if not os.path.exists(args.root_dir):
        os.makedirs(args.root_dir)

    # get the dataset name
    dataset_name = os.path.basename(args.dataset_path).replace(".jsonl", "").replace(".json", "")

    # check if log path already exists
    log_dir = os.path.join(args.root_dir, args.run_name)
    log_path = os.path.join(
        log_dir, f"{dataset_name}_reflexion_gpt4o_mini_{args.model}_{args.language}.jsonl")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # check if the dataset path exists and load the dataset
    if not os.path.exists(args.dataset_path):
        raise ValueError(f"Dataset path `{args.dataset_path}` does not exist")

    # Load dataset based on file extension
    if args.dataset_path.endswith(".jsonl.gz"):
        dataset = read_jsonl_gz(args.dataset_path)
    elif args.dataset_path.endswith(".jsonl"):
        dataset = read_jsonl(args.dataset_path)
    elif args.dataset_path.endswith(".json"):
        with open(args.dataset_path, 'r') as f:
            dataset = json.load(f)
    else:
        raise ValueError(f"Dataset path `{args.dataset_path}` has unsupported file extension")

    # Limit dataset if num_samples is specified
    if args.num_samples is not None:
        dataset = dataset[:args.num_samples]
        print(f"Limited to {len(dataset)} samples for testing")

    print(f"Loaded {len(dataset)} examples from {args.dataset_path}")
    print(f"Results will be saved to: {log_path}")
    print(f"Using GPT solutions from: {args.gpt_solutions_path}")
    print(f"Model: {args.model}")
    print(f"\nAlgorithm: Phase 1 (4 GPT-guided attempts) + Phase 2 (3 reflexion attempts)")

    # Run reflexion with GPT-4o-mini approach
    run_reflexion_gpt4o_mini(
        dataset=dataset,
        model_name=args.model,
        language=args.language,
        log_path=log_path,
        verbose=args.verbose,
        gpt_solutions_path=args.gpt_solutions_path
    )

    print(f"\n✓ Reflexion GPT-4o-mini approach completed!")
    print(f"Results saved to: {log_path}")


if __name__ == "__main__":
    args = get_args()
    main(args)
