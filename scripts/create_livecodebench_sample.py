#!/usr/bin/env python3
"""
Create a stratified sample from LiveCodeBench test dataset.
Randomly selects 50 samples from each difficulty level (easy, medium, hard).
"""

import json
import random
from pathlib import Path
from collections import defaultdict

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Configuration
SAMPLES_PER_DIFFICULTY = 50
INPUT_FILE = "livecodebench/test.jsonl"
OUTPUT_FILE = "livecodebench/test_small.jsonl"


def load_jsonl(file_path):
    """Load JSONL file into a list of dictionaries."""
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def save_jsonl(samples, file_path):
    """Save list of dictionaries to JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def main():
    # Load the full dataset
    print(f"Loading dataset from {INPUT_FILE}...")
    all_samples = load_jsonl(INPUT_FILE)
    print(f"Loaded {len(all_samples)} total samples")

    # Group samples by difficulty
    samples_by_difficulty = defaultdict(list)
    for sample in all_samples:
        difficulty = sample.get('difficulty')
        if difficulty:
            samples_by_difficulty[difficulty].append(sample)

    # Print statistics
    print("\nDataset statistics:")
    for difficulty in ['easy', 'medium', 'hard']:
        count = len(samples_by_difficulty[difficulty])
        print(f"  {difficulty.capitalize()}: {count} samples")

    # Sample from each difficulty
    selected_samples = []
    print(f"\nSelecting {SAMPLES_PER_DIFFICULTY} samples from each difficulty...")

    for difficulty in ['easy', 'medium', 'hard']:
        available = samples_by_difficulty[difficulty]
        if len(available) < SAMPLES_PER_DIFFICULTY:
            print(f"Warning: Only {len(available)} {difficulty} samples available, using all of them")
            selected = available
        else:
            selected = random.sample(available, SAMPLES_PER_DIFFICULTY)

        selected_samples.extend(selected)
        print(f"  Selected {len(selected)} {difficulty} samples")

    # Save the sampled dataset
    print(f"\nSaving {len(selected_samples)} samples to {OUTPUT_FILE}...")
    save_jsonl(selected_samples, OUTPUT_FILE)

    # Verify and print summary
    print("\nDone! Summary:")
    print(f"  Total samples saved: {len(selected_samples)}")
    print(f"  Output file: {OUTPUT_FILE}")
    print(f"  Random seed: {RANDOM_SEED}")


if __name__ == "__main__":
    main()
