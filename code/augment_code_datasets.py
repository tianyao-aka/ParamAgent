import os
import json
import time
from openai import OpenAI, OpenAIError
from typing import List, Dict
import re
import sys
from tqdm import tqdm  # pip install tqdm
import argparse
import random
# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Instantiate new OpenAI client (v1.20+) using env OPENAI_API_KEY
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    raise RuntimeError("Please set the OPENAI_API_KEY environment variable.")

MODEL_NAME  = "gpt-4o-mini"
TEMPERATURE = 1.0
MAX_TOKENS  = 256
RETRY_DELAY = 2.0
DUP_RETRIES  = 5
MEMORY_SIZE  = 150   # look-back window in prompt
CATEGORIES = [
    # ─────────────────── Core Text & Parsing ───────────────────
    "String Manipulation",
    "Regular‐Expression Parsing",
    "Natural-Language Tokenisation",
    "CSV / JSON Parsing",
    "URL / URI Parsing",               # ← new
    "Text Justification / Word-Wrapping",  # ← new

    # ─────────────────── Lists, Arrays, SEQ ────────────────────
    "Array / List Algorithms",
    "Two-Pointer / Sliding-Window",
    "Sorting & Searching",
    "Statistical Summary of Sequences",

    # ─────────────────── Maths & Numbers ───────────────────────
    "Elementary Arithmetic / Algebra",
    "Number Theory & Divisibility",
    "Bitwise Operations",
    "Combinatorics & Counting",
    "Probability / Statistics",

    # ─────────────────── Data-Structures ───────────────────────
    "Hash / Set / Dict Operations",
    "Stack / Queue Simulation",
    "Linked-List Manipulation",
    "Matrix Operations",
    "Heap / Priority Queue Operations",    # ← new
    "Trie / Prefix-Tree",                  # ← new

    # ─────────────────── Graphs & Trees ────────────────────────
    "Graph / Tree Traversal",
    "Binary Search Trees",
    "Dynamic Programming",
    "Recursion / Backtracking",
    "Union-Find / Disjoint Set",           # ← new

    # ─────────────────── Geometry / Coordinates ───────────────
    "Geometry & Coordinate Computation",

    # ─────────────────── Dates / Times / Calendars ─────────────
    "Date & Time Calculations",

    # ─────────────────── Miscellaneous Practical ──────────────
    "File & Path Utilities",
    "Data-Type Conversion & Formatting",
    "Cipher / Encoding",                   # ← new
    "Simulation / Game Logic",
    "Misc Small-Scale Algorithms"
]




# ------------------------------------------------------------------
# 0. Template example (two-line, no JSON braces)
# ------------------------------------------------------------------
template_example = (
    "func_sign: def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
    "docstring: 'Check if any two numbers in the list are closer than the given threshold.'"
)

# ------------------------------------------------------------------
# 1. System prompt with the example
# ------------------------------------------------------------------


def generate_tasks(k: int) -> List[str]:
    """
    Generate k unique two-line task definitions using GPT.
    Returns a list of raw strings, each:
      func_sign: ...
      docstring: '...'
    """
    tasks = []
    seen_names  = set()    # global set of all function signatures
    seen_hashes = set()    # global set of all docstring hashes

    for i in tqdm(range(k), desc="Generating tasks"):
        # Prepare rolling memory of recent names
        category = random.choice(CATEGORIES)
        recent = list(seen_names)[-MEMORY_SIZE:]
        memory_block = ""
        if recent:
            memory_block = "Do NOT reuse any of these function names:\n" + \
                "\n".join(f"- {n}" for n in recent) + "\n"

        # Build prompts
        cat_list = "\n".join(f"{j+1}. {c}" for j, c in enumerate(CATEGORIES))
        system_content = (
            "You are an expert Python engineer crafting interview problems.\n"
            f"Follow this EXACT format:\n\n{template_example}\n\n"
            "- Randomly pick ONE category from the list below.\n"
            "- Output EXACTLY two lines:\n"
            "    func_sign: <signature with colon>\n"
            "    docstring: '<single-quoted string with \\n escapes>'\n"
            "- Do NOT wrap in JSON or triple quotes.\n"
            "- Avoid any collisions with past tasks.\n\n"
            + memory_block
        )
        system_msg = {"role": "system", "content": system_content}
        user_msg   = {
            "role": "user",
            "content": f"\nGenerate problem in category: {category}\n\n"
        }
        # Attempt until unique or out of retries
        for attempt in range(DUP_RETRIES):
            try:
                resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[system_msg, user_msg],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                raw = resp.choices[0].message.content.strip()
                lines = [l.strip() for l in raw.splitlines() if l.strip()]
                if len(lines) != 2 or not lines[0].startswith("func_sign:"):
                    raise ValueError("Malformed output")

                func_sign = lines[0].split("func_sign:",1)[1].strip()
                docstring = lines[1].split("docstring:",1)[1].strip().strip("'\"")

                # Global deduplication check
                if func_sign in seen_names:
                    raise ValueError("Duplicate func_sign")

                # Accept it
                seen_names.add(func_sign)
                tasks.append(raw)
                break

            except (OpenAIError, ValueError) as e:
                # If last retry, log and move on
                if attempt == DUP_RETRIES - 1:
                    print(f"⚠️   Skipping task #{i+1} after dup/errors: {e}")
                time.sleep(RETRY_DELAY)
        # end retry loop

    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="Generate a set of unique, HumanEval-style Python tasks"
    )
    parser.add_argument(
        "-k", "--num_tasks",
        type=int,
        default=1000,
        help="How many tasks to generate"
    )
    parser.add_argument(
        "-o", "--outfile",
        type=str,
        default="tasks.json",
        help="Path to the JSON file where tasks will be saved"
    )
    args = parser.parse_args()

    # Generate the tasks
    tasks = generate_tasks(args.num_tasks)  # returns List[str] of two-line entries

    # Save to JSON
    with open(args.outfile, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)

    print(f"✔ Generated {len(tasks)} unique tasks → {args.outfile}")

if __name__ == "__main__":
    main()