import json
import time
import argparse
from typing import List, Dict
from datasets import load_dataset
from tqdm import tqdm
import openai


system_prompt = (
        "You are an AI assistant for Python coding. Given a function signature and docstring, "
        "use your knowledge to propose potential pitfalls for the implementation, "
        "and list the possible pitfalls, and generate up to 6 flawed implementations "
        "specific to the function signature that cover as many pitfalls as possible. "
        "Use <Pitfalls> and <Flawed Implementations> before pitfalls and implementations."
    )


few_shot = """Example:
[Function Signature]:
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\"Check if any two numbers in the list are closer than the threshold.\"\"\"

<Pitfalls>:
1. **Empty or Single-Element Lists** must return `False`, not `True`.
2. **Duplicate Values** must be compared (difference 0), so never drop duplicates.
3. Always use **absolute difference** (`abs(a - b)`), not raw subtraction.
4. Use the correct **strictness** (`< threshold`, not `<=`).
5. Ensure you don’t **exit too early**—check all distinct pairs.

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

def extract_signature_and_doc(
    question: str,
    solution: str,
    difficulty: str,
    model: str = "gpt-4o-mini"
) -> str:
    system_prompt = (
        "You are an AI assistant that reads a problem description, difficulty, and a sample solution, "
        "and writes a Python function signature with a concise docstring that describes its behavior."
    )
    
    few_shot = """Example:
[Function Signature]:
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\"Check if any two numbers in the list are closer than the threshold.\"\"\"
"""
    user_prompt = (
        f"{few_shot}\n"
        f"Question:\n{question}\n\n"
        f"Difficulty: {difficulty}\n\n"
        f"Sample solution:\n```python\n{solution}\n```\n\n"
        "Now output **only** a new `[Function Signature]:` block in the same format."
    )
    resp = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0.1,
        max_tokens=150
    )
    return resp.choices[0].message.content.strip()

# Second LLM call: generate pitfalls + flawed implementations
def extract_pitfalls_and_flawed(func_block: str, model="gpt-4o-mini") -> (str, str):
    user_prompt = f"{few_shot}\n\n{func_block}"
    resp = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0.5,
        max_tokens=812
    )
    text = resp.choices[0].message.content
    if "<Flawed Implementations>" in text:
        pits, flawed = text.split("<Flawed Implementations>", 1)
        pits = pits.replace("<Pitfalls>:", "<Pitfalls>:").strip()
        flawed = flawed.strip()
    else:
        pits = text
        flawed = ""
    return text, pits, flawed

def main(output_path: str):
    ds = load_dataset(
        "codeparrot/apps",
        "all",             # ← use the 'all' config, not 'train'
        split="train",     # ← now request the 'train' split
        trust_remote_code=True
    )
    out = []
    for sample in tqdm(ds, desc="Sampling APP"):
        diff = sample.get("difficulty", "").lower()
        if diff not in ("interview", "introductory"):
            continue
        
        question = sample["question"].strip()
        sols = sample.get("solutions") or []
        solution = sols

        # 1) get signature + docstring block
        try:
            func_block = extract_signature_and_doc(question, solution, diff)
        except Exception as e:
            print("Signature error:", e)
            time.sleep(1)
            continue
        time.sleep(0.1)

        # 2) get pitfalls + flawed implementations
        try:
            raw_text,pitfalls, flawed_impl = extract_pitfalls_and_flawed(func_block)
        except Exception as e:
            print("Pitfalls error:", e)
            time.sleep(1)
            continue
        time.sleep(0.1)

        out.append({
            "question": question,
            "func_sign": func_block,
            "difficulty": diff,
            "pitfalls": pitfalls,
            "flawed_impl": flawed_impl,
            "raw_text": raw_text
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(out)} samples to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", help="path to output JSON")
    args = parser.parse_args()
    main(args.output_path)
