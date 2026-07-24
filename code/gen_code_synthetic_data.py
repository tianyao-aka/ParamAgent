import json
import time
import openai
import argparse
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

# System prompt
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
    
"""

def enrich_samples(
    samples: List[Dict],
    model: str = "gpt-4o-mini",
    batch_save_every: int = 500
):
    
    """
    For each sample, call the LLM to generate potential pitfalls and flawed implementations,
    then add them under "pitfalls" and "flawed_impl" keys.
    """
    enriched = []
    
    for i, sample in enumerate(tqdm(samples)):
        sample = parse_sample_string(sample)
        func_sign = sample["func_sign"].strip()
        docstring = sample["docstring"].strip().replace("\n", " ")

        # Build signature + docstring block
        signature_block = (
            f"{func_sign}\n"
            f"    \"\"\"{docstring}\"\"\"\n"
        )

        user_prompt = (
            f"{few_shot}\n\n"
            f"{signature_block}"
        )
        
        # Call the LLM
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            max_tokens=812,
        )
        
        text = response.choices[0].message.content
        sample["raw_texts"] = text
        # print (user_prompt)
        # print (text)
        
        # Split into pitfalls and flawed implementations
        if "<Flawed Implementations>" in text:
            pits, flawed = text.split("<Flawed Implementations>", 1)
            pits = pits.replace("<Pitfalls>", "").strip()
            flawed = flawed.strip()
        else:
            pits = ""
            flawed = text.strip()
        
        # Save into sample
        sample["pitfalls"] = pits
        sample["flawed_impl"] = flawed
        
        enriched.append(sample)
        
        # Periodic saving
        if (i + 1) % batch_save_every == 0:
            print(f"Processed {i+1} samples, saving intermediate file...")
            yield enriched, True  # True signals to save intermediate
    
        # simple rate-limiting
        time.sleep(0.1)
    yield enriched, False  # Final output, False means no intermediate afterwards

def main(json_dir: str, save_path: str):
    # Load input JSON
    with open(json_dir, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Process and save
    last_batch = None
    for enriched, is_intermediate in enrich_samples(data):
        last_batch = enriched
        if is_intermediate:
            with open(save_path, "w", encoding="utf-8") as out_f:
                json.dump(enriched, out_f, ensure_ascii=False, indent=2)
            print(f"Intermediate file saved to {save_path}")

    # final save
    with open(save_path, "w", encoding="utf-8") as out_f:
        json.dump(last_batch, out_f, ensure_ascii=False, indent=2)
    print(f"Final output saved to {save_path}")
    
    print(f"All done, final output saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrich APP samples with pitfalls and flawed implementations")
    parser.add_argument("--json_dir", help="Path to input JSON file")
    parser.add_argument("--save_path", help="Path to output JSON file")
    args = parser.parse_args()
    
    main(args.json_dir, args.save_path)

