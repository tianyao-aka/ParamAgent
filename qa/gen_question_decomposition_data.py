import json
import random
import time
import argparse
import os

from datasets import load_dataset
import openai
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

openai.api_key = os.getenv("OPENAI_API_KEY")

# ──────────────────────────── Prompts ────────────────────────────

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
1. Retrieve factual data about Maltsev and Turchin’s academic domains.  
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
   - Identify Rihanna’s **fourth studio album** released on that date and promoted by the tour  

4. **Answer Type Expected**:  
   - Album title (e.g., "Rated R")  

5. **Reasoning Type**:  
   - Factual retrieval from discography and tour association  

6. **Required Background**:  
   - Rihanna’s discography and tour-album mapping  
--------------------------------------------------------------------------------

🧠 **Inference Trace**  
1. Locate Rihanna’s albums around 2009.  
2. Find the one released November 20, 2009.  
3. Confirm it was promoted by “The Last Girl on Earth” tour.  
4. Conclude the album is “Rated R”.  
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

# “Transient” errors we want to retry on
TRANSIENT_ERRORS = (
    openai.RateLimitError,      # 429s
    openai.APIStatusError,      # 5xx responses
    openai.APIConnectionError,  # network blips
    openai.Timeout,             # request time-outs
)

# use tenacy

@retry(
    retry=retry_if_exception_type(TRANSIENT_ERRORS),
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
)
def decompose_question(question: str, model: str = "gpt-4o-mini") -> str:
    """
    Call the LLM to decompose `question` into Intent, Key Components,
    Inference Trace, etc., using the few-shot examples.
    """
    user_prompt = (
        f"{PRE_INSIGHT_FEWSHOT}\n"
        f"q: {question}\n\n"
        "### Question Parsing and Intent Extraction"
    )
    
    resp = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": user_prompt}
        ],
        temperature=0.1,
        max_tokens=600
    )
    return resp.choices[0].message.content.strip()

# ────────────────────────────── Main ──────────────────────────────

def main(output_path: str, sample_size_per_level: int = 10000):
    # 1. Load dataset
    ds = load_dataset("hotpot_qa", "distractor", trust_remote_code=True)
    train = ds["train"]

    # 2. Gather indices by level
    levels = {"easy": [], "medium": [], "hard": []}
    for idx, ex in enumerate(train):
        lvl = ex["level"].lower()
        if lvl in levels:
            levels[lvl].append(idx)

    # 3. Sample N indices per level
    sampled_indices = []
    for lvl in ("easy", "medium", "hard"):
        idxs = levels[lvl]
        random.shuffle(idxs)
        sampled_indices.extend(idxs[:sample_size_per_level])

    random.shuffle(sampled_indices)

    # 4. Decompose and save
    results = []
    for i, idx in enumerate(tqdm(sampled_indices, desc="Decomposing")):
        question = train[idx]["question"]
        try:
            decomposition = decompose_question(question)
        except Exception as e:
            print(f"[Error] idx={idx}: {e}")
            time.sleep(1)
            continue

        results.append({
            "question": question,
            "decomposition": decomposition,
            "level": train[idx]["level"]
        })
        # save intermediate every 500
        if (i + 1) % 1000 == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Saved intermediate {i+1} items to {output_path}")
        time.sleep(0.01)

    # final save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Done: saved {len(results)} decompositions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample and decompose HotpotQA questions"
    )
    parser.add_argument("--output_path", help="Where to write the JSON output")
    parser.add_argument(
        "--per_level", type=int, default=10000,
        help="Number of samples per difficulty level"
    )
    
    args = parser.parse_args()
    main(args.output_path, args.per_level)
