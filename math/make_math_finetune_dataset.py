import random
import json
import argparse
from collections import defaultdict
from openai import OpenAI
from datasets import load_dataset
from tenacity import (
    retry,
    stop_after_attempt,  # type: ignore
    wait_random_exponential,  # type: ignore
)
from tqdm import tqdm

# The OpenAI client reads its credentials from the environment.
client = OpenAI()

@retry(wait=wait_random_exponential(min=10, max=120), stop=stop_after_attempt(6))
def process_dataset(dataset: list[dict], save_path: str) -> None:
    """
    Sample up to 1500 examples per subject, query GPT-4o-mini for pitfalls,
    and save the augmented dataset to a JSON file—checkpointing every 500 samples.
    """
    # 1. Group samples by subject
    by_subject = defaultdict(list)
    for sample in dataset:
        subj = sample.get('subject')
        if subj:
            by_subject[subj].append(sample)

    # 2. Sample up to 1500 per subject
    processed = []
    for subj, samples in by_subject.items():
        chosen = random.sample(samples, 1500) if len(samples) > 1500 else samples
        processed.extend(chosen)

    # 3. Enrich each sample with GPT-generated pitfalls, checkpointing every 500
    
    enriched = []
    for idx, sample in enumerate(
        tqdm(processed, desc="Enriching samples")
    ):
        problem = sample.get('problem', '').strip()
        if not problem:
            sample['pitfalls'] = []
        else:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": f"Here is one example: {EXAMPLE}\nProblem:\n{problem}"}
                ],
                max_tokens=900,
                temperature=0.1,
            )
            sample['pitfalls'] = response.choices[0].message.content

        enriched.append(sample)

        # every 500 samples, write a checkpoint
        if idx % 200 == 0:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(enriched, f, ensure_ascii=False, indent=2)

    # 4. Final save of all enriched samples
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)
        print ('writing samples')
    print(f"Done! Saved total {len(enriched)} samples to {save_path}")


system_msg = (
    "Given a single math word problem, list up to 6 potential common reasoning mistakes for this problem. You will be given a example."
    "Return your answer as a Markdown-formatted numbered list, without any additional commentary."
)

EXAMPLE = """
Question: Circle $O$ is located on the coordinate plane with center at $(2,3)$.  One endpoint of a diameter is at $(-1,-1)$.  What are the coordinates of the other endpoint of this diameter?  Express your answer as an ordered pair.

<Pitfalls>

### Potential Mistakes

1. **Confusing the center with an endpoint**
   - Mistake: Assuming that the center of the circle is one of the endpoints of the diameter.
   - Consequence: Might incorrectly calculate the other endpoint by treating the given point as the center.

2. **Incorrect use of midpoint formula**
   - Mistake: Forgetting that the center is the midpoint of the diameter, or misapplying the formula.
   - Example: Using subtraction or division incorrectly, such as `(x + x_2)/2 = center_x` → solving incorrectly for `x_2`.

3. **Using the wrong coordinates for the midpoint**
   - Mistake: Plugging the endpoint coordinates in place of the center coordinates or vice versa.
   - Consequence: Leads to solving for the wrong unknowns.

4. **Arithmetic errors**
   - Mistake: Small math mistakes like sign errors when solving equations, such as `2 = (-1 + x)/2` → solving to get `x = 3` instead of `x = 5`.

5. **Switching x and y**
   - Mistake: Solving for `y` using the x-coordinates, or vice versa.
   - Example: Mixing the formulas for x and y midpoints when solving.

6. **Incorrect interpretation of the diameter**
   - Mistake: Thinking the diameter extends in the same direction from the center as the given point.
   - Consequence: This might result in incorrectly doubling the vector or reflecting in the wrong direction.

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a JSON dataset by sampling, adding GPT-generated pitfalls, and saving the result."
    )
    parser.add_argument("--save_path", type=str, help="Path where the output JSON will be saved.")
    args = parser.parse_args()

    
    dataset = load_dataset("nlile/hendrycks-MATH-benchmark", cache_dir='benchmarks/', trust_remote_code=True)
    dataset = dataset['train']
    # Process and save
    process_dataset(dataset, args.save_path)

