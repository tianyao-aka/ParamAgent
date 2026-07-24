from typing import Any
from .executor_types import ExecuteResult, Executor
from openai import OpenAI  # if not already imported

class MathExecutor(Executor):
    def execute(self, func: str, tests: list, timeout: int = 5) -> ExecuteResult:
        # MultihopQA does not use execute in this implementation
        raise NotImplementedError("MultiHopQAExecutor.execute is not used for evaluation.")

    def evaluate(self, answer: str, golden_truth: str, timeout: int = 5) -> bool:
        """
        Evaluate whether the predicted answer matches the gold answer.

        Args:
            answer (str): The predicted answer string.
            golden_truth (str): The ground-truth answer string.
            timeout (int): Timeout (unused).

        Returns:
            bool: True if answers are considered equivalent (exact match or semantically).
        """
        
        # First check exact match
        if answer.strip().lower() == golden_truth.strip().lower():
            return True

        prompt = f"""
        Determine if the following two answers are equivalent in the context of a mathematical problem.

        Answer 1: {answer}
        Answer 2: {golden_truth}

        Respond only with "Yes" or "No".
        """

        try:
            client = OpenAI(timeout=60.0)  # 60 second timeout to prevent hanging
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt.strip()}],
                temperature=0.00001,
                max_tokens=8,
            )
            result = response.choices[0].message.content.strip().lower()
            print("executor eval:", result)
            return result.startswith("yes")
        except Exception:
            print("executor eval: exception")
            return False
