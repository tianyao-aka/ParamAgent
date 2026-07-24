from generators.model import ModelBase
from .generator_types import Generator
from .parse import add_code_block  # noqa: F401 (kept for parity with template)
from openai import OpenAI
from typing import Optional, List, Dict, Union
from .generator_utils import (
    mathqa_generate_self_reflection,
    mathqa_generate_self_reflection_diverse,
    generic_generate_mathqa_impl,  # We reuse the generic helper
    mathqa_generate_self_reflection_diverse_parametric,
)
from tenacity import (
    retry,
    stop_after_attempt,  # type: ignore
    wait_random_exponential,  # type: ignore
)

# ──────────────────────────────  PROMPTS  ──────────────────────────────
# Task‑specific chat instructions adapted for mathematics datasets

MATH_SIMPLE_CHAT_INSTRUCTION = (
    "You are an AI agent specialised in solving mathematics problems. "
    "A user will provide a single question.\n\n"
    "❖ Think step‑by‑step, "
    "then, on a new line, prefix the final result with **`Answer:`**.\n"
    "❖ The final answer should be simplified to its simplest form,e.g., 25, 2516_8, \frac{1}{36}, etc."
    "❖ Do **not** include citations."
)


MATH_REFLEXION_CHAT_INSTRUCTION = (
    "You are revising your previous answer to a mathematics problem. You will "
    "receive: (1) the original *question*, (2) your *last answer*, (3) *feedback* "
    "(Right or Wrong) explaining why that answer was unsatisfactory, and (4) your brief "
    "*self‑reflection* on the mistake.\n\n"
    "Produce a revised response with the following format:\n"
    "1. **Reasoning**: step‑by‑step thoughts correcting the error.\n"
    "2. **Answer:** the final result on its own line.\n"
    "❖ The final answer should be simplified to its simplest form,e.g., 25, 2516_8, \frac{1}{36}, etc."
    "Do not add citations."
)

MATH_REFLEXION_CHAT_INSTRUCTION_PARAMETRIC = (
    "You are revising your previous answer to a mathematics problem. You will "
    "receive: (1) the original *question*, (2) potential mistakes and pitfalls, "
    "(3) your *last answer*, (4) *feedback* (Right or Wrong) explaining why that answer was unsatisfactory, and (5) your brief "
    "*self‑reflection* on the mistake.\n\n"
    "Respond with:\n"
    "1. **Reasoning**: updated step‑by‑step thoughts.\n"
    "2. **Answer:** the corrected final result."
    "❖ The final answer should be simplified to its simplest form,e.g., 25, 2516_8, \frac{1}{36}, etc."
)

# ───── Intent extraction prompt (optional) ─────

PRE_INSIGHT_SYS_INTENT = (
    "You will be given a mathematics problem. "
    "Extract the potential pitfalls and mistakes for this problem. Following the output format of the given example. "
)

PRE_INSIGHT_FEWSHOT = """
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
# ────────────────────────  Math‑QA  Generator  ────────────────────────


class MathQAGenerator(Generator):
    """Generator class for single‑question mathematics datasets."""

    # ------------------------------------------------------------------ #
    #  (Optional) PRE‑INSIGHTS                                           #
    # ------------------------------------------------------------------ #
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
    def generate_pre_insights(self, question: str) -> str:
        """Return a brief structured intent description for the math question."""
        client = OpenAI()
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": PRE_INSIGHT_SYS_INTENT},
            {
                "role": "user",
                "content": (
                    f"{PRE_INSIGHT_FEWSHOT}\n"
                    "Given the above example, now do it for my question:\n"
                    f"[question]: {question}"
                ),
            },
        ]
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1,
            max_tokens=500,
        )
        return resp.choices[0].message.content.strip()

    # ------------------------------------------------------------------ #
    #  SELF‑REFLECTION                                                   #
    # ------------------------------------------------------------------ #
    def self_reflection(
        self,
        question: str,
        answer: str,
        feedback: str,
        model: ModelBase,
    ) -> str:  # noqa: D401
        """Generate a single self‑reflection string for the wrong *answer*."""
        return mathqa_generate_self_reflection(question, answer, feedback, model)


    def self_reflection_diverse(
        self,
        question: str,
        answer: str,
        feedback: str,
        model: ModelBase,
        diverse_reflections: int,
    ) -> List[str]:
        """Generate *diverse_reflections* alternative self‑reflections."""
        return mathqa_generate_self_reflection_diverse(
            question, answer, feedback, model, diverse_reflections
        )

    def self_reflection_diverse_parametric(
        self,
        question: str,
        answer: str,
        feedback: str,
        model: ModelBase,
        diverse_reflections: int,
        insights: str,
    ) -> List[str]:
        """Generate *diverse_reflections* alternative self‑reflections with insights."""
        return mathqa_generate_self_reflection_diverse_parametric(
            question, answer, feedback, model, diverse_reflections, insights
        )

    # ------------------------------------------------------------------ #
    #  MAIN IMPLEMENTATION (func_impl)                                   #
    # ------------------------------------------------------------------ #
    def func_impl(
        self,
        question: str,
        model: ModelBase,
        strategy: str,
        prev_answers: Optional[str] = None,
        feedback: Optional[str] = None,
        self_reflection: Optional[str] = None,
        num_comps: int = 1,
        temperature: float = 0.1,
        mistake_insights: Optional[str] = None,
        fewshot_example: Optional[str] = None,
    ) -> Union[str, List[str]]:
        """Generate **answer(s)** for a given *question*."""

        return generic_generate_mathqa_impl(
            question=question,
            model=model,
            strategy=strategy,
            prev_answers=prev_answers,
            feedback=feedback,
            self_reflection=self_reflection,
            num_comps=num_comps,
            temperature=temperature,
            simple_chat_instruction=MATH_SIMPLE_CHAT_INSTRUCTION,
            reflexion_chat_instruction=(
                MATH_REFLEXION_CHAT_INSTRUCTION
                if mistake_insights is None
                else MATH_REFLEXION_CHAT_INSTRUCTION_PARAMETRIC
            ),
            simple_completion_instruction=MATH_SIMPLE_CHAT_INSTRUCTION,
            reflexion_completion_instruction=(
                MATH_REFLEXION_CHAT_INSTRUCTION
                if mistake_insights is None
                else MATH_REFLEXION_CHAT_INSTRUCTION_PARAMETRIC
            ),
            mistake_insights=mistake_insights,
            fewshot_example=fewshot_example,
        )
