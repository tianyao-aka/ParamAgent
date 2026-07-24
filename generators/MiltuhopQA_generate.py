from generators.model import ModelBase
from .generator_types import Generator
from .parse import add_code_block  # noqa: F401  (kept for parity with template)
from openai import OpenAI
from typing import Optional, List, Dict, Union
from .generator_utils import (
    multihopqa_generate_self_reflection,
    multihopqa_generate_self_reflection_diverse,
    generic_generate_multihopqa_impl,
    multihopqa_generate_self_reflection_parametric,
    multihopqa_generate_self_reflection_diverse_parametric,
)
from tenacity import (
    retry,
    stop_after_attempt,  # type: ignore
    wait_random_exponential,  # type: ignore
)

# ──────────────────────────────  PROMPTS  ──────────────────────────────
# ───── Task‑specific chat instructions (simple + reflexion) ─────

MQ_SIMPLE_CHAT_INSTRUCTION = (
    "You are an AI agent specialised in answering *multi‑hop* factual questions "
    "(e.g. HotpotQA). A user will provide one question and a collection of sentences. Some of them are relevant to the question, some not. Asnwer the question based on the given sentences.\n\n"
    "❖ Respond with **only one short answer phrase** that "
    "correctly answers the question.\n"
    "❖ Do **not** include other texts such as explanations and citations."
)

MQ_REFLEXION_CHAT_INSTRUCTION = (
    "You are revising your previous answer to a multi‑hop QA question.  You will "
    "receive: (1) the original *question*, (2) your *last answer*, (3) *supporting context*, (4) *feedback* (Right or Wrong) "
    "explaining why that answer was unsatisfactory, and (5) your brief "
    "*self‑reflection* on the mistake.\n\n"
    "Produce a **new single‑phrase answer** that resolves the error and fully "
    "answers the question.  Output **only** the answer—no commentary, no code."
)


MQ_REFLEXION_CHAT_INSTRUCTION_PARAMETRIC = (
    "You are revising your previous answer to a multi‑hop QA question.  You will "
    "receive: (1) the original *question*, (2) some key points, the underlying intent, "
    "and possible inference patterns that facilitates answering this quetion,  (3) your *last answer*, (4) *supporting context*, (5) *feedback* (Right or Wrong) "
    "explaining why that answer was unsatisfactory, and (6) your brief "
    "*self‑reflection* on the mistake.\n\n"
    "Based on the inputs, produce a **new single‑phrase answer** that resolves the error and fully "
    "answers the question.  Output **only** the answer—no commentary, no code."
)


# ───── We keep the following template strings unchanged for now — they will
# be repurposed (or removed) in later iterations.  They are left here to avoid
# breaking import dependencies further down the pipeline. ─────

PRE_INSIGHT_SYS_INTENT = (
    "You will be given a question.  "
    "Use your knowledge to extract the key point, the underlying intent, "
    "and possible inference patterns needed to answer the question."
)

# PRE_INSIGHT_FEWSHOT = """
# <Example 1>
# q: Anatoly Maltsev and Valentin Turchin were both from Russia, which of the two is known for his work as a mathematician?

# Question Parsing and Intent Extraction
# Intent:
# --------------------------------------------------------------------------------
# 🔍 Key Components
# 1. Concept A:
# - **Owner earnings** — a financial metric often associated with a more accurate measure of a company's profitability, typically defined as:
#   - Used in contrast to traditional accounting earnings (e.g., net income or EBITDA)
# 2. Entity B:
# - **Warren Buffett** — investor, chairman and CEO of Berkshire Hathaway, known for value investing and long-term capital allocation
# 3. Implied Relationship:
# - The activity that ties Buffett and this financial metric — likely something Buffett actively practices, promotes, or uses in analysis
# 4. Answer Type Expected:
# - A financial activity or philosophy (e.g., \"value investing\", \"intrinsic valuation\", \"fundamental analysis\")
# --------------------------------------------------------------------------------
# 🧠 Inference Trace
# - **Warren Buffett** coined and popularized the term \"owner earnings\" in his Berkshire Hathaway shareholder letters
# - He uses **owner earnings** to estimate the **intrinsic value** of a business
# - This ties directly into his **value investing** framework — seeking to buy businesses below their intrinsic value based on future cash flows
# """


PRE_INSIGHT_FEWSHOT = """
<Example 1>
q: Anatoly Maltsev and Valentin Turchin were both from Russia, which of the two is known for his work as a mathematician?

Question Parsing and Intent Extraction  
Intent:  
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
The Last Girl on Earth was the third concert tour by Barbadian recording artist Rihanna, the tour visited Europe, Asia, North America and Australia to support her fourth studio album, which 2009 , and fourth studio album by Barbadian singer Rihanna, and released on November 20, 2009 by Def Jam Recordings and SRP Records?

### Question Parsing and Intent Extraction

**Intent:**
--------------------------------------------------------------------------------
🔍 **Key Components**
1. **Entity A**:  
   - *The Last Girl on Earth* — Rihanna's third concert tour, associated with promoting a studio album  

2. **Entity B**:  
   - *Fourth studio album* by Rihanna — referenced multiple times, released in 2009  

3. **Key Relationship / Constraint**:  
   - Identify the name of Rihanna’s **fourth studio album**, which was released on **November 20, 2009**, and **supported by** her third concert tour, *The Last Girl on Earth*  

4. **Answer Type Expected**:  
   - Album title (e.g., *Rated R*)  

5. **Reasoning Type**:  
   - Factual entity retrieval based on event-album association and release date  

6. **Required Background**:  
   - Rihanna’s discography: album release dates and which albums were promoted during which concert tours  

--------------------------------------------------------------------------------

🧠 **Inference Trace**
- Determine the name of Rihanna’s **fourth studio album**  
- Confirm that this album was released on **November 20, 2009**  
- Verify that this album was the basis for the **"The Last Girl on Earth"** tour  
- Conclude that the album is **Rated R**

--------------------------------------------------------------------------------

📝 **Disambiguation Note**
- Although the question includes fragmented/redundant phrasing, the focus is clear: determine the album that matches both the **release date** and **tour association**
"""

# ────────────────────────  Multihop‑QA  Generator  ──────────────────────

class MultiHopQAGenerator(Generator):
    """Generator class for multi‑hop QA tasks (e.g., HotpotQA)."""

    # ------------------------------------------------------------------ #
    #  (Optional) PRE‑INSIGHTS                                           #
    # ------------------------------------------------------------------ #
    # The multihop‑QA setting generally does not require the same kind of
    # “pitfall synthesis” used in the Game‑24 template.  We therefore
    # implement a stub that simply returns an empty string.  This keeps the
    # public interface identical to the superclass and lets downstream
    # training pipelines call *generate_pre_insights* unconditionally.
    #
    # When richer decomposition hints are available (e.g. automated
    # question decomposition or supporting‑fact retrieval), they can be
    # surfaced here in a later revision.
    # ------------------------------------------------------------------ #

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
    def generate_pre_insights(self, question: str) -> str:
        """
        Given a multi‐hop question, extract its key point, intent, and
        inference pattern. Returns the structured intent description.
        """
        client = OpenAI()
        messages: List[Dict[str, str]] = [
            {"role": "system",  "content": PRE_INSIGHT_SYS_INTENT},
            {
                "role": "user",
                "content": (
                    f"Here is one example:\n\n{PRE_INSIGHT_FEWSHOT}\n\n"
                    "Now do it for my question:\n"
                    f"[question]: {question}"
                ),
            },
        ]
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1,
            max_tokens=400,
        )
        return resp.choices[0].message.content.strip()

    # ------------------------------------------------------------------ #
    #  SELF‑REFLECTION                                                   #
    # ------------------------------------------------------------------ #
    def self_reflection(self, question:str,answer:str, context: str, feedback: str, model: ModelBase) -> str:  # noqa: D401
        """Generate a single self‑reflection string for the wrong *answer*."""
        return multihopqa_generate_self_reflection(question, answer, context, feedback, model)


    def self_reflection_parametric(self, question:str, answer:str, context: str, feedback: str, insights: str, model: ModelBase) -> str:  # noqa: D401
        """Generate a single self‑reflection string for the wrong *answer*."""
        return multihopqa_generate_self_reflection_parametric(question, answer, context, feedback, insights, model)

    def self_reflection_diverse(
        self,
        question: str,
        answer: str,
        context: str,
        feedback: str,
        model: ModelBase,
        diverse_reflections: str,
    ) -> List[str]:
        """Generate *diverse_reflections* alternative self‑reflections."""
        return multihopqa_generate_self_reflection_diverse(
            question, answer, context, feedback, model, diverse_reflections
        )

    def self_reflection_diverse_parametric(
        self,
        question: str,
        answer: str,
        context: str,
        feedback: str,
        model: ModelBase,
        diverse_reflections: int,
        insights: str,
    ) -> List[str]:
        """Generate *diverse_reflections* alternative self‑reflections."""
        return multihopqa_generate_self_reflection_diverse_parametric(
            question, answer, context, feedback, model, diverse_reflections,insights
        )

    # ------------------------------------------------------------------ #
    #  MAIN IMPLEMENTATION (func_impl)                                   #
    # ------------------------------------------------------------------ #
    def func_impl(
        self,
        question: str,
        context: str,
        model: ModelBase,
        strategy: str,
        prev_answers: Optional[str] = None,
        feedback: Optional[str] = None,
        self_reflection: Optional[str] = None,
        num_comps: int = 1,
        temperature: float = 0.1,
        question_decomposition: Optional[str] = None,  # replaces mistake_insights
        fewshot_example: Optional[str] = None
    ) -> Union[str, List[str]]:
        """Generate **answer(s)** for a given *question* via the generic helper."""

        return generic_generate_multihopqa_impl(
            question=question,
            context=context,
            model=model,
            strategy=strategy,
            prev_answers=prev_answers,
            feedback=feedback,
            self_reflection=self_reflection,
            num_comps=num_comps,
            temperature=temperature,
            simple_chat_instruction=MQ_SIMPLE_CHAT_INSTRUCTION,
            reflexion_chat_instruction=MQ_REFLEXION_CHAT_INSTRUCTION if question_decomposition is None else MQ_REFLEXION_CHAT_INSTRUCTION_PARAMETRIC,
            simple_completion_instruction=MQ_SIMPLE_CHAT_INSTRUCTION,
            reflexion_completion_instruction=MQ_REFLEXION_CHAT_INSTRUCTION if question_decomposition is None else MQ_REFLEXION_CHAT_INSTRUCTION_PARAMETRIC,
            question_decomposition=question_decomposition,
            fewshot_example=fewshot_example,
        )

