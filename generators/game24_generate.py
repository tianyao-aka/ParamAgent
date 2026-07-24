# game24_generator.py
from generators.model import ModelBase
from .generator_types import Generator
from .parse import add_code_block
from openai import OpenAI
from typing import Optional, List, Dict,Union
from .generator_utils import game24_generate_self_reflection, game24_generate_self_reflection_diverse,generic_generate_game24_impl
from tenacity import (
    retry,
    stop_after_attempt,  # type: ignore
    wait_random_exponential,  # type: ignore
)
    
# ──────────────────────────────  PROMPTS  ──────────────────────────────
G24_SIMPLE_CHAT_INSTRUCTION = (
    "You are an AI that plays the 24-Game. "
    "The user will give you exactly four numbers (e.g. 1 1 11 11). "
    "Respond with one and only one arithmetic expression that\n"
    "  • uses each number **exactly once**\n"
    "  • may use +  −  *  /  and parentheses\n"
    "  • evaluates to **24**.\n"
    "Return the expression as plain text, NO commentary, NO code-block fences."
)


G24_REFLEXION_CHAT_INSTRUCTION = (
    "You are fixing your previous 24-Game expression.  "
    "You will be given:\n"
    "  • your last expression,\n"
    "  • evaluation result (not 24), and\n"
    "  • a brief reflection on the error.\n"
    "Write a new expression that obeys the rules and evaluates to 24. "
    "Return ONLY the expression."
)


# PRE_INSIGHT_SYS_NO_MODEL_INSIGHTS = (
#     "You are an AI assistant for the 24-Game.  "
#     "Given exactly four integers, do the following:\n"
#     "1.  Use your own knowledge to propose a list of potential pitfalls that "
#     "could keep an arithmetic expression (using + − * / and parentheses, each "
#     "number exactly once) from evaluating to 24.  List the pitfalls.\n"
#     "2.  Create **up to six** flawed arithmetic expressions that each use the "
#     "four numbers exactly once and demonstrate one (or more) of those pitfalls "
#     "(e.g. precedence mistake, duplicate usage, division-by-zero, off-by-one, "
#     "incorrect operator, bad grouping, etc.).  After each expression, add a "
#     "brief comment indicating which pitfall it illustrates.\n\n"
#     "Output format:\n"
#     "Refined Pitfalls:\n"
#     "1. …\n"
#     "2. …\n"
#     "Flawed Expressions:\n"
#     "```text\n"
#     "expr1   # pitfall description\n"
#     "expr2   # …\n"
#     "```"
# )

# 24-Game version (used when NO seed pitfalls are provided)
# PRE_INSIGHT_SYS_NO_MODEL_INSIGHTS = (
#     "You are an AI assistant for the 24-Game.  "
#     "Given exactly four integers, do the following:\n"
#     "1.  Use your knowledge to propose a list of most common pitfalls for the given problem that "
#     "could keep an arithmetic expression (using + − * / and parentheses, each "
#     "number exactly once) from evaluating to 24.  List the pitfalls.\n"
#     "2.  Create **no more than 10** flawed arithmetic expressions that each use the "
#     "four numbers exactly once and demonstrate one (or more) of those pitfalls "
#     "(e.g. precedence mistake, duplicate usage, division-by-zero, off-by-one, "
#     "incorrect operator, bad grouping, etc.).  After each expression, add a "
#     "brief comment indicating which pitfall it illustrates.\n\n"
#     "Output format:\n"
#     "Refined Pitfalls:\n"
#     "1. …\n"
#     "2. …\n"
#     "Flawed Expressions:\n"
#     "```text\n"
#     "expr1   # pitfall description\n"
#     "expr2   # …\n"
#     "```"
# )


# G24_PRE_INSIGHTS_FEWSHOT = """Example:
# [numbers]:
# 1,1,11,11

# [pre-collected pitfalls]:
# - Using one of the numbers twice
# - Forgetting operator precedence
# - Division by zero
# - Subtracting instead of adding

# [assistant output]:

# Refined Pitfalls:
# 1. **Operator precedence** – e.g. writing 1+1*11+11 without parentheses (gives 23).
# 2. **Redundant number usage** – accidentally using a number twice or omitting one.
# 3. **Suboptimal grouping** – parentheses in the wrong place give 25 or 22, not 24.

# Flawed Expressions:
# ```text
# 1+1*11+11         # precedence error (23)
# (11-1)*(11-1)     # grouping: uses each once but evaluates to 100
# 11+11+1-1         # adds/subtracts → 22
# (11/1)+11/(1+1)   # redundant use of '1' (uses five numbers)
# 11/(1-1)+11*1     # division by zero
# END OF EXAMPLE
# """


# 24-Game prompt (when no seed pitfalls are supplied)
PRE_INSIGHT_SYS_NO_MODEL_INSIGHTS = (
    "You are an AI assistant for the 24-Game.\n"
    "Given a set of four integers ⟨a,b,c,d⟩, do the following:\n"
    "1▸  List the most common pitfalls that can keep an expression, formed with "
    "those four numbers (each used exactly once with + − * / and parentheses), "
    "from evaluating to 24.  Put these under “Refined Pitfalls:”.\n"
    "2▸  Produce **≤ 10** flawed expressions that each illustrate one or more of "
    "those pitfalls, followed by a short “# pitfall …” comment.\n"
    "3▸  Generate one **neighbour set** by perturbing exactly **one** of the four "
    "numbers by ±1 or ±2 (your choice).  For this neighbour set, output the "
    "best expression you can find—preferably one that equals 24; if that’s "
    "impossible, return the expression whose value is numerically closest to 24.\n\n"
    "Use the format:\n"
    "Numbers: a,b,c,d\n"
    "Refined Pitfalls:\n"
    "1. …\n"
    "2. …\n"
    "Flawed Expressions:\n"
    "```text\n"
    "expr1   # pitfall description\n"
    "expr2   # …\n"
    "```\n"
    "Neighbour & Solution:\n"
    "neighbour_numbers → expr_best\n"
)

# ───────────────────────── 2 few-shot examples ─────────────────────────
G24_PRE_INSIGHTS_FEWSHOT = """Example 1
Numbers: 1,1,11,11

Refined Pitfalls:
1. Operator precedence – e.g. 1+1*11+11 gives 23.
2. Duplicate / missing number – using a number twice or skipping one.
3. Mis-grouping – wrong parentheses lead to 22 or 100.
4. Division by zero – denominator becomes 0.

Flawed Expressions:
```text
1+1*11+11         # precedence (23)
11+11+1-1         # adds/subtracts → 22
(11-1)*(11-1)     # grouping → 100
11/(1-1)+11*1     # division by zero
(11/1)+(11/(1+1)) # duplicate 1 (uses five numbers)
Neighbour & Solution:
1,3,11,11 → (11+1)/(3-1)*11 # equals 24

Example 2
Numbers: 2,5,7,9

Refined Pitfalls:

1 Forgetting order of operations – 2+5*7-9 = 28.

2 Off-by-one subtraction – replacing + with − yields 10 instead of 24.

3 Fractional noise – 9/5≈1.8 then scaling yields 23.4.

4 Reusing a number – 7 appears twice, 2 unused.

Flawed Expressions:
2+5*7-9           # precedence (28)
(9-2)*5-7         # off-by-one (26)
9/5+7*2           # fractions (≈23.4)
(7+7+5)+2         # duplicate 7
9/(2-2)+7*5       # division by zero
Neighbour & Solution:
2,6,7,9 → (9-7)(6+2) # (2)(8) = 16 (best found, 8 away from 24)
END OF EXAMPLES
"""

# ──────────────────────────  Game-24  Generator  ────────────────────────
class Game24Generator(Generator):
    """
    A stripped-down generator specialised for 24-Game puzzles.
    """

    # ------------------------------------------------------------------ #
    #  PRE-INSIGHTS  (optional – here we just echo the pitfalls)          #
    # ------------------------------------------------------------------ #
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def generate_pre_insights(
        self,
        numbers: str,          # e.g. "1 1 11 11"
    ) -> str:
        """
        Ask GPT-4o-mini for refined pitfalls and flawed expressions
        for these four numbers.
        """
        client = OpenAI()
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": PRE_INSIGHT_SYS_NO_MODEL_INSIGHTS},
            {
                "role": "user",
                "content": (
                    f"Here is one example:\n{G24_PRE_INSIGHTS_FEWSHOT}\n\n"
                    "Now do it for my puzzle.\n"
                    f"[numbers]:\n{numbers}\n\n"
                ),
            },
        ]

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=1.0,
            max_tokens=600,
        )

        # print (resp.choices[0].message.content.strip())
        return resp.choices[0].message.content.strip()


    # ------------------------------------------------------------------ #
    #  SELF-REFLECTION                                                   #
    # ------------------------------------------------------------------ #
    def self_reflection(self, expr, feedback, model):
        return game24_generate_self_reflection(expr, feedback, model)

    def self_reflection_diverse(self, expr, feedback, model, diverse_reflections):
        return game24_generate_self_reflection_diverse(
            expr, feedback, model, diverse_reflections
        )

    def func_impl(
        self,
        numbers: str,
        model: ModelBase,
        strategy: str,
        prev_func_impl: Optional[str] = None,
        feedback: Optional[str] = None,
        self_reflection: Optional[str] = None,
        num_comps: int = 1,
        temperature: float = 0.5,
        mistake_insights = None
    ) -> Union[str, List[str]]:
        return generic_generate_game24_impl(
            numbers=numbers,
            model=model,
            strategy=strategy,
            prev_expr=prev_func_impl,
            feedback=feedback,
            self_reflection=self_reflection,
            num_comps=num_comps,
            temperature=temperature,
            simple_chat_instruction=G24_SIMPLE_CHAT_INSTRUCTION,
            reflexion_chat_instruction=G24_REFLEXION_CHAT_INSTRUCTION,
            simple_completion_instruction=G24_SIMPLE_CHAT_INSTRUCTION,
            reflexion_completion_instruction=G24_REFLEXION_CHAT_INSTRUCTION,
            mistake_insights= mistake_insights
        )
        
        
        