from generators.model import ModelBase, Message
import random

from typing import Union, List, Optional, Callable
import re

# ───────────────────────────  PROMPTS  ────────────────────────────
G24_REFLECT_SYS = (
    "You are an AI assistant for the 24-Game. "
    "You will be shown an arithmetic expression that was intended to "
    "make 24 from four given numbers, the test feedback, and possibly "
    "earlier reflections.  "
    "Write a short reflection (2-4 sentences) diagnosing the problem "
    "and hinting how to fix it.  Do NOT output any new expression."
)


G24_DIVERSE_REFLECTION_ONESHOT_CHAT_INSTRUCTION = \
"""You are a 24-Game assistant.  
You will be given:

  • the four input numbers,  
  • your previous arithmetic expression (which failed),  
  • the numeric feedback (e.g. “evaluates to 23 which is not 24”), and  
  • any reflections you have already written.  

Write **three new, distinct reflections** to help correct the expression.  
Use the format:

Problem: <concise diagnosis of what went wrong>.  
Fix: <specific hint or change that would make a legal 24-Game expression>.  
"""

def remove_unicode_chars(text: str) -> str:
    return re.sub(r'[^\x00-\x7F]+', '', text)



"""Multihop-QA generator adapted from the Game-24 template (REVISED).

Key changes in this revision:
  • `func_impl` now accepts a mandatory *context* string that contains the
    supporting passages for the question.
  • The method propagates this context to `generic_generate_multihopqa_impl` so
    that both *simple* and *reflexion* strategies can condition on evidence.
  • No other functional changes were introduced.
"""

from generators.model import ModelBase
from .generator_types import Generator
from .parse import add_code_block  # kept for parity with template, may be unused
from openai import OpenAI
from typing import Optional, List, Dict, Union
from tenacity import (
    retry,
    stop_after_attempt,  # type: ignore
    wait_random_exponential,  # type: ignore
)


def generic_generate_multihopqa_impl(
    *,
    question: str,
    context: str,
    model: ModelBase,
    strategy: str,                     # "simple" or "reflexion"
    prev_answers: Optional[str],
    feedback: Optional[str],
    self_reflection: Optional[str],
    num_comps: int,
    temperature: float,
    simple_chat_instruction: str,
    reflexion_chat_instruction: str,
    simple_completion_instruction: str,
    reflexion_completion_instruction: str,
    question_decomposition: Optional[str] = None,
    fewshot_example: Optional[str] = None,
) -> Union[str, List[str]]:
    
    """
    Returns
    -------
    Union[str, List[str]]
        A single answer string (if num_comps==1) or a list of answers.
    """
    
    if strategy not in {"simple", "reflexion"}:
        raise ValueError("strategy must be 'simple' or 'reflexion'")

    if strategy == "reflexion" and (
        prev_answers is None or feedback is None or self_reflection is None
    ):
        raise ValueError(
            "Reflexion strategy requires prev_answers, feedback, and self_reflection"
        )

    question_parsing_instruction = (
        "You will be given a multi-hop question, with some context sentences, some of them are relevant to the question, some are not. "
        "You will be provided with some extracted insights about the queston to facilitate your understanding of the question, and possibly some demonstrations from similar questions. "
        "Using the information provided, write a short answer (a phrase or word) to the question. "
        " Return **only** the answer—no other texts such as explanations "
    )

    # --- Chat models ---
    if model.is_chat:
        if strategy == "simple":
            # If we have decomposition or context, inject them
            if question_decomposition:
                print ('use question intent')
                parts = [f"[question]: {question}"]
                if question_decomposition:
                    parts.append(f"[question intent decomposition]:\n{question_decomposition}")
                parts.append(f"[context]:\n{context}")
                user_block = "\n\n".join(parts)
                messages = [
                    Message(role="system", content=question_parsing_instruction),
                    Message(role="user", content=user_block),
                ]
            else:
                # Vanilla from-scratch prompt
                parts = [f"[question]: {question}"]
                parts.append(f"[context]:\n{context}")
                user_block = "\n\n".join(parts)
                messages = [
                    Message(role="system", content=simple_chat_instruction),
                    Message(role="user", content=user_block),
                ]
        else:
            # Reflexion strategy
            if question_decomposition is None:
                if fewshot_example is None:
                    messages = [
                        Message(role="system", content=reflexion_chat_instruction),
                        Message(
                            role="user",
                            content=(
                                f"[question]: {question}\n"
                                f"[previous answer]: {prev_answers}\n"
                                f"[why wrong]: {self_reflection}\n"
                                f"[feedback]: {feedback}\n"
                                f"[context]:\n{context}"
                            ),
                        ),
                    ]
                else:
                    messages = [
                        Message(role="system", content=reflexion_chat_instruction),
                        Message(
                            role="user",
                            content=(
                                f"[question]: {question}\n"
                                f"[previous answer]: {prev_answers}\n"
                                f"[why wrong]: {self_reflection}\n"
                                f"[feedback]: {feedback}\n"
                                f"[context]:\n{context}"
                                f"[demonstrations]: {fewshot_example}\n"
                            ),
                        ),
                    ]
            else:
                messages = [
                    Message(role="system", content=reflexion_chat_instruction),
                    Message(
                        role="user",
                        content=(
                            f"[question]: {question}\n"
                            f"[question intent decomposition]: {question_decomposition}\n"
                            f"[previous answer]: {prev_answers}\n"
                            f"[why wrong]: {self_reflection}\n"
                            f"[feedback]: {feedback}\n"
                            f"[context]:\n{context}"
                        ),
                    ),
                ]    

        out = model.generate_chat(
            messages=messages,
            max_tokens=128,
            temperature=temperature,
            num_comps=num_comps,
        )

    # --- Completion models ---
    else:
        raise NotImplementedError(
            "Multihop-QA generation is not implemented for completion models."
        )

    # --- Normalize outputs ---
    if num_comps == 1:
        return out.strip() if isinstance(out, str) else out[0].strip()
    else:
        return [ans.strip() for ans in out]  # type: ignore


def multihopqa_generate_self_reflection_parametric(
    question: str,
    answer: str,
    context: str,
    feedback: str,
    insights: str,
    model: ModelBase,
) -> str:
    """Produce a concise self-reflection for a wrong multi-hop QA answer.

    Given the previous `answer`, the supporting `context`, and `feedback` on why
    the answer was unsatisfactory, return a short reflection identifying the mistake."""
    system_prompt = (
        "You are a self-reflection assistant for multi-hop QA.  "
        "Given a question, using the extracted insights, the provided context, your previous answer, and feedback, "
        "write a concise reflection that pinpoints the reasoning error or omission."
    )
    user_content = (
        f"[question]: {question}\n"
        f"[question intent]: {insights}\n"
        f"[context]:\n{context}\n"
        f"[answer]: {answer}\n"
        f"[feedback]: {feedback}"
    )
    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_content),
    ]
    out = model.generate_chat(
        messages=messages,
        max_tokens=500,
        temperature=0.7,
        num_comps=1,
    )
    # normalize
    return out.strip() if isinstance(out, str) else out[0].strip()


def multihopqa_generate_self_reflection(
    question: str,
    answer: str,
    context: str,
    feedback: str,
    model: ModelBase,
) -> str:
    """Produce a concise self-reflection for a wrong multi-hop QA answer.

    Given the question, previous answer, the supporting context, and `feedback` on why
    the answer was unsatisfactory, return a short reflection identifying the mistake."""
    system_prompt = (
        "You are a self-reflection assistant for multi-hop QA.  "
        "Given the question, using only the provided context, your previous answer, and feedback, "
        "write a concise reflection that pinpoints the reasoning error or omission."
    )
    user_content = (
        f"[question]: {question}\n"
        f"[answer]: {answer}\n"
        f"[context]:\n{context}\n"
        f"[feedback]: {feedback}"
    )
    
    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_content),
    ]
    out = model.generate_chat(
        messages=messages,
        max_tokens=500,
        temperature=0.7,
        num_comps=1,
    )
    # normalize
    return out.strip() if isinstance(out, str) else out[0].strip()


def multihopqa_generate_self_reflection_diverse(
    question: str,
    answer: str,
    context: str,
    feedback: str,
    model: ModelBase,
    diverse_reflections: list,
) -> str:
    """Generate and append new self-reflections for a wrong answer.

    Parameters
    ----------
    answer : str
        The previous incorrect answer.
    context : str
        Supporting context paragraphs.
    feedback : str
        Explanation why the answer was unsatisfactory.
    model : ModelBase
        Chat model wrapper.
    diverse_reflections : list
        A list of string containing prior reflections, one per line.

    Returns
    -------
    str
        Combined reflections: existing ones plus up to three new distinct reflections."""
        
        
    reflections_history = ""
    if len(diverse_reflections):
        for idx, ref in enumerate(diverse_reflections, 1):
            reflections_history += f"{idx}: {ref}. \n"        
        
    system_prompt = (
        "You are a self-reflection assistant for multi-hop QA.  "
        "Given the question, previous answer, context, feedback, and existing reflections, "
        "produce **no more than 5** new concise reflections, **each on its own line (use `\n\n` to split)**, "
        "highlighting distinct and diverse reasoning flaws."
    )
    user_content = (
        f"[question]: {question}"
        f"[answer]: {answer}"
        f"[context]:{context}"
        f"[feedback]: {feedback}"
        f"[previous reflections]:{reflections_history}"    
    )
    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_content),
    ]
    out = model.generate_chat(
        messages=messages,
        max_tokens=400,
        temperature=0.7,
        num_comps=1,
    )

    # Normalize output into lines
    new_text = out.strip() if isinstance(out, str) else out[0].strip()
    new_lines = [line.strip() for line in new_text.splitlines() if line.strip()]
    # Deduplicate against existing reflections
    prev_lines = [line.strip() for line in reflections_history.splitlines() if line.strip()]
    additions = [ln for ln in new_lines if ln not in prev_lines]
    # Limit to first 3 new reflections
    additions = additions[:2]
    # Return concatenated existing + new
    combined = prev_lines + additions
    return "\n".join(combined)


def multihopqa_generate_self_reflection_diverse_parametric(
    question: str,  
    answer: str,
    context: str,
    feedback: str,
    model: ModelBase,
    diverse_reflections: list,
    insights: str,
) -> str:
    """Generate and append new self-reflections for a wrong answer.

    Parameters
    ----------
    answer : str
        The previous incorrect answer.
    context : str
        Supporting context paragraphs.
    feedback : str
        Explanation why the answer was unsatisfactory.
    model : ModelBase
        Chat model wrapper.
    diverse_reflections : str
        A string containing prior reflections, one per line.

    Returns
    -------
    str
        Combined reflections: existing ones plus up to three new distinct reflections."""
        

    reflections_history = ""
    if len(diverse_reflections):
        for idx, ref in enumerate(diverse_reflections, 1):
            reflections_history += f"{idx}: {ref}. \n" 

    system_prompt = (
        "You are a self-reflection assistant for multi-hop QA.  "
        "Given the question, some extracted insights for the question, previous answer, context, feedback, and existing reflections, "
        "produce **no more than 5** new concise reflections, **each on its own line (use `\n\n` to split)**, "
        "highlighting distinct and diverse reasoning flaws."
    )
    user_content = (
        f"[question]: {question}"
        f"[question intent]: {insights}"
        f"[answer]: {answer}"
        f"[context]:{context}"
        f"[feedback]: {feedback}"
        f"[existing reflections]:{reflections_history}"    
    )
    
    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_content),
    ]
    out = model.generate_chat(
        messages=messages,
        max_tokens=400,
        temperature=0.7,
        num_comps=1,
    )
    
    # Normalize output into lines
    new_text = out.strip() if isinstance(out, str) else out[0].strip()
    new_lines = [line.strip() for line in new_text.splitlines() if line.strip()]
    # Deduplicate against existing reflections
    prev_lines = [line.strip() for line in reflections_history.splitlines() if line.strip()]
    additions = [ln for ln in new_lines if ln not in prev_lines]
    # Limit to first 3 new reflections
    additions = additions[:2]
    # Return concatenated existing + new
    combined = prev_lines + additions
    return "\n".join(combined)



def generic_generate_mathqa_impl(
    *,
    question: str,
    model: ModelBase,
    strategy: str,                     # "simple" or "reflexion"
    prev_answers: Optional[str],
    feedback: Optional[str],
    self_reflection: Optional[str],
    num_comps: int,
    temperature: float,
    simple_chat_instruction: str,
    reflexion_chat_instruction: str,
    simple_completion_instruction: str,
    reflexion_completion_instruction: str,
    fewshot_example: Optional[str] = None,
    mistake_insights: Optional[str] = None,
) -> Union[str, List[str]]:
    """
    Returns
    -------
    Union[str, List[str]]
        A single answer string (if num_comps==1) or a list of answers.
    """
    
    if strategy not in {"simple", "reflexion"}:
        raise ValueError("strategy must be 'simple' or 'reflexion'")

    if strategy == "reflexion" and (
        prev_answers is None or feedback is None or self_reflection is None
    ):
        raise ValueError(
            "Reflexion strategy requires prev_answers, feedback, and self_reflection"
        )

    math_pitfalls_instruction = (
        "You will be given a mathematics question, and will be provided with some potential mistakes and pitfalls about the queston. You will also possibly be given a demonstration of similar problems. "
        "Using the information provided, think step-by-step and then write a short answer to the question. "
        "❖ The final answer should be simplified to its simplest form,e.g., 25, 2516_8, \frac{1}{36}, etc."
    )

    # --- Chat models ---
    if model.is_chat:
        if strategy == "simple":
            # If we have decomposition or context, inject them
            if mistake_insights:
                print ('use question intent')
                parts = [f"[question]: {question}"]
                if mistake_insights:
                    parts.append(f"[mistake insights]:\n{mistake_insights}")
                user_block = "\n\n".join(parts)
                messages = [
                    Message(role="system", content=math_pitfalls_instruction),
                    Message(role="user", content=user_block),
                ]
            else:
                # Vanilla from-scratch prompt
                parts = [f"[question]: {question}"]
                user_block = "\n\n".join(parts)
                messages = [
                    Message(role="system", content=simple_chat_instruction),
                    Message(role="user", content=user_block),
                ]
        else:
            # Reflexion strategy
            if mistake_insights is None:
                if fewshot_example is None:
                    messages = [
                        Message(role="system", content=reflexion_chat_instruction),
                        Message(
                            role="user",
                            content=(
                                f"[question]: {question}\n"
                                f"[previous answer]: {prev_answers}\n"
                                f"[why wrong]: {self_reflection}\n"
                                f"[feedback]: {feedback}\n"
                            ),
                        ),
                    ]
                else:
                    messages = [
                        Message(role="system", content=reflexion_chat_instruction),
                        Message(
                            role="user",
                            content=(
                                f"[question]: {question}\n"
                                f"[previous answer]: {prev_answers}\n"
                                f"[why wrong]: {self_reflection}\n"
                                f"[feedback]: {feedback}\n"
                                f"[demonstrations]: {fewshot_example}\n"
                            ),
                        ),
                    ]
            else:
                messages = [
                    Message(role="system", content=reflexion_chat_instruction),
                    Message(
                        role="user",
                        content=(
                            f"[question]: {question}\n"
                            f"[mistake insights]: {mistake_insights}\n"
                            f"[previous answer]: {prev_answers}\n"
                            f"[why wrong]: {self_reflection}\n"
                            f"[feedback]: {feedback}"
                        ),
                    ),
                ]

        out = model.generate_chat(
            messages=messages,
            max_tokens=1024,
            temperature=temperature,
            num_comps=num_comps,
        )

    # --- Completion models ---
    else:
        raise NotImplementedError(
            "Multihop-QA generation is not implemented for completion models."
        )

    # --- Normalize outputs ---
    if num_comps == 1:
        return out.strip() if isinstance(out, str) else out[0].strip()
    else:
        return [ans.strip() for ans in out]  # type: ignore


def mathqa_generate_self_reflection(
    question: str,
    answer: str,
    feedback: str,
    model: ModelBase,
) -> str:
    """Produce a concise self-reflection for a wrong mathematics problem answer.

    Given the question, previous answer, and `feedback` on why
    the answer was unsatisfactory, return a short reflection identifying the mistake."""
    system_prompt = (
        "You are a self-reflection assistant for mathematics problem. "
        "Given the question, using your previous answer, and feedback, "
        "write a concise reflection that pinpoints the reasoning error or omission."
    )
    user_content = (
        f"[question]: {question}\n"
        f"[answer]: {answer}\n"
        f"[feedback]: {feedback}"
    )
    
    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_content),
    ]
    out = model.generate_chat(
        messages=messages,
        max_tokens=512,
        temperature=0.7,
        num_comps=1,
    )
    # normalize
    return out.strip() if isinstance(out, str) else out[0].strip()


def mathqa_generate_self_reflection_diverse(
    question: str,
    answer: str,
    feedback: str,
    model: ModelBase,
    diverse_reflections: list,
    fewshot_example = None
) -> str:
    """Generate and append new self-reflections for a wrong answer.

    Parameters
    ----------
    answer : str
        The previous incorrect answer.
    feedback : str
        Explanation why the answer was unsatisfactory.
    model : ModelBase
        Chat model wrapper.
    diverse_reflections : list
        A list of string containing prior reflections, one per line.

    Returns
    -------
    str
        Combined reflections: existing ones plus up to three new distinct reflections."""    
    
    reflections_history = ""
    if len(diverse_reflections):
        for idx, ref in enumerate(diverse_reflections, 1):
            reflections_history += f"{idx}: {ref}. \n"        
    
    system_prompt = (
        "You are a self-reflection assistant for mathematics problem. "
        "Given the question, previous answer, feedback, and existing reflections, and possibly some examples from other problems, "
        "produce **no more than 5** new concise reflections, **each on its own line (use `\n` to split)**, "
        "highlighting distinct and diverse reasoning flaws."
    )
    if fewshot_example is None:
        user_content = (
            f"[question]: {question}"
            f"[answer]: {answer}"
            f"[feedback]: {feedback}"
            f"[previous reflections]:{reflections_history}"    
        )
    else:
        user_content = (
            f"[question]: {question}"
            f"[answer]: {answer}"
            f"[feedback]: {feedback}"
            f"[previous reflections]:{reflections_history}"    
            f"[demonstration from similar problems]:{fewshot_example}"    
        )
    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_content),
    ]
    out = model.generate_chat(
        messages=messages,
        max_tokens=512,
        temperature=0.7,
        num_comps=1,
    )

    # Normalize output into lines
    new_text = out.strip() if isinstance(out, str) else out[0].strip()
    new_lines = [line.strip() for line in new_text.splitlines() if line.strip()]
    # Deduplicate against existing reflections
    prev_lines = [line.strip() for line in reflections_history.splitlines() if line.strip()]
    additions = [ln for ln in new_lines if ln not in prev_lines]
    # Limit to first 3 new reflections
    additions = additions[:3]
    # Return concatenated existing + new
    combined = prev_lines + additions
    return "\n".join(combined)


def mathqa_generate_self_reflection_diverse_parametric(
    question: str,  
    answer: str,
    feedback: str,
    model: ModelBase,
    diverse_reflections: list,
    insights: str,
) -> str:
    """Generate and append new self-reflections for a wrong answer.

    Parameters
    ----------
    answer : str
        The previous incorrect answer.
    context : str
        Supporting context paragraphs.
    feedback : str
        Explanation why the answer was unsatisfactory.
    model : ModelBase
        Chat model wrapper.
    diverse_reflections : str
        A string containing prior reflections, one per line.

    Returns
    -------
    str
        Combined reflections: existing ones plus up to three new distinct reflections."""
        

    reflections_history = ""
    if len(diverse_reflections):
        for idx, ref in enumerate(diverse_reflections, 1):
            reflections_history += f"{idx}: {ref}. \n" 

    system_prompt = (
        "You are a self-reflection assistant for mathematics problem.  "
        "Given the question, some potential mistakes and pitfalls for the question, previous answer, feedback, and existing reflections, "
        "produce **no more than 5** new concise reflections, **each on its own line (use `\n` to split)**, "
        "highlighting distinct and diverse reasoning flaws."
    )
    user_content = (
        f"[question]: {question}"
        f"[question intent]: {insights}"
        f"[answer]: {answer}"
        f"[feedback]: {feedback}"
        f"[existing reflections]:{reflections_history}"    
    )
    
    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_content),
    ]
    out = model.generate_chat(
        messages=messages,
        max_tokens=400,
        temperature=0.7,
        num_comps=1,
    )
    
    # Normalize output into lines
    new_text = out.strip() if isinstance(out, str) else out[0].strip()
    new_lines = [line.strip() for line in new_text.splitlines() if line.strip()]
    # Deduplicate against existing reflections
    prev_lines = [line.strip() for line in reflections_history.splitlines() if line.strip()]
    additions = [ln for ln in new_lines if ln not in prev_lines]
    # Limit to first 3 new reflections
    additions = additions[:3]
    # Return concatenated existing + new
    combined = prev_lines + additions
    return "\n".join(combined)


from generators.model import ModelBase, Message
import re
from typing import List, Optional, Union
from tenacity import retry, stop_after_attempt, wait_random_exponential

# ───────────────────────────── Utility ──────────────────────────────

def remove_unicode_chars(text: str) -> str:
    """Strip non‑ASCII chars (useful when models emit odd unicode)."""
    return re.sub(r"[^\x00-\x7F]+", "", text)

# ───────────────────── Generic generation helper ─────────────────────




def generic_generate_game24_impl(
    numbers: str,                      # "1 1 11 11"
    model: ModelBase,
    strategy: str,                     # "simple" or "reflexion"
    prev_expr: Optional[str],
    feedback: Optional[str],
    self_reflection: Optional[str],
    num_comps: int,
    temperature: float,
    simple_chat_instruction: str,
    reflexion_chat_instruction: str,
    simple_completion_instruction: str,
    reflexion_completion_instruction: str,
    mistake_insights: Optional[str] = None
) -> Union[str, List[str]]:
    
    """
    Generate a 24-Game expression with or without reflexion.
    """
    if strategy not in {"simple", "reflexion"}:
        raise ValueError("strategy must be 'simple' or 'reflexion'")

    if strategy == "reflexion" and (
        prev_expr is None or feedback is None or self_reflection is None
    ):
        raise ValueError(
            "Reflexion strategy requires prev_expr, feedback, and self_reflection"
        )
    mistake_instructions = (
        "You will be given four numbers and a summary of pitfalls along with "
        "several flawed expressions that fail to make 24. "
        "Based on the failure experiences, using only those four numbers once each, the operators + - * / and parentheses, "
        "write ONE new expression that correctly evaluates to 24. "
        "Return ONLY the expression—no extra text."
    )
    # ========== Chat models ==========
    if model.is_chat:
        # ---------------- SIMPLE ----------------
        if strategy == "simple":
            if mistake_insights:
                # Give the model the flawed‐expressions hints
                messages = [
                    Message(role="system", content=mistake_instructions),
                    Message(
                        role="user",
                        content=(
                            f"[numbers]: {numbers}\n\n"
                            f"[pitfalls & flawed expressions]:\n{mistake_insights}"
                        ),
                    ),
                ]
            else:
                # Vanilla puzzle prompt
                messages = [
                    Message(role="system", content=simple_chat_instruction),
                    Message(role="user", content=f"Numbers: {numbers}"),
                ]
        # ---------------- REFLEXION -------------
        else:
            messages = [
                Message(role="system", content=reflexion_chat_instruction),
                Message(
                    role="user",
                    content=(
                        f"[numbers]: {numbers}\n"
                        f"[previous expression]: {prev_expr}\n"
                        f"[why wrong]: {self_reflection}\n"
                        f"[feedback]: {feedback}"
                    ),
                ),
            ]

        out = model.generate_chat(
            messages=messages,
            max_tokens=128,
            temperature=temperature,
            num_comps=num_comps,
        )

    # ========== Completion models =============
    else:
        if strategy == "simple":
            prompt = f"{simple_completion_instruction}\nNumbers: {numbers}\nExpression:"
        else:
            prompt = (
                f"{reflexion_completion_instruction}\n"
                f"Numbers: {numbers}\n"
                f"Previous: {prev_expr}\n"
                f"Why wrong: {self_reflection}\n"
                f"Feedback: {feedback}\n"
                f"New expression:"
            )

        out = model.generate(
            prompt,
            num_comps=num_comps,
            temperature=temperature,
        )

    # -------- Normalise output(s) -------------
    if num_comps == 1:
        return out.strip() if isinstance(out, str) else out[0].strip()
    else:
        return [expr.strip() for expr in out]  # type: ignore

def game24_generate_self_reflection(
    expr: str,
    feedback: str,
    model: ModelBase,
) -> str:
    """
    Produce one reflection for why `expr` failed to reach 24.
    """
    if model.is_chat:
        messages = [
            Message(role="system", content=G24_REFLECT_SYS),
            Message(
                role="user",
                content=(
                    f"[previous expression]: {expr}\n"
                    f"[feedback]: {feedback}\n\n"
                    "[reflection]:"
                ),
            ),
        ]
        return model.generate_chat(messages=messages, temperature=1.0)  # type: ignore
    else:
        prompt = (
            f"{G24_REFLECT_SYS}\n"
            f"[previous expression]: {expr}\n"
            f"[feedback]: {feedback}\n\n"
            "Reflection:"
        )
        return model.generate(prompt, temperature=1.0)  # type: ignore

# ─────────────────────  DIVERSE SELF-REFLECTION  ──────────────────
def game24_generate_self_reflection_diverse(
    expr: str,
    feedback: str,
    model: ModelBase,
    previous_reflections: List[str],
) -> str:
    """
    Generate a *new* reflection that avoids repeating any of the
    `previous_reflections`.
    """
    reflections_history = ""
    if previous_reflections:
        reflections_history = "\n".join(
            f"{idx}: {ref}" for idx, ref in enumerate(previous_reflections, 1)
        )

    if model.is_chat:
        messages = [
            Message(role="system", content=G24_DIVERSE_REFLECTION_ONESHOT_CHAT_INSTRUCTION),
            Message(
                role="user",
                content=(
                    f"[previous expression]: {expr}\n"
                    f"[feedback]: {feedback}\n"
                    f"[earlier reflections]:\n{reflections_history}\n\n"
                    "[new reflection]:"
                ),
            ),
        ]
        return model.generate_chat(messages=messages, temperature=1.0)  # type: ignore
    else:
        prompt = (
            f"{G24_REFLECT_SYS}\n"
            f"[previous expression]: {expr}\n"
            f"[feedback]: {feedback}\n"
            f"[earlier reflections]:\n{reflections_history}\n\n"
            "New reflection:"
        )
        return model.generate(prompt, temperature=1.0)  # type: ignore


def generic_generate_func_impl(
    func_sig: str,
    model: ModelBase,
    strategy: str,
    prev_func_impl,
    feedback,
    self_reflection,
    num_comps,
    temperature,
    reflexion_chat_instruction: str,
    reflexion_few_shot: str,
    simple_chat_instruction: str,
    reflexion_completion_instruction: str,
    simple_completion_instruction: str,
    code_block_instruction: str,
    parse_code_block: Callable[[str], str],
    add_code_block: Callable[[str], str],
    mistake_insights: Optional[str] = None
) -> Union[str, List[str]]:
    if strategy != "reflexion" and strategy != "simple":
        raise ValueError(
            f"Invalid strategy: given `{strategy}` but expected one of `reflexion` or `simple`")
    if strategy == "reflexion" and (prev_func_impl is None or feedback is None or self_reflection is None):
        raise ValueError(
            f"Invalid arguments: given `strategy=reflexion` but `prev_func_impl`, `feedback`, or `self_reflection` is None")

    if model.is_chat:
        if strategy == "reflexion":
            message = f"{reflexion_few_shot}\n[previous impl]:\n{add_code_block(prev_func_impl)}\n\n[unit test results from previous impl]:\n{feedback}\n\n[reflection on previous impl]:\n{self_reflection}\n\n[improved impl]:\n{func_sig}"
            prompt = f"{reflexion_chat_instruction}\n{code_block_instruction}"
            # func_bodies is a really bad name, as it can also be just 1 string
            # print_messages(prompt, message)
            if mistake_insights is None:
                messages = [
                    Message(
                        role="system",
                        content=prompt,
                    ),
                    Message(
                        role="user", #TODO: check this
                        content=reflexion_few_shot,
                    ),
                    Message(
                        role="assistant",
                        content=add_code_block(prev_func_impl),
                    ),
                    Message(
                        role="user",
                        content=f"[unit test results from previous impl]:\n{feedback}\n\n[reflection on previous impl]:",
                    ),
                    Message(
                        role="assistant",
                        content=self_reflection,
                    ),
                    Message(
                        role="user",
                        content=f"[improved impl]:\n{func_sig}",
                    ),
                ]
            else:
                messages = [
                    Message(
                        role="system",
                        content=prompt,
                    ),
                    Message(
                        role="user", #TODO: check this
                        content=reflexion_few_shot,
                    ),
                    Message(
                        role="assistant",
                        content=add_code_block(prev_func_impl),
                    ),
                    Message(
                        role="user",
                        content=f"{mistake_insights}\n[unit test results from previous impl]:\n{feedback}\n\n[reflection on previous impl]:",
                    ),
                    Message(
                        role="assistant",
                        content=self_reflection,
                    ),
                    Message(
                        role="user",
                        content=f"[improved impl]:\n{func_sig}",
                    ),
                ]
            
            func_bodies = model.generate_chat(messages=messages, num_comps=num_comps, temperature=temperature)
        else:
            if mistake_insights is None:
                system_prompt = f"{simple_chat_instruction}\n{code_block_instruction}"
                print_messages(system_prompt, func_sig)
                messages = [
                    Message(
                        role="system",
                        content=f"{simple_chat_instruction}\n{code_block_instruction}",
                    ),
                    Message(
                        role="user",
                        content=func_sig,
                    ),
                ]
            else:
                mistake_instructions = "You will be given a function signature and its docstring, along with a set of summary hints that highlight common pitfalls and several flawed implementation examples. Based on these hints and flawed examples, write a correct implementation of the function. You are an AI that responds only with Python code—no English. Restate the full function signature, and provide the complete implementation."
                system_prompt = f"{mistake_instructions}\n{code_block_instruction}"
                print_messages(system_prompt, func_sig)
                messages = [
                    Message(
                        role="system",
                        content=f"{mistake_instructions}\n{code_block_instruction}",
                    ),
                    Message(
                        role="user",
                        content=f"{mistake_insights}\n [func signature]: {func_sig}",
                    ),
                ]
            
            func_bodies = model.generate_chat(messages=messages, num_comps=num_comps, temperature=temperature)          
    else:
        if strategy == "reflexion":
            prompt = f"{reflexion_completion_instruction}\n{add_code_block(prev_func_impl)}\n\nunit tests:\n{feedback}\n\nhint:\n{self_reflection}\n\n# improved implementation\n{func_sig}\n{code_block_instruction}"
            func_bodies = model.generate(
                prompt, num_comps=num_comps, temperature=temperature)
        else:
            prompt = f"{simple_completion_instruction}\n{func_sig}\n{code_block_instruction}"
            func_bodies = model.generate(
                prompt, num_comps=num_comps, temperature=temperature)

    if num_comps == 1:
        assert isinstance(func_bodies, str)

        # Debug: Log raw LLM response before parsing
        print(f"DEBUG: Raw LLM response before parsing (first 500 chars):")
        print(f"{func_bodies[:500] if func_bodies else 'EMPTY/NONE'}")
        print("=" * 60)

        func_body_str = parse_code_block(func_bodies)

        # Debug: Log parsing result
        if func_body_str is None:
            print(f"WARNING: parse_code_block returned None!")
            print(f"Full raw response:\n{func_bodies}")
            print("=" * 60)

        print_generated_func_body(func_body_str)
        return func_body_str

    else:
        func_bodies = [parse_code_block(func_body) for func_body in func_bodies]
        print_generated_func_body("\n\n".join(func_bodies))
        return func_bodies


def generic_generate_internal_tests(
        func_sig: str,
        model: ModelBase,
        max_num_tests: int,
        test_generation_few_shot: str,
        test_generation_chat_instruction: str,
        test_generation_completion_instruction: str,
        parse_tests: Callable[[str], List[str]],
        is_syntax_valid: Callable[[str], bool],
        is_react: bool = False
) -> List[str]:
    """Generates tests for a function."""
    
    if model.is_chat:
    
        if is_react:
            messages = [
                Message(
                    role="system",
                    content=test_generation_chat_instruction,
                ),
                Message(
                    role="user",
                    content=f"{test_generation_few_shot}\n\n[func signature]:\n{func_sig}\n\n[think]:"
                )
            ]
            
            output = model.generate_chat(messages=messages, max_tokens=1024)
            print(f'React test generation output: {output}')
        else:
            messages = [
                Message(
                    role="system",
                    content=test_generation_chat_instruction,
                ),
                Message(
                    role="user",
                    content=f"{test_generation_few_shot}\n\n[func signature]:\n{func_sig}\n\n[unit tests]:",
                )
            ]
            # messages = [
            #     Message(
            #         role="system",
            #         content=f"{test_generation_chat_instruction}\n\n{test_generation_few_shot}",
            #     ),
            #     Message(
            #         role="user",
            #         content=f"[func signature]:\n{func_sig}\n\n[unit tests]:",
            #     )
            # ]
            output = model.generate_chat(messages=messages, max_tokens=1024)

    else:
        prompt = f'{test_generation_completion_instruction}\n\nfunc signature:\n{func_sig}\nunit tests:'
        output = model.generate(prompt, max_tokens=1024)
    all_tests = parse_tests(output)  # type: ignore
    valid_tests = [test for test in all_tests if is_syntax_valid(test)]

    return sample_n_random(valid_tests, max_num_tests)

def generate_self_reflection_diverse_oneshot(
        func: str,
        feedback: str,
        model: ModelBase,
        self_reflection_chat_instruction: str,
        add_code_block: Callable[[str], str],
        self_reflection_few_shot: Optional[str] = None,
        previous_reflections: list = None
) -> str:

    # concat previous reflections
    reflections_history = ""
    if len(previous_reflections):
        for idx, ref in enumerate(previous_reflections, 1):
            reflections_history += f"{idx}: {ref}. \n"


    if model.is_chat:
        if self_reflection_few_shot is not None:
            messages = [
                Message(
                    role="system",
                    content=self_reflection_chat_instruction,
                ),
                Message(
                    role="user",
                    content=f'{self_reflection_few_shot}\n\n[function impl]:\n{add_code_block(func)}\n\n[unit test results]:\n{feedback}\n\n[Previously generated reflections:]\n{reflections_history}\n\n[self-reflection]:',
                )
            ]
            reflection = model.generate_chat(messages=messages, temperature=0.7)
            
            print(f'Self reflection output: {reflection}')
        else:
            messages = [
                Message(
                    role="system",
                    content=self_reflection_chat_instruction,
                ),
                Message(
                    role="user",
                    content=f'[function impl]:\n{add_code_block(func)}\n\n[unit test results]:\n{feedback}\n\n[Previously generated reflections:]\n{reflections_history}\n\n[self-reflection]:',
                )
            ]
            reflection = model.generate_chat(messages=messages)
    
    return reflection



def generate_self_reflection_diverse_oneshot_parametric(
        func: str,
        feedback: str,
        model: ModelBase,
        self_reflection_chat_instruction: str,
        add_code_block: Callable[[str], str],
        self_reflection_few_shot: Optional[str] = None,
        previous_reflections: list = None,
        mistake_insights = None
) -> str:

    # concat previous reflections
    reflections_history = ""
    if len(previous_reflections):
        for idx, ref in enumerate(previous_reflections, 1):
            reflections_history += f"{idx}: {ref}. \n"

    if model.is_chat:
        if self_reflection_few_shot is not None:
            messages = [
                Message(
                    role="system",
                    content=self_reflection_chat_instruction,
                ),
                Message(
                    role="user",
                    content=f'[Example]: {self_reflection_few_shot}\n\n[Pitfalls]: {mistake_insights}\n\n[function impl]:\n{add_code_block(func)}\n\n[unit test results]:\n{feedback}\n\n[Previously generated reflections:]\n{reflections_history}\n\n[self-reflection]:',
                )
            ]
            reflection = model.generate_chat(messages=messages, temperature=0.7)
            
            print(f'Self reflection output: {reflection}')
        else:
            messages = [
                Message(
                    role="system",
                    content=self_reflection_chat_instruction,
                ),
                Message(
                    role="user",
                    content=f'[function impl]:\n{add_code_block(func)}\n\n[unit test results]:\n{feedback}\n\n[Previously generated reflections:]\n{reflections_history}\n\n[self-reflection]:',
                )
            ]
            reflection = model.generate_chat(messages=messages)
    return reflection

def generate_self_reflection_diverse_oneshot(
        func: str,
        feedback: str,
        model: ModelBase,
        self_reflection_chat_instruction: str,
        add_code_block: Callable[[str], str],
        self_reflection_few_shot: Optional[str] = None,
        previous_reflections: list = None
) -> str:

    # concat previous reflections
    reflections_history = ""
    if len(previous_reflections):
        for idx, ref in enumerate(previous_reflections, 1):
            reflections_history += f"{idx}: {ref}. \n"


    if model.is_chat:
        if self_reflection_few_shot is not None:
            messages = [
                Message(
                    role="system",
                    content=self_reflection_chat_instruction,
                ),
                Message(
                    role="user",
                    content=f'{self_reflection_few_shot}\n\n[function impl]:\n{add_code_block(func)}\n\n[unit test results]:\n{feedback}\n\n[Previously generated reflections:]\n{reflections_history}\n\n[self-reflection]:',
                )
            ]
            reflection = model.generate_chat(messages=messages, temperature=0.7)
            
            print(f'Self reflection output: {reflection}')
        else:
            messages = [
                Message(
                    role="system",
                    content=self_reflection_chat_instruction,
                ),
                Message(
                    role="user",
                    content=f'[function impl]:\n{add_code_block(func)}\n\n[unit test results]:\n{feedback}\n\n[Previously generated reflections:]\n{reflections_history}\n\n[self-reflection]:',
                )
            ]
            reflection = model.generate_chat(messages=messages)
    
    return reflection

def generate_self_reflection_diverse(
        func: str,
        feedback: str,
        model: ModelBase,
        self_reflection_chat_instruction: str,
        add_code_block: Callable[[str], str],
        self_reflection_few_shot: Optional[str] = None,
        previous_reflections: list = None
) -> str:

    # concat previous reflections
    reflections_history = ""
    if len(previous_reflections):
        for idx, ref in enumerate(previous_reflections, 1):
            reflections_history += f"{idx}: {ref}. \n"

    if model.is_chat:
        if self_reflection_few_shot is not None:
            messages = [
                Message(
                    role="system",
                    content=self_reflection_chat_instruction,
                ),
                Message(
                    role="user",
                    content=f'{self_reflection_few_shot}\n\n[function impl]:\n{add_code_block(func)}\n\n[unit test results]:\n{feedback}\n\n[Previously generated reflections:]\n{reflections_history}\n\n[self-reflection]:',
                )
            ]
            reflection = model.generate_chat(messages=messages)
            
            print(f'Self reflection output: {reflection}')
        else:
            messages = [
                Message(
                    role="system",
                    content=self_reflection_chat_instruction,
                ),
                Message(
                    role="user",
                    content=f'[function impl]:\n{add_code_block(func)}\n\n[unit test results]:\n{feedback}\n\n[Previously generated reflections:]\n{reflections_history}\n\n[self-reflection]:',
                )
            ]
            reflection = model.generate_chat(messages=messages)

    return reflection


def generic_generate_self_reflection(
        func: str,
        feedback: str,
        model: ModelBase,
        self_reflection_chat_instruction: str,
        self_reflection_completion_instruction: str,
        add_code_block: Callable[[str], str],
        self_reflection_few_shot: Optional[str] = None,
) -> str:
    if model.is_chat:
        if self_reflection_few_shot is not None:
            messages = [
                Message(
                    role="system",
                    content=self_reflection_chat_instruction,
                ),
                Message(
                    role="user",
                    content=f'{self_reflection_few_shot}\n\n[function impl]:\n{add_code_block(func)}\n\n[unit test results]:\n{feedback}\n\n[self-reflection]:',
                )
            ]
            reflection = model.generate_chat(messages=messages, temperature=1.)
            print(f'Self reflection output: {reflection}')
        else:
            messages = [
                Message(
                    role="system",
                    content=self_reflection_chat_instruction,
                ),
                Message(
                    role="user",
                    content=f'[function impl]:\n{add_code_block(func)}\n\n[unit test results]:\n{feedback}\n\n[self-reflection]:',
                )
            ]
            reflection = model.generate_chat(messages=messages, temperature=1.)
    else:
        reflection = model.generate(
            f'{self_reflection_completion_instruction}\n{add_code_block(func)}\n\n{feedback}\n\nExplanation:', temperature=1.)
    return reflection  # type: ignore


def generic_generate_self_reflection_parametric(
        func: str,
        feedback: str,
        model: ModelBase,
        parametric_insights: str,
        self_reflection_chat_instruction: str,
        self_reflection_completion_instruction: str,
        add_code_block: Callable[[str], str],
        self_reflection_few_shot: Optional[str] = None,
) -> str:
    if model.is_chat:
        if self_reflection_few_shot is not None:
            messages = [
                Message(
                    role="system",
                    content=self_reflection_chat_instruction,
                ),
                Message(
                    role="user",
                    content=f'[Example]:{self_reflection_few_shot}\n\n[Pitfalls]: {parametric_insights}\n\n[function impl]:\n{add_code_block(func)}\n\n[unit test results]:\n{feedback}\n\n[self-reflection]:',
                )
            ]
            reflection = model.generate_chat(messages=messages, temperature=1.)
            print(f'Self reflection output: {reflection}')
        else:
            messages = [
                Message(
                    role="system",
                    content=self_reflection_chat_instruction,
                ),
                Message(
                    role="user",
                    content=f'[function impl]:\n{add_code_block(func)}\n\n[unit test results]:\n{feedback}\n\n[self-reflection]:',
                )
            ]
            reflection = model.generate_chat(messages=messages, temperature=1.)
    else:
        reflection = model.generate(
            f'{self_reflection_completion_instruction}\n{add_code_block(func)}\n\n{feedback}\n\nExplanation:', temperature=1.)
    return reflection  # type: ignore

def sample_n_random(items: List[str], n: int) -> List[str]:
    """Sample min(n, len(items)) random items from a list"""
    assert n >= 0
    if n >= len(items):
        return items
    return random.sample(items, n)


def print_messages(system_message_text: str, user_message_text: str) -> None:
    print(f"""----------------------- SYSTEM MESSAGE -----------------------)
{system_message_text}
----------------------------------------------
----------------------- USER MESSAGE -----------------------
{user_message_text}
----------------------------------------------
""", flush=True)


def print_generated_func_body(func_body_str: str) -> None:
    print(f"""--------------------- GENERATED FUNC BODY ---------------------
{func_body_str}
------------------------------------------""")
