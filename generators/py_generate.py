from generators.model import ModelBase, message_to_str
from .generator_types import Generator
from .generator_utils import generic_generate_func_impl, \
                            generic_generate_internal_tests, \
                            generic_generate_self_reflection, \
                            generate_self_reflection_diverse, \
                            generate_self_reflection_diverse_oneshot, \
                            generic_generate_self_reflection_parametric, \
                             generate_self_reflection_diverse_oneshot_parametric

from typing import Optional, List, Union, Dict, Callable
from together import Together
import os
import ast
import re
from .parse import parse_code_block, add_code_block
from openai import OpenAI
import sys
from tenacity import (
    retry,
    stop_after_attempt,  # type: ignore
    wait_random_exponential,  # type: ignore
)

PY_SIMPLE_COMPLETION_INSTRUCTION = "# Write the body of this function only."
PY_REFLEXION_COMPLETION_INSTRUCTION = "You are a Python writing assistant. You will be given your past function implementation, a series of unit tests, and a hint to change the implementation appropriately. Write your full implementation (restate the function signature).\n\n-----"
PY_SELF_REFLECTION_COMPLETION_INSTRUCTION = "You are a Python writing assistant. You will be given a function implementation and a series of unit tests. Your goal is to write a few sentences to explain why your implementation is wrong as indicated by the tests. You will need this as a hint when you try again later. Only provide the few sentence description in your answer, not the implementation.\n\n-----"
USE_PYTHON_CODEBLOCK_INSTRUCTION = "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"

PY_SIMPLE_CHAT_INSTRUCTION = "You are an AI that only responds with python code, NOT ENGLISH. You will be given a function signature and its docstring by the user. Write your full implementation (restate the function signature)."
# PY_SIMPLE_CHAT_INSTRUCTION = "You are an AI that only responds with python code, NOT ENGLISH. You will be given a function signature and its docstring by the user. Think step by step and carefully, and write your full implementation (restate the function signature)."
PY_SIMPLE_CHAT_INSTRUCTION_V2 = "You are an AI that only responds with only python code. You will be given a function signature and its docstring by the user. Write your full implementation (restate the function signature)."
PY_REFLEXION_CHAT_INSTRUCTION = "You are an AI Python assistant. You will be given your past function implementation, a series of unit tests, and a hint to change the implementation appropriately. Write your full implementation (restate the function signature)."
PY_REFLEXION_CHAT_INSTRUCTION_V2 = "You are an AI Python assistant. You will be given your previous implementation of a function, a series of unit tests results, and your self-reflection on your previous implementation. Write your full implementation (restate the function signature)."
PY_REFLEXION_FEW_SHOT_ADD = '''Example 1:
[previous impl]:
```python
def add(a: int, b: int) -> int:
    """
    Given integers a and b, return the total value of a and b.
    """
    return a - b
```

[unit test results from previous impl]:
Tested passed:

Tests failed:
assert add(1, 2) == 3 # output: -1
assert add(1, 2) == 4 # output: -1

[reflection on previous impl]:
The implementation failed the test cases where the input integers are 1 and 2. The issue arises because the code does not add the two integers together, but instead subtracts the second integer from the first. To fix this issue, we should change the operator from `-` to `+` in the return statement. This will ensure that the function returns the correct output for the given input.

[improved impl]:
```python
def add(a: int, b: int) -> int:
    """
    Given integers a and b, return the total value of a and b.
    """
    return a + b
```
'''


PY_SELF_REFLEXION_PARAMETRIC_FEWSHOT_FUNC_IMPL = '''Example:
[Function Signature]:
from typing import *

def longest_subarray_with_sum_limit(nums: List[int], target: int) -> List[int]:
    """
    Return the longest **contiguous** subarray of `nums` whose elements sum to at most `target`.

    • If several subarrays tie for maximum length, return the **left‑most**.
    • If no valid subarray exists, return the empty list `[]`.
    • The input may contain negative as well as positive integers.

    Complexity requirements: time O(n), auxiliary space O(1).
    """

[Pitfalls]:
1. **No‑solution case** — must return `[]`, not `[x]` or `None`.
2. **Length update rule** — use strictly greater (`>`); otherwise, a later equal‑length window overwrites the earlier left‑most one.
3. **Negatives in the window** — shrinking only while `current_sum > target` can leave an over‑target sum if later negatives cancel it.

[Flawed Implementations]:
```python
from typing import List

def longest_subarray_with_sum_limit_v1(nums: List[int], target: int) -> List[int]:
    # BUG 1 – returns the first acceptable window and stops searching.
    current_sum = 0
    start = 0
    for end, x in enumerate(nums):
        current_sum += x
        if current_sum <= target:
            return nums[start:end + 1]
        while current_sum > target and start <= end:
            current_sum -= nums[start]
            start += 1
    return []


def longest_subarray_with_sum_limit_v2(nums: List[int], target: int) -> List[int]:
    # BUG 2 – uses >= when updating the best window, so later ties clobber the left‑most.
    left = 0
    best: List[int] = []
    cur_sum = 0
    for right, x in enumerate(nums):
        cur_sum += x
        while cur_sum > target and left <= right:
            cur_sum -= nums[left]
            left += 1
        if right - left + 1 >= len(best):  # ← should be >
            best = nums[left:right + 1]
    return best


def longest_subarray_with_sum_limit_v3(nums: List[int], target: int) -> List[int]:
    # BUG 3 – never shrinks when negatives drop the sum again; can keep an over‑target window.
    left = 0
    best = []
    cur_sum = 0
    for right, x in enumerate(nums):
        cur_sum += x
        if cur_sum <= target and right - left + 1 > len(best):
            best = nums[left:right + 1]
        # Missing loop to shrink when cur_sum > target
    return best


def longest_subarray_with_sum_limit_v4(nums: List[int], target: int) -> List[int]:
    # BUG 4 – brute‑force O(n²) scan; correct but violates efficiency constraint.
    n = len(nums)
    best: List[int] = []
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += nums[j]
            if s <= target and j - i + 1 > len(best):
                best = nums[i:j + 1]
    return best
```

[previous impl]:
```python
from typing import List

def longest_subarray_with_sum_limit(nums: List[int], target: int) -> List[int]:
    n = len(nums)
    left, right = 0, 0
    max_length = 0
    current_sum = 0
    result: List[int] = []
    while right < n:
        current_sum += nums[right]
        while current_sum > target and left <= right:
            current_sum -= nums[left]
            left += 1
        if right - left + 1 >= max_length:
            max_length = right - left + 1
            result = nums[left:right + 1]
        right += 1
    return result
```

[unit test results from previous impl]:
```
# Tests passed:
assert longest_subarray_with_sum_limit([1, 2, 3, 4, 5], 8) == [1, 2, 3]
assert longest_subarray_with_sum_limit([1, 2, 3, 4, 5], 15) == [1, 2, 3, 4, 5]
assert longest_subarray_with_sum_limit([1, -1, 2, -2, 3, -3], 2) == [1, -1, 2, -2, 3]
assert longest_subarray_with_sum_limit([], 10) == []
assert longest_subarray_with_sum_limit([], 0) == []
assert longest_subarray_with_sum_limit([], -5) == []

# Tests failed:
assert longest_subarray_with_sum_limit([5, 6, 7, 8, 9], 4) == []  # got [5]
```

[reflection on previous impl]:
After the inner while‑loop collapses the window (`left` surpasses `right`), the algorithm still evaluates `right - left + 1`, which can be zero or negative. Since `max_length` starts at zero, this zero‑length "window" can overwrite `result` with a stale slice captured *before* contraction finished. A robust fix is to **skip the length‑update step whenever `left > right`** or to reset `current_sum` and synchronise `left = right` when the window collapses. This prevents invalid zero‑length comparisons and preserves the correct empty result when no feasible subarray exists.

[improved impl]:
```python
from typing import List

def longest_subarray_with_sum_limit(nums: List[int], target: int) -> List[int]:
    """Improved: fixes zero‑length window overwrite and ≥ bug, O(n) time, O(1) space."""
    left = 0
    cur_sum = 0
    best_len = 0
    best_left = 0

    for right, val in enumerate(nums):
        cur_sum += val
        # Shrink window until sum ≤ target
        while cur_sum > target and left <= right:
            cur_sum -= nums[left]
            left += 1

        # Only consider non‑empty window and strictly longer length
        if left <= right and (right - left + 1) > best_len:
            best_len = right - left + 1
            best_left = left

    return nums[best_left:best_left + best_len]
```

END OF EXAMPLE
'''



PY_REFLEXION_FEW_SHOT_WITH_PITFALLS = '''Example 1:
[previous impl]:
```python
from typing import *
def fullJustify(words: List[str], maxWidth: int) -> List[str]:
    """
    Given an array of words and a width maxWidth, format the text such that each line has exactly maxWidth characters and is fully (left and right) justified.
    You should pack your words in a greedy approach; that is, pack as many words as you can in each line. Pad extra spaces `' '` when necessary so that each line has exactly maxWidth characters.
    Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a line do not divide evenly between words, the empty slots on the left will be assigned more spaces than the slots on the right.
    For the last line of text, it should be left justified and no extra space is inserted between words.
    Note:
    A word is defined as a character sequence consisting of non-space characters only.
    Each word's length is guaranteed to be greater than 0 and not exceed maxWidth.
    The input array `words` contains at least one word.
    """
    res = []
    cur_line = []
    cur_len = 0

    for word in words:
        if cur_len + len(word) + len(cur_line) > maxWidth:
            if len(cur_line) == 1:
                res.append(cur_line[0] + ' ' * (maxWidth - cur_len))
            else:
                spaces = maxWidth - cur_len
                space_between = spaces // (len(cur_line) - 1)
                extra_spaces = spaces % (len(cur_line) - 1)
                line = ''
                for i, w in enumerate(cur_line[:-1]):
                    line += w + ' ' * (space_between + (i < extra_spaces))
                line += cur_line[-1]
                res.append(line)
            cur_line = []
            cur_len = 0
        cur_line.append(word)
        cur_len += len(word)

    last_line = ' '.join(cur_line)
    last_line += ' ' * (maxWidth - len(last_line))
    res.append(last_line)

    return res
```

[unit test results from previous impl]:
Tested passed:

Tests failed:
assert fullJustify([], 10) == [] # output: ['          ']
assert fullJustify([], 0) == [] # output: ['']

[reflection on previous impl]:
The implementation failed the test cases where the input list of words is empty. The issue arises because the code does not handle the case where there are no words to process. As a result, it still appends a line with spaces to the result list, even when there are no words. To fix this issue, we should add a condition at the beginning of the function to check if the input list is empty, and return an empty list if it is. This will ensure that the function returns the correct output for empty input lists.

[improved impl]:
```python
from typing import *
def fullJustify(words: List[str], maxWidth: int) -> List[str]:
    """
    Given an array of words and a width maxWidth, format the text such that each line has exactly maxWidth characters and is fully (left and right) justified.
    You should pack your words in a greedy approach; that is, pack as many words as you can in each line. Pad extra spaces `' '` when necessary so that each line has exactly maxWidth characters.
    Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a line do not divide evenly between words, the empty slots on the left will be assigned more spaces than the slots on the right.
    For the last line of text, it should be left justified and no extra space is inserted between words.
    Note:
    A word is defined as a character sequence consisting of non-space characters only.
    Each word's length is guaranteed to be greater than 0 and not exceed maxWidth.
    The input array `words` contains at least one word.
    """
    if not words:
        return []

    res = []
    cur_line = []
    cur_len = 0

    for word in words:
        if cur_len + len(word) + len(cur_line) > maxWidth:
            if len(cur_line) == 1:
                res.append(cur_line[0] + ' ' * (maxWidth - cur_len))
            else:
                spaces = maxWidth - cur_len
                space_between = spaces // (len(cur_line) - 1)
                extra_spaces = spaces % (len(cur_line) - 1)
                line = ''
                for i, w in enumerate(cur_line[:-1]):
                    line += w + ' ' * (space_between + (i < extra_spaces))
                line += cur_line[-1]
                res.append(line)
            cur_line = []
            cur_len = 0
        cur_line.append(word)
        cur_len += len(word)

    last_line = ' '.join(cur_line)
    last_line += ' ' * (maxWidth - len(last_line))
    res.append(last_line)

    return res
```
END EXAMPLES

'''
PY_SELF_REFLECTION_CHAT_INSTRUCTION = "You are a Python programming assistant. You will be given a function implementation and a series of unit tests. Your goal is to write a few sentences to explain why your implementation is wrong as indicated by the tests. You will need this as a hint when you try again later. Only provide the few sentence description in your answer, not the implementation."

PY_SELF_REFLECTION_PARAMETRIC_CHAT_INSTRUCTION = (
    "You are a Python programming assistant. You will be provided with: "
    "1) A set of potential pitfalls and flawed implementation examples for the given coding task; "
    "2) A candidate function implementation along with a suite of unit tests. "
    "Your objective is to write a concise explanation describing why the implementation fails, as indicated by the test results. "
    "You reflection should not overlap with the given pitfalls, and should bring new information to help identify the error. Only provide the few sentence description in your answer, not the implementation."
)


PY_SELF_REFLEXION_PARAMETRIC_FEW_SHOT = '''Example:
[Function Signature]:
from typing import *

def longest_subarray_with_sum_limit(nums: List[int], target: int) -> List[int]:
    """
    Return the longest **contiguous** subarray of `nums` whose elements sum to at most `target`.

    • If several subarrays tie for maximum length, return the **left‑most**.
    • If no valid subarray exists, return the empty list `[]`.
    • The input may contain negative as well as positive integers.

    Complexity requirements: time O(n), auxiliary space O(1).
    """

[Pitfalls]:
1. **No‑solution case** — must return `[]`, not `[x]` or `None`.
2. **Length update rule** — use strictly greater (`>`); otherwise, a later equal‑length window overwrites the earlier left‑most one.
3. **Negatives in the window** — shrinking only while `current_sum > target` can leave an over‑target sum if later negatives cancel it.

[Flawed Implementations]:
```python
from typing import List

def longest_subarray_with_sum_limit_v1(nums: List[int], target: int) -> List[int]:
    # BUG 1 – returns the first acceptable window and stops searching.
    current_sum = 0
    start = 0
    for end, x in enumerate(nums):
        current_sum += x
        if current_sum <= target:
            return nums[start:end + 1]
        while current_sum > target and start <= end:
            current_sum -= nums[start]
            start += 1
    return []


def longest_subarray_with_sum_limit_v2(nums: List[int], target: int) -> List[int]:
    # BUG 2 – uses >= when updating the best window, so later ties clobber the left‑most.
    left = 0
    best: List[int] = []
    cur_sum = 0
    for right, x in enumerate(nums):
        cur_sum += x
        while cur_sum > target and left <= right:
            cur_sum -= nums[left]
            left += 1
        if right - left + 1 >= len(best):  # ← should be >
            best = nums[left:right + 1]
    return best


def longest_subarray_with_sum_limit_v3(nums: List[int], target: int) -> List[int]:
    # BUG 3 – never shrinks when negatives drop the sum again; can keep an over‑target window.
    left = 0
    best = []
    cur_sum = 0
    for right, x in enumerate(nums):
        cur_sum += x
        if cur_sum <= target and right - left + 1 > len(best):
            best = nums[left:right + 1]
        # Missing loop to shrink when cur_sum > target
    return best


def longest_subarray_with_sum_limit_v4(nums: List[int], target: int) -> List[int]:
    # BUG 4 – brute‑force O(n²) scan; correct but violates efficiency constraint.
    n = len(nums)
    best: List[int] = []
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += nums[j]
            if s <= target and j - i + 1 > len(best):
                best = nums[i:j + 1]
    return best
```

[previous impl]:
```python
from typing import List

def longest_subarray_with_sum_limit(nums: List[int], target: int) -> List[int]:
    n = len(nums)
    left, right = 0, 0
    max_length = 0
    current_sum = 0
    result: List[int] = []
    while right < n:
        current_sum += nums[right]
        while current_sum > target and left <= right:
            current_sum -= nums[left]
            left += 1
        if right - left + 1 >= max_length:
            max_length = right - left + 1
            result = nums[left:right + 1]
        right += 1
    return result
```

[unit test results from previous impl]:
```
# Tests passed:
assert longest_subarray_with_sum_limit([1, 2, 3, 4, 5], 8) == [1, 2, 3]
assert longest_subarray_with_sum_limit([1, 2, 3, 4, 5], 15) == [1, 2, 3, 4, 5]
assert longest_subarray_with_sum_limit([1, -1, 2, -2, 3, -3], 2) == [1, -1, 2, -2, 3]
assert longest_subarray_with_sum_limit([], 10) == []
assert longest_subarray_with_sum_limit([], 0) == []
assert longest_subarray_with_sum_limit([], -5) == []

# Tests failed:
assert longest_subarray_with_sum_limit([5, 6, 7, 8, 9], 4) == []  # got [5]
```

[reflection on previous impl]:
After the inner while‑loop collapses the window (`left` surpasses `right`), the algorithm still evaluates `right - left + 1`, which can be zero or negative. Since `max_length` starts at zero, this zero‑length "window" can overwrite `result` with a stale slice captured *before* contraction finished. A robust fix is to **skip the length‑update step whenever `left > right`** or to reset `current_sum` and synchronise `left = right` when the window collapses. This prevents invalid zero‑length comparisons and preserves the correct empty result when no feasible subarray exists.

END OF EXAMPLE
'''



PY_SELF_REFLECTION_CHAT_INSTRUCTION_V2 = "You are a Python programming assistant. You will be given a function implementation and a series of unit test results. Your goal is to write a few sentences to explain why your implementation is wrong as indicated by the tests. You will need this as guidance when you try again later. Only provide the few sentence description in your answer, not the implementation. You will be given a few examples by the user."
PY_SELF_REFLECTION_FEW_SHOT = """Example:
[function impl]:
```python
def longest_subarray_with_sum_limit(nums: List[int], target: int) -> List[int]:
    n = len(nums)
    left, right = 0, 0
    max_length = 0
    current_sum = 0
    result = []
    while right < n:
        current_sum += nums[right]
        while current_sum > target:
            current_sum -= nums[left]
            left += 1
        if right - left + 1 >= max_length:
            max_length = right - left + 1
            result = nums[left:right+1]
        right += 1
    return result
```
[unit test results]:
Tests passing:
assert longest_subarray_with_sum_limit([1, 2, 3, 4, 5], 8) == [1, 2, 3]
assert longest_subarray_with_sum_limit([1, 2, 3, 4, 5], 15) == [1, 2, 3, 4, 5]
assert longest_subarray_with_sum_limit([1, -1, 2, -2, 3, -3], 2) == [1, -1, 2, -2, 3]
assert longest_subarray_with_sum_limit([], 10) == []
assert longest_subarray_with_sum_limit([], 0) == []
assert longest_subarray_with_sum_limit([], -5) == []  
Tests failing:
assert longest_subarray_with_sum_limit([5, 6, 7, 8, 9], 4) == [] # output: [5]
[self-reflection]:
The implementation failed the where no subarray fulfills the condition. The issue in the implementation is due to the use of >= instead of > in the condition to update the result. Because of this, it returns a subarray even when the sum is greater than the target, as it still updates the result when the current subarray length is equal to the previous longest subarray length. To overcome this error, we should change the condition to only update the result when the current subarray length is strictly greater than the previous longest subarray length. This can be done by replacing >= with > in the condition.

END OF EXAMPLES
"""

PY_TEST_GENERATION_FEW_SHOT = """Examples:
func signature:
def add3Numbers(x, y, z):
    \"\"\" Add three numbers together.
    This function takes three numbers as input and returns the sum of the three numbers.
    \"\"\"
unit tests:
assert add3Numbers(1, 2, 3) == 6
assert add3Numbers(-1, 2, 3) == 4
assert add3Numbers(1, -2, 3) == 2
assert add3Numbers(1, 2, -3) == 0
assert add3Numbers(-3, -2, -1) == -6
assert add3Numbers(0, 0, 0) == 0
"""

PY_TEST_GENERATION_COMPLETION_INSTRUCTION = f"""You are an AI coding assistant that can write unique, diverse, and intuitive unit tests for functions given the signature and docstring.

{PY_TEST_GENERATION_FEW_SHOT}"""

PY_TEST_GENERATION_CHAT_INSTRUCTION = """You are an AI coding assistant that can write unique, diverse, and intuitive unit tests for functions given the signature and docstring."""


# ----------------------------------
# Added these prompts
# ----------------------------------

PY_TEST_EXTRACTION_CHAT_INSTRUCTION = \
"""You are an AI coding assistant that can extract unit tests from a given function signature and docstring. 
The docstring includes example use cases starting after the string "Examples:". 
Additionally, you can synthesize extra valid unit tests based on the information from the function signature and docstring."""

PY_EXTRACT_TEST_CASES_INSTRUCTION  = \
"""
Given a Python function signature along with a docstring that describes the 
function's purpose and provides example use cases starting after the string 
"Examples:", generate unit tests using assert statements based on the provided examples. 
Additionally, synthesize extra valid unit tests using the information from 
the function signature and the docstring.

**Input:**

```python
from typing import List

def minPath(grid: List[List[int]], k: int) -> List[int]:
    Given a grid with N rows and N columns (N >= 2) and a positive integer k, 
    each cell of the grid contains a value. Every integer in the range [1, N * N]
    inclusive appears exactly once on the cells of the grid.

    You have to find the minimum path of length k in the grid. You can start
    from any cell, and in each step you can move to any of the neighbor cells,
    in other words, you can go to cells which share an edge with you current
    cell.
    Please note that a path of length k means visiting exactly k cells (not
    necessarily distinct).
    You CANNOT go off the grid.
    A path A (of length k) is considered less than a path B (of length k) if
    after making the ordered lists of the values on the cells that A and B go
    through (let's call them lst_A and lst_B), lst_A is lexicographically less
    than lst_B, in other words, there exist an integer index i (1 <= i <= k)
    such that lst_A[i] < lst_B[i] and for any j (1 <= j < i) we have
    lst_A[j] = lst_B[j].
    It is guaranteed that the answer is unique.
    Return an ordered list of the values on the cells that the minimum path go through.

    Examples:    
    >>> minPath([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 3)
    [1, 2, 1]

    >>> minPath([[5, 9, 3], [4, 1, 6], [7, 8, 2]], 1)
    [1]
```

**Output:**

```python
unit tests:
assert minPath([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 3) == [1, 2, 1]
assert minPath([[5, 9, 3], [4, 1, 6], [7, 8, 2]], 1) == [1]
# Additional synthesized tests
assert minPath([[10, 11, 12], [13, 14, 15], [16, 17, 18]], 2) == [1, 10]
assert minPath([[21, 22, 23], [24, 25, 26], [27, 28, 29]], 4) == [1, 10, 1, 10]
assert minPath([[30, 31, 32], [33, 34, 35], [36, 37, 38]], 1) == [1]
```

**Instructions for the LLM:**

1. Extract the examples provided after the string "Examples:" in the docstring.
2. Convert each example into an assert statement for unit testing.
3. Synthesize additional valid unit tests based on the function signature and docstring.
4. Ensure the assert statements follow this format:
    - `assert function_name(arguments) == expected_output`
5. Use the constraints and descriptions in the docstring to create meaningful additional test cases.
6. The output should be a block of assert statements separated by newline characters.
7. Do not include any additional text or comments in the output.
"""

PY_DIVERSE_REFLECTION_CHAT_INSTRUCTION = \
"""You are a Python programming assistant. 
You will be given a function implementation, unit tests, and previously generated reflections. 
Write a terse and insightful explanation of why the implementation is incorrect based on the tests. 
Ensure your reflections are accurate and leverage previous reflections to avoid repetition. 
Aim for diversity in your explanations while prioritizing the correctness of your hints. 
Only provide the few sentence description in your answer, not the implementation.
"""


PY_DIVERSE_REFLECTION_PARAMETRIC_ONESHOT_CHAT_INSTRUCTION = \
"""You are a Python programming assistant. You will be given some potential pitfalls and flawed implementations, a function implementation, unit tests, and previously generated reflections.
Write 3 unique and diverse reflections to fix the problem. Each reflection should follow this structure:

Problem: {a terse description of the identified problem}.
Fix: {proposed fix or hint to fix the identified problem}.

Ensure your reflections are accurate and leverage previous reflections and pitfalls to avoid repetition and overlap.
Aim for diversity in your explanations while prioritizing the correctness of your hints.
Add "\n\n" at the end of each proposed reflection.
"""


PY_DIVERSE_REFLECTION_PARAMETRIC_FEW_SHOT = \
"""
Example 1:

<Pitfalls>:
1. **Non-strict length comparison**: Using `>=` when updating `max_length` allows updating on equal-length subarrays, so an invalid subarray can override the empty result.  
2. **No explicit empty-result handling**: The code never returns `[]` when no subarray meets the criteria, so `result` may remain populated by a default slice.  
3. **Lack of tie-breaker on sum closeness**: When two subarrays share the same length, the one with sum farther from `target` may be chosen.  
4. **Sliding-window with negatives**: Shrinking the window unconditionally on `current_sum > target` can skip valid longer subarrays if negative numbers are present.  

<Flawed Implementations>:

```python
from typing import *

# v1: uses >= instead of >
def longest_subarray_with_sum_limit_v1(nums: List[int], target: int) -> List[int]:
    left = 0; current_sum = 0; max_length = 0; result = []
    for right in range(len(nums)):
        current_sum += nums[right]
        while current_sum > target:
            current_sum -= nums[left]; left += 1
        # BUG: should be ">" not ">="
        if right - left + 1 >= max_length:
            max_length = right - left + 1
            result = nums[left:right+1]
    return result

# v2: never returns [] when no valid subarray exists
def longest_subarray_with_sum_limit_v2(nums: List[int], target: int) -> List[int]:
    # BUG: no empty check, result starts as []
    left = 0; current_sum = 0; max_length = 0; result = []
    for right in range(len(nums)):
        current_sum += nums[right]
        while current_sum > target:
            current_sum -= nums[left]; left += 1
        if right - left + 1 > max_length:
            max_length = right - left + 1
            result = nums[left:right+1]
    return result  # returns first element slice if every single element > target

# v3: no tie-breaker on closeness
def longest_subarray_with_sum_limit_v3(nums: List[int], target: int) -> List[int]:
    left = 0; current_sum = 0; max_length = 0; result = []
    for right in range(len(nums)):
        current_sum += nums[right]
        while current_sum > target:
            current_sum -= nums[left]; left += 1
        if right - left + 1 > max_length:
            max_length = right - left + 1
            result = nums[left:right+1]
        elif right - left + 1 == max_length:
            # BUG: does not compare abs(sum - target) to choose closer
            result = nums[left:right+1]
    return result

# v4: mishandles negatives
def longest_subarray_with_sum_limit_v4(nums: List[int], target: int) -> List[int]:
    left = 0; current_sum = 0; max_length = 0; result = []
    for right, val in enumerate(nums):
        current_sum += val
        # BUG: always shrink even if val < 0 could bring sum back under target
        while current_sum > target:
            current_sum -= nums[left]
            left += 1
        if right - left + 1 > max_length:
            max_length = right - left + 1
            result = nums[left:right+1]
    return result


[function impl]:
```python
def longest_subarray_with_sum_limit(nums: List[int], target: int) -> List[int]:
    n = len(nums)
    left, right = 0, 0
    max_length = 0
    current_sum = 0
    result = []
    while right < n:
        current_sum += nums[right]
        while current_sum > target:
            current_sum -= nums[left]
            left += 1
        if right - left + 1 >= max_length:
            max_length = right - left + 1
            result = nums[left:right+1]
        right += 1
    return result
```

[unit test results]:
Tests passing:
assert longest_subarray_with_sum_limit([1, 2, 3, 4, 5], 8) == [1, 2, 3]
assert longest_subarray_with_sum_limit([1, 2, 3, 4, 5], 15) == [1, 2, 3, 4, 5]
assert longest_subarray_with_sum_limit([1, -1, 2, -2, 3, -3], 2) == [1, -1, 2, -2, 3]
assert longest_subarray_with_sum_limit([], 10) == []
assert longest_subarray_with_sum_limit([], 0) == []
assert longest_subarray_with_sum_limit([], -5) == []

Tests failing:
assert longest_subarray_with_sum_limit([5, 6, 7, 8, 9], 4) == [] # output: [5]

[self-reflection]:
**Problem**: The implementation failed the test where no subarray fulfills the condition. The issue is due to the use of `>=` instead of `>` in the condition to update the result.  
**Fix**: Change the condition to only update the result when the current subarray length is strictly greater than the previous longest subarray length by replacing `>=` with `>` in the condition.  

**Problem**: The current implementation does not handle the case where the sum of any subarray exceeds the target, leading to incorrect results.  
**Fix**: Add a condition to check if the sum of the subarray exceeds the target before updating the result.  

**Problem**: The algorithm might not correctly identify the longest subarray when multiple subarrays have the same length but different sums.  
**Fix**: Introduce an additional check to ensure that the subarray with the sum closest to the target is selected when lengths are equal.  

**Problem**: The code does not consider the possibility of an empty array or a target that is too small to be achieved by any subarray.  
**Fix**: Implement a preliminary check to return an empty array if the target is smaller than the smallest element in `nums` or if `nums` is empty.  
``` 

END OF EXAMPLES
"""

PY_DIVERSE_REFLECTION_ONESHOT_CHAT_INSTRUCTION = \
"""You are a Python programming assistant. You will be given a function implementation, unit tests, and previously generated reflections.
Write 3 unique and diverse reflections to fix the problem. Each reflection should follow this structure:

Problem: {a terse description of the identified problem}.
Fix: {proposed fix or hint to fix the identified problem}.

Ensure your reflections are accurate and leverage previous reflections to avoid repetition.
Aim for diversity in your explanations while prioritizing the correctness of your hints.
Add "\n\n" at the end of each proposed reflection.
"""

PY_DIVERSE_REFLECTION_FEW_SHOT = \
"""
Example 1:
[function impl]:
```python
def longest_subarray_with_sum_limit(nums: List[int], target: int) -> List[int]:
    n = len(nums)
    left, right = 0, 0
    max_length = 0
    current_sum = 0
    result = []
    while right < n:
        current_sum += nums[right]
        while current_sum > target:
            current_sum -= nums[left]
            left += 1
        if right - left + 1 >= max_length:
            max_length = right - left + 1
            result = nums[left:right+1]
        right += 1
    return result
```
[unit test results]:
Tests passing:
assert longest_subarray_with_sum_limit([1, 2, 3, 4, 5], 8) == [1, 2, 3]
assert longest_subarray_with_sum_limit([1, 2, 3, 4, 5], 15) == [1, 2, 3, 4, 5]
assert longest_subarray_with_sum_limit([1, -1, 2, -2, 3, -3], 2) == [1, -1, 2, -2, 3]
assert longest_subarray_with_sum_limit([], 10) == []
assert longest_subarray_with_sum_limit([], 0) == []
assert longest_subarray_with_sum_limit([], -5) == []
Tests failing:
assert longest_subarray_with_sum_limit([5, 6, 7, 8, 9], 4) == [] # output: [5]
[self-reflection]:
**Problem**: The implementation failed the test where no subarray fulfills the condition. The issue is due to the use of `>=` instead of `>` in the condition to update the result.  
**Fix**: Change the condition to only update the result when the current subarray length is strictly greater than the previous longest subarray length by replacing `>=` with `>` in the condition.  

**Problem**: The current implementation does not handle the case where the sum of any subarray exceeds the target, leading to incorrect results.  
**Fix**: Add a condition to check if the sum of the subarray exceeds the target before updating the result.  

**Problem**: The algorithm might not correctly identify the longest subarray when multiple subarrays have the same length but different sums.  
**Fix**: Introduce an additional check to ensure that the subarray with the sum closest to the target is selected when lengths are equal.  

**Problem**: The code does not consider the possibility of an empty array or a target that is too small to be achieved by any subarray.  
**Fix**: Implement a preliminary check to return an empty array if the target is smaller than the smallest element in `nums` or if `nums` is empty.  
``` 

END OF EXAMPLES
"""

PY_DoT_CHAT_INSTRUCTION = "You are an AI Python assistant. You will be given your previous implementation of a function, a series of unit tests results, and your self-reflection on your previous implementation. ALWAYS WRITE your full implementation (restate the function signature)."

PY_DoT_CHAT_INSTRUCTION_WITH_PITFALLS = "You are an AI Python assistant. You will be some potential pitfalls and several flawed implementations for the coding challenge, and given your previous implementation of a function, a series of unit tests results, and your self-reflection on your previous implementation. Try to avoid errors of your previous implementation and the pitfalls. ALWAYS WRITE your full implementation (restate the function signature)."
# """ 
# You are an AI Python assistant. 
# You will be given your past function implementation, a series of unit tests, and a reflection identifying the problem and suggesting a potential fix. 
# Using the provided reflection, WRITE the full implementation of the function (restate the function signature). 
# Ensure the new implementation addresses the identified problem and incorporates the suggested fix.
# THE OUTPUT SHOULD ALWAYS CONTAIN GENERATED PYTHON CODE ALONG WITH TEST CASES AND THEIR OUTPUT BUT NO TEXT.
# """


PRE_INSIGHT_SYS = "You are an AI assistant for Python coding. Given a function signature and a list of potential pitfalls—originally extracted from other coding problems—carefully evaluate their relevance to the current function. If a pitfall is appropriate, adopt it directly; if not, adapt it to fit the specific context of the function or discard it if inapplicable. Return the refined pre-insights, and up to 6 flawed implementations specific to the function signature that cover as many pitfalls as possible."

PRE_INSIGHT_SYS_NO_MODEL_INSIGHTS = "You are an AI assistant for Python coding. Given a function signature, use your knowledge to propose potential pitfalls for the implementation, and list the possible pitfalls, and generate up to 6 flawed implementations specific to the function signature that cover as many pitfalls as possible."

PY_PRE_INSIGHTS_FEW_SHOT = """Example:
[Function Signature]:
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\"Check if any two numbers in the list are closer than the threshold.\"\"\"

[Refined Pre-Insights for this Signature]:
1. **Empty or Single-Element Lists** must return `False`, not `True`.
2. **Duplicate Values** must be compared (difference 0), so never drop duplicates.
3. Always use **absolute difference** (`abs(a - b)`), not raw subtraction.
4. Use the correct **strictness** (`< threshold`, not `<=`).
5. Ensure you don’t **exit too early**—check all distinct pairs.

[Flawed Implementations Illustrating Each Pitfall]:

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
    
END OF EXAMPLE
"""

class PyGenerator(Generator):
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def generate_pre_insights(
        self,
        func_sign: str,
        pre_insights: str,
        add_code_block: Callable[[str], str]=lambda code: add_code_block(code, "python")
    ) -> str:
        """
        Use GPT-4o-mini to generate precautionary notes and flawed-implementation
        examples for a given function signature, based on preliminary insights.
        """
        # Initialize the GPT-4o-mini client
        client = OpenAI()

        # Build the chat messages
        messages: List[Dict[str, str]] = []

        # 1) System prompt
        messages.append({
            "role": "system",
            "content":  PRE_INSIGHT_SYS
        })

        messages.append({
            "role": "user",
            "content": f"Here is one example.\n {PY_PRE_INSIGHTS_FEW_SHOT}.\n Now, given the function signature and pre-insights, generate the refined pre-insights and flawed implementations.\n [func_sign]:\n {add_code_block(func_sign)}\n[Possible pitfalls]: \n{pre_insights}\n\n"
        })
        
        # messages.append({
        #     "role": "user",
        #     "content": f"Here is one example.\n {PY_PRE_INSIGHTS_FEW_SHOT}.\n Now, given the function signature, generate the pre-insights and flawed implementations.\n [func_sign]:\n {add_code_block(func_sign)}\n\n\n"
        # })

        # Call the model
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        
        # Return the assistant’s reply
        return resp.choices[0].message.content

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def generate_pre_insights_llama3(
        self,
        func_sign: str,
        pre_insights: str,
        add_code_block: Callable[[str], str] = lambda code: add_code_block(code, "python"),
        *,
        model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        temperature: float = 0.5,
        max_tokens: int = 1000,
    ):
        """
        Use TogetherAI Llama-3.1-8B-Instruct-Turbo to produce:
        1) refined pitfalls relevant to `func_sign`
        2) five flawed Python snippets illustrating those pitfalls.
        """
        # ---------- initialise Together client -------------------------
        client = Together(api_key=os.environ.get("TOGETHER_API_KEY", "xxxxx"))

        # ---------- craft chat messages --------------------------------
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": PRE_INSIGHT_SYS},
            {
                "role": "user",
                "content": (
                    "Here is one example.\n"
                    f"{PY_PRE_INSIGHTS_FEW_SHOT}\n\n"
                    "Now, given the function signature and pre-insights, generate the "
                    "refined pre-insights and flawed implementations.\n"
                    "[func_sign]:\n"
                    f"{add_code_block(func_sign)}\n"
                    "[Possible pitfalls]:\n"
                    f"{pre_insights}\n"
                ),
            },
        ]

        # ---------- call Together chat API -----------------------------
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            n=1,
        )

        # ---------- return assistant reply -----------------------------
        return response.choices[0].message.content


    def self_reflection(self, func: str, feedback: str, model: ModelBase) -> str:
        return generic_generate_self_reflection(
            func=func,
            feedback=feedback,
            model=model,
            self_reflection_chat_instruction=PY_SELF_REFLECTION_CHAT_INSTRUCTION,
            self_reflection_completion_instruction=PY_SELF_REFLECTION_COMPLETION_INSTRUCTION,
            add_code_block=lambda x: add_code_block(x, "python"),
            self_reflection_few_shot=PY_SELF_REFLECTION_FEW_SHOT
        )

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def self_reflection_parametric(self, func: str, feedback: str,parametric_insights: str,  model: ModelBase) -> str:
        return generic_generate_self_reflection_parametric(
            func=func,
            feedback=feedback,
            model=model,
            parametric_insights=parametric_insights,
            self_reflection_chat_instruction=PY_SELF_REFLECTION_PARAMETRIC_CHAT_INSTRUCTION,
            self_reflection_completion_instruction=PY_SELF_REFLECTION_COMPLETION_INSTRUCTION,
            add_code_block=lambda x: add_code_block(x, "python"),
            self_reflection_few_shot=PY_SELF_REFLEXION_PARAMETRIC_FEW_SHOT
        )
        
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def self_reflection_diverse(self, func: str, feedback: str, model: ModelBase, diverse_reflections: list) -> str:
        return generate_self_reflection_diverse(
                func=func,
                feedback=feedback,
                model=model,
                self_reflection_chat_instruction=PY_DIVERSE_REFLECTION_CHAT_INSTRUCTION,
                add_code_block=lambda x: add_code_block(x, "python"),
                self_reflection_few_shot=PY_SELF_REFLECTION_FEW_SHOT,
                previous_reflections=diverse_reflections
        )

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def self_reflection_diverse_oneshot(self, func: str, feedback: str, model: ModelBase, diverse_reflections: list) -> str:
        return generate_self_reflection_diverse_oneshot(
                func=func,
                feedback=feedback,
                model=model,
                self_reflection_chat_instruction=PY_DIVERSE_REFLECTION_ONESHOT_CHAT_INSTRUCTION,
                add_code_block=lambda x: add_code_block(x, "python"),
                self_reflection_few_shot=PY_DIVERSE_REFLECTION_FEW_SHOT,
                previous_reflections=diverse_reflections
        )
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def self_reflection_diverse_oneshot_parametric(self, func: str, feedback: str, model: ModelBase, diverse_reflections: list, mistake_insights: str) -> str:
        return generate_self_reflection_diverse_oneshot_parametric(
                func=func,
                feedback=feedback,
                model=model,
                self_reflection_chat_instruction=PY_DIVERSE_REFLECTION_PARAMETRIC_ONESHOT_CHAT_INSTRUCTION,
                add_code_block=lambda x: add_code_block(x, "python"),
                self_reflection_few_shot=PY_DIVERSE_REFLECTION_PARAMETRIC_FEW_SHOT,
                previous_reflections=diverse_reflections,
                mistake_insights=mistake_insights
        )

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def func_impl(
        self,
        func_sig: str,
        model: ModelBase,
        strategy: str,
        prev_func_impl: Optional[str] = None,
        feedback: Optional[str] = None,
        self_reflection: Optional[str] = None,
        num_comps: int = 1,
        temperature: float = 0.1,
        ref_chat_instruction=None,
        mistake_insights = None
    ) -> Union[str, List[str]]:
        
        if ref_chat_instruction == 'dot':
            ref_chat_instruction = PY_DoT_CHAT_INSTRUCTION if mistake_insights is None else PY_DoT_CHAT_INSTRUCTION_WITH_PITFALLS
        else:
            reflexion_chat_instruction = PY_REFLEXION_CHAT_INSTRUCTION
        
        return generic_generate_func_impl(
            func_sig=func_sig,
            model=model,
            strategy=strategy,
            prev_func_impl=prev_func_impl,
            feedback=feedback,
            self_reflection=self_reflection,
            num_comps=num_comps,
            temperature=temperature,
            reflexion_chat_instruction=ref_chat_instruction,
            reflexion_few_shot=PY_REFLEXION_FEW_SHOT_ADD if mistake_insights is None else PY_SELF_REFLEXION_PARAMETRIC_FEWSHOT_FUNC_IMPL,
            reflexion_completion_instruction=PY_REFLEXION_COMPLETION_INSTRUCTION,
            simple_chat_instruction=PY_SIMPLE_CHAT_INSTRUCTION,
            simple_completion_instruction=PY_SIMPLE_COMPLETION_INSTRUCTION,
            code_block_instruction=USE_PYTHON_CODEBLOCK_INSTRUCTION,
            parse_code_block=lambda x: parse_code_block(x, "python"),
            add_code_block=lambda x: add_code_block(x, "python"),
            mistake_insights = mistake_insights
        )


    def internal_tests(self, func_sig: str, model: ModelBase, max_num_tests: int = 5) -> List[str]:
        def parse_tests(tests: str) -> List[str]:
            return [test.strip() for test in tests.splitlines() if "assert" in test]
        """
        Generates tests for a function.
        """
        return generic_generate_internal_tests(
            func_sig=func_sig,
            model=model,
            max_num_tests=max_num_tests,
            
            test_generation_few_shot=PY_TEST_GENERATION_FEW_SHOT,
            test_generation_chat_instruction=PY_TEST_GENERATION_CHAT_INSTRUCTION,
            # extract tests cases from doc strings and synthesize additional
            # test_generation_few_shot=PY_EXTRACT_TEST_CASES_INSTRUCTION,
            # test_generation_chat_instruction=PY_TEST_EXTRACTION_CHAT_INSTRUCTION,
            
            test_generation_completion_instruction=PY_TEST_GENERATION_COMPLETION_INSTRUCTION,
            parse_tests=parse_tests,
            is_syntax_valid=py_is_syntax_valid,
        )


DUMMY_FUNC_SIG = "def func():"
DUMMY_FUNC_CALL = "func()"


def handle_first_line_indent(func_body: str) -> str:
    if func_body.startswith("    "):
        return func_body
    split = func_body.splitlines()
    return f"    {split[0]}\n" + "\n".join(split[1:])

def handle_entire_body_indent(func_body: str) -> str:
    split = func_body.splitlines()
    res = "\n".join(["    " + line for line in split])
    return res

def fix_turbo_response(func_body: str) -> str:
    return fix_markdown(remove_unindented_signatures(func_body))

def fix_markdown(func_body: str) -> str:
    return re.sub("`{3}", "", func_body)

def remove_unindented_signatures(code: str) -> str:
    regex = r"^def\s+\w+\s*\("

    before_signature = []
    after_signature = []
    signature_found = False

    for line in code.split("\n"):
        if re.match(regex, line):
            signature_found = True
            continue

        if signature_found:
            after_signature.append(line)
        else:
            if not line.startswith("    ") and line.strip():
                line = "    " + line
            before_signature.append(line)

    return "\n".join(before_signature + after_signature)


def py_fix_indentation(func_body: str) -> str:
    func_body = fix_turbo_response(func_body)
    """
    3 cases:
        1. good syntax
        2. first line not good
        3. entire body not good
    """
    def parse_indent_rec(f_body: str, cur_state: int) -> str:
        f_body = fix_markdown(f_body)
        if cur_state > 1:
            return f_body
        code = f'{DUMMY_FUNC_SIG}\n{f_body}\n{DUMMY_FUNC_CALL}'
        try:
            exec(code)
            return f_body
        except (IndentationError, SyntaxError):
            p_func = handle_first_line_indent if cur_state == 0 else handle_entire_body_indent
            return parse_indent_rec(p_func(func_body), cur_state + 1)
        except Exception:
            return f_body
    return parse_indent_rec(func_body, 0)


def py_is_syntax_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception:
        return False

