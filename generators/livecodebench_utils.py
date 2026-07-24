
import json
import base64
import zlib
import pickle
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class TestCase:
    """
    A single I/O test case for a LiveCodeBench problem.

    Attributes
    ----------
    input : str
        Raw input string for the program (typically stdin content).
    output : str
        Expected output string (typically stdout content).
    testtype : str
        Type of test interface. In LiveCodeBench code generation it is usually "stdin".
    """
    input: str
    output: str
    testtype: str


def parse_public_tests(sample: Dict[str, Any]) -> List[TestCase]:
    """
    Parse the public_test_cases field of a LiveCodeBench sample.

    Parameters
    ----------
    sample : dict
        A single row from the LiveCodeBench dataset, e.g. ds['test'][i].

    Returns
    -------
    List[TestCase]
        List of parsed public test cases, each with input/output strings.
    """
    raw = sample["public_test_cases"]
    data = json.loads(raw)
    return [TestCase(**tc) for tc in data]


def _decode_private_tests_raw(raw: str) -> list[Dict[str, Any]]:
    """
    Decode the private_test_cases payload into a Python list[dict].

    Parameters
    ----------
    raw : str
        The raw string from sample['private_test_cases'].

    Returns
    -------
    list[dict]
        A list of dictionaries with keys 'input', 'output', 'testtype'.

    Notes
    -----
    Matches the official LiveCodeBench loader logic:

    1. First try plain JSON.
    2. If that fails, assume:
       base64-encoded → zlib-compressed → pickle-serialized → JSON string.
    """
    # 1) Try plain JSON (newer entries / some configs)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2) Fallback: base64 + zlib + pickle → JSON string → list[dict]
    try:
        compressed_bytes = base64.b64decode(raw.encode("utf-8"))
        pickled_obj = zlib.decompress(compressed_bytes)
        json_str = pickle.loads(pickled_obj)          # gives a JSON string
        data = json.loads(json_str)                   # convert JSON string to Python object
        return data
    except Exception as e:
        raise ValueError(f"Could not decode private_test_cases: {e}")


def parse_private_tests(sample: Dict[str, Any]) -> List[TestCase]:
    """
    Parse the private_test_cases field of a LiveCodeBench sample.

    Parameters
    ----------
    sample : dict
        A single row from the LiveCodeBench dataset, e.g. ds['test'][i].

    Returns
    -------
    List[TestCase]
        List of parsed private test cases.

    Notes
    -----
    - Private tests are only for evaluation, do not include them in the prompt.
    - Handles both plain-JSON and compressed encodings used by LiveCodeBench.
    """
    raw = sample["private_test_cases"]
    data = _decode_private_tests_raw(raw)
    return [TestCase(**tc) for tc in data]



def format_livecodebench_prompt(
    sample: Dict[str, Any],
    language: str = "python",
    include_public_tests: bool = True,
    max_public_tests: Optional[int] = None,
) -> str:
    """
    Construct a code-generation prompt for a LiveCodeBench problem.

    Parameters
    ----------
    sample : dict
        A single row from the LiveCodeBench dataset, e.g. ds['test'][i].
        Expected keys: 'question_title', 'question_content',
        'platform', 'question_id', 'difficulty', 'public_test_cases', etc.
    language : str, optional
        Target programming language (for instructions in the prompt), by default "python".
    include_public_tests : bool, optional
        Whether to append public (sample) tests as examples in the prompt, by default True.
    max_public_tests : int or None, optional
        If not None, limit the number of public tests included.

    Returns
    -------
    str
        A text prompt suitable for feeding into a code LLM.

    Behavior
    --------
    - Uses problem title and statement from the dataset.
    - Instructs the model to write a full program that reads from stdin and writes to stdout.
    - Optionally appends the public tests as example input/output pairs.
    """
    title = sample["question_title"].strip()
    description = sample["question_content"].strip()
    difficulty = sample.get("difficulty", "").strip()
    platform = sample.get("platform", "").strip()
    qid = sample.get("question_id", "").strip()

    lines = []

    # Problem header and metadata (useful but not strictly necessary)
    header = f"{title}"
    meta_bits = []
    if platform:
        meta_bits.append(f"Platform: {platform}")
    if qid:
        meta_bits.append(f"Problem ID: {qid}")
    if difficulty:
        meta_bits.append(f"Difficulty: {difficulty}")
    if meta_bits:
        header += " (" + ", ".join(meta_bits) + ")"

    lines.append(header)
    lines.append("")
    lines.append(description)
    lines.append("")

    # Main instruction for the model.
    lines.append(
        f"Write a complete {language} program that solves this problem."
        " The program must:"
    )
    lines.append(
        "- Read input from standard input (stdin) using the format described above."
    )
    lines.append(
        "- Write the required output to standard output (stdout) without extra text,"
        " debug prints, or explanations."
    )
    lines.append(
        "- Be efficient enough to handle the input constraints."
    )

    # Optionally include public test cases as examples.
    if include_public_tests:
        public_tests = parse_public_tests(sample)
        if max_public_tests is not None:
           public_tests = public_tests[:max_public_tests]

        if public_tests:
            lines.append("")
            lines.append("Here are example input/output pairs:")
            for i, tc in enumerate(public_tests, start=1):
                lines.append(f"Example {i}:")
                lines.append("Input:")
                lines.append(tc.input.rstrip("\n"))
                lines.append("Output:")
                lines.append(tc.output.rstrip("\n"))
                lines.append("")

    # Small reminder about the entry point (helps some models behave).
    if language.lower().startswith("python"):
        lines.append(
            'Implement your solution with a `main()` or under `if __name__ == "__main__":`.'
        )
    else:
        lines.append("Implement the standard `main` entry point for this language.")

    prompt = "\n".join(lines)
    return prompt



# example usage
# Example: use the first sample in the test split
# from datasets import load_dataset
# ds = load_dataset("livecodebench/code_generation_lite", version_tag="release_v6")
# sample = ds["test"][1]  # e.g. your 'B. Good Kid' example

# # Stage 1: build prompt
# prompt = format_livecodebench_prompt(sample, language="python", max_public_tests=2)
# print("=== Prompt ===")
# print(prompt)

# # Stage 2: parse public tests
# public_tests = parse_public_tests(sample)
# print(f"Parsed {len(public_tests)} public tests.")

# # Stage 3: parse private tests (for evaluation only)
# private_tests = parse_private_tests(sample)
# print(f"Parsed {len(private_tests)} private tests.")

