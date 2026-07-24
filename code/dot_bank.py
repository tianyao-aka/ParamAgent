import random
import pickle as pkl
import sys
from copy import deepcopy
from pprint import pprint
from tqdm import tqdm
from termcolor import colored

from utils import enumerate_resume_dotbank, \
                  make_printv, \
                  write_jsonl, \
                  resume_success_count, read_jsonl
from executors import executor_factory
from generators import generator_factory, model_factory
from typing import List

from gpt_usage import gpt_usage
import subprocess
import tempfile
import os

# memory bank imports
from scipy.spatial import distance
from memory_utils import get_cohere_embedding, \
                         get_openai_embedding, \
                         get_top_k_closest, \
                         get_random_k_indices
import generators.py_generate as py_generate
import generators.generator_utils as gen_utils
from generators.parse import parse_code_block, add_code_block
import os
import re
import ast


def _strip_tests_and_main(code: str) -> str:
    if not isinstance(code, str):
        return code
    # remove `import unittest` lines
    code = re.sub(r'^\s*import\s+unittest\s*\n', '', code, flags=re.MULTILINE)
    # remove unittest.TestCase classes (greedy until next top-level)
    code = re.sub(
    r'^\s*class\s+\w+\(unittest\.TestCase\):[\s\S]*?(?=^\S|\Z)',
    '',
    code,
     flags=re.MULTILINE)
    # remove `unittest.main()` lines
    code = re.sub(
    r'^\s*unittest\.main\s*\(\s*\)\s*$',
    '',
    code,
     flags=re.MULTILINE)
    # remove `if __name__ == "__main__":` blocks to EOF
    code = re.sub(
    r'^\s*if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:\s*[\s\S]*\Z',
    '',
    code,
     flags=re.MULTILINE)
    return code


def _extract_function_only(code: str, entry_point: str) -> str:
    """
    Keep only the target function `def <entry_point>(...): ...`.
    Uses AST when possible, falls back to regex.
    """
    if not isinstance(code, str):
        return code
    try:
        tree = ast.parse(code)
        lines = code.splitlines()
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == entry_point:
                start = node.lineno - 1
                end = getattr(node, 'end_lineno', None)
                if end is None:
                    # indentation-based fallback
                    end = _find_end_by_indent(lines, start)
                return '\n'.join(lines[start:end]).rstrip() + '\n'
    except Exception:
        pass
    # regex fallback
    pattern = r'(?:^|\n)def\s+' + re.escape(entry_point) + \
                 r'\s*\(.*?\):(?:\n(?:[ \t].*|\n)*)'
    m = re.search(pattern, code, flags=re.DOTALL)
    return m.group(0).strip() + '\n' if m else code


def _find_end_by_indent(lines, start):
    base = len(re.match(r'[ \t]*', lines[start]).group(0))
    i = start + 1
    while i < len(lines):
        line = lines[i]
        if line.strip() and not line.lstrip().startswith('#'):
            indent = len(re.match(r'[ \t]*', line).group(0))
            if indent <= base:
                break
        i += 1
    return i


def _postprocess_impl(code: str, entry_point: str) -> str:
    """Strip tests/main and keep only the target function."""
    code = _strip_tests_and_main(code)
    code = _extract_function_only(code, entry_point)
    return code


def _evaluate_with_feedback_livecodebench(
    exe, identifier, program, test_cases, timeout=20):
    """
    Evaluate LiveCodeBench program and return detailed feedback.

    Args:
        exe: Executor (not used, kept for consistency)
        identifier: Problem identifier for logging
        program: Complete Python program code
        test_cases: List of TestCase objects from parse_public_tests/parse_private_tests
        timeout: Execution timeout per test (seconds)

    Returns:
        (bool, str): (is_passing, feedback_string)
    """
    feedback_lines = []
    all_passing = True

    for i, test in enumerate(test_cases):
        temp_file = None
        try:
            # Write and execute program
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                f.write(program)
                temp_file = f.name

            result = subprocess.run(
                ['python', temp_file],
                input=test.input,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            # Clean up temp file
            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)
                temp_file = None

            # Check result
            if result.returncode != 0:
                feedback_lines.append(
                    f"Test {i+1} FAILED - Runtime error:\n{result.stderr[:200]}")
                all_passing = False
            else:
                actual = result.stdout.strip()
                expected = test.output.strip()

                if actual != expected:
                    feedback_lines.append(
                        f"Test {i+1} FAILED\n"
                        f"  Input: {test.input[:50].strip()}{'...' if len(test.input) > 50 else ''}\n"
                        f"  Expected: {expected[:50]}{'...' if len(expected) > 50 else ''}\n"
                        f"  Got: {actual[:50]}{'...' if len(actual) > 50 else ''}"
                    )
                    all_passing = False
                else:
                    feedback_lines.append(f"Test {i+1} passed")

        except subprocess.TimeoutExpired:
            feedback_lines.append(f"Test {i+1} TIMEOUT (>{timeout}s)")
            all_passing = False
            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)
        except Exception as e:
            feedback_lines.append(f"Test {i+1} ERROR: {str(e)[:100]}")
            all_passing = False
            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)

    feedback = "\n".join(feedback_lines)
    return all_passing, feedback


def run_dot_bank(
    dataset: List[dict],
    model_name: str,
    language: str,
    max_iters: int,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
    is_leetcode: bool = False,
    visible_tests: any = None,
    dataset_type: str = "humaneval",
    **kargs
) -> None:
    exe = executor_factory(language, is_leet=is_leetcode)
    gen = generator_factory(language)
    model = model_factory(model_name)

    print_v = make_printv(verbose)

    num_items = len(dataset)
    num_success = resume_success_count(dataset)

    print("Running DoT-Bank")

    # init memory bank related file paths
    root_path = '/'.join(log_path.split('/')[:-1])
    mem_bank_file_path = root_path + '/mem_bank.pkl'
    failed_probs_path = root_path + '/failed_probs.pkl'

    # check if memory-bank already exists
    if os.path.exists(mem_bank_file_path):
        with open(mem_bank_file_path, 'rb') as f:
            memory_bank = pkl.load(f)
    else:
        # initialize memory bank
        memory_bank = {
            "positive_trajectories": [],
            "negative_trajectories": [],
        }

    if os.path.exists(failed_probs_path):
        with open(failed_probs_path, 'rb') as f:
            failed_problems = pkl.load(f)
    else:
        # store all problems that failed visible/synthetic tests in the first
        # pass
        failed_problems = []

    # Determine primary key based on dataset type
    if dataset_type == 'livecodebench':
        primary_key = "question_id"
    elif "task_id" in dataset[0].keys():
        primary_key = "task_id"
    else:
        primary_key = "name"

    # Import LiveCodeBench utilities if needed
    if dataset_type == 'livecodebench':
        from generators.livecodebench_utils import (
            format_livecodebench_prompt,
            parse_public_tests,
            parse_private_tests
        )

    # Snapshot the entire first-stage log to a new JSONL
    root_path = '/'.join(log_path.split('/')[:-1])
    first_stage_json = root_path + '/first_stage_log.jsonl'
    second_stage_json = root_path + '/second_stage_log.jsonl'
    if os.path.exists(second_stage_json):
        skip_first = True
    else:
        skip_first = False
    # First Pass
    for i, item in enumerate_resume_dotbank(dataset, log_path):
        if skip_first: break

        # Normalize field access based on dataset type
        if dataset_type == 'livecodebench':
            prompt = format_livecodebench_prompt(
    item, language="python", include_public_tests=True)
            identifier = item.get(
    "question_title", item.get(
        "question_id", f"problem_{i}"))
            public_tests = parse_public_tests(item)
            private_tests = parse_private_tests(item)
        else:
            prompt = item["prompt"]
            identifier = item["entry_point"]
            test_code = item["test"]

        cur_pass = 0
        is_solved = False
        diverse_reflections = []
        implementations = []
        test_feedback = []
        all_levels_reflections_scores = []
        all_levels_implementations = []
        cur_func_impl = None

        cur_prob_passed = False

        try:

            while cur_pass < pass_at_k and not is_solved:
                if dataset_type == 'livecodebench':
                    tests_i = public_tests
                elif is_leetcode:
                    tests_i = item['visible_tests']
                else:

                    if visible_tests:
                        # Use visible test cases
                        print("using visible test cases")
                        tests_i = visible_tests[identifier]['given_tests']

                    else:
                        print("generating synthetic test cases")
                        tests_i = gen.internal_tests(prompt, model, 1)

                fail_cnt = 0
                while cur_func_impl is None:
                    cur_func_impl = gen.func_impl(
    prompt, model, "simple", temperature=1.)
                    fail_cnt += 1
                    if fail_cnt > 1:
                        break

                implementations.append(cur_func_impl)
                assert isinstance(cur_func_impl, str)

                # Only postprocess for HumanEval (extract function only)
                if dataset_type != 'livecodebench':
                    entry_point = item.get('entry_point', None)
                    if entry_point:
                        cur_func_impl = _postprocess_impl(
                            cur_func_impl, entry_point)

                # Execute intermediate tests and get feedback
                if dataset_type == 'livecodebench':
                    is_passing, feedback = _evaluate_with_feedback_livecodebench(
                        exe, identifier, cur_func_impl, public_tests, timeout=20
                    )
                    test_feedback.append(feedback)
                else:
                    is_passing, feedback, _ = exe.execute(
                        cur_func_impl, tests_i)
                    test_feedback.append(feedback)

                print(gpt_usage(backend=model_name))

                # if solved, exit early
                if is_passing:

                    # populate memory bank if first attempt passed all visible
                    # tests
                    trajectory = {
                                    "task_id": item[primary_key],
                                    "prompt": prompt,
                                    "gen_solution": cur_func_impl,
                                    "prompt_embedding": get_openai_embedding([prompt]),
                                }

                    # update memory bank
                    cur_prob_passed = True
                    memory_bank['positive_trajectories'].append(trajectory)

                    # evaluate on hidden test cases
                    if dataset_type == 'livecodebench':
                        is_passing = exe.evaluate_livecodebench(
    identifier, cur_func_impl, private_tests, timeout=20)
                    else:
                        is_passing = exe.evaluate(
    identifier, cur_func_impl, test_code, timeout=20)

                    is_solved = is_passing
                    num_success += int(is_passing)
                    print(is_solved, num_success)
                    break

                # conditional sampling on prior reflections to promote
                # diversity
                cur_iter = 1
                cur_feedback = feedback
                while cur_iter < max_iters:

                    # iterative sampling
                    # # get self-reflection-diverse
                    # reflection = gen.self_reflection_diverse(
                    #     cur_func_impl, cur_feedback, model, diverse_reflections)
                    # diverse_reflections += [reflection]

                    # one-shot sampling
                    # get multiple diverse reflections
                    div_reflections = gen.self_reflection_diverse_oneshot(
                        cur_func_impl, cur_feedback, model, diverse_reflections).split("\n\n")

                    # filter out reflections if they are less than few
                    # characters
                    div_reflections = [
    ref for ref in div_reflections if len(ref) > 10]

                    # revisit later
                    diverse_reflections += div_reflections
                    cur_func_impl_copy = deepcopy(cur_func_impl)

                    temp_implementations = []
                    reflections_scores = []
                    div_reflections_feedbacks = []

                    ref_id = 0
                    pbar = tqdm(total=len(div_reflections))
                    while ref_id < min(len(div_reflections), 2):

                        # re-init executor
                        del exe
                        exe = executor_factory(language, is_leet=is_leetcode)

                        reflection = div_reflections[ref_id]
                        print(f"Attempting reflection-{ref_id}:")
                        pprint(reflection)
                        print()

                        # apply self-reflection in the next attempt
                        new_func_impl = None
                        fail_cnt = 0
                        while new_func_impl is None:
                            new_func_impl = gen.func_impl(
                                func_sig=prompt,
                                model=model,
                                strategy="reflexion",
                                prev_func_impl=cur_func_impl_copy,
                                feedback=cur_feedback,
                                self_reflection=reflection,
                                temperature=1.0,
                                ref_chat_instruction='dot'
                            )
                            fail_cnt += 1
                            if fail_cnt > 1: break
                        cur_func_impl = new_func_impl

                        try:
                            assert isinstance(cur_func_impl, str)
                        except:
                            print("skipping func impl.")
                            ref_id += 1
                            continue

                        # Will be used later to sample a probable solution
                        temp_implementations.append(cur_func_impl)

                        # check if all internal unit tests pass
                        if dataset_type == 'livecodebench':
                            is_passing, cur_feedback = _evaluate_with_feedback_livecodebench(
                                exe, identifier, cur_func_impl, public_tests, timeout=20
                            )
                            test_feedback.append(cur_feedback)
                            div_reflections_feedbacks.append(cur_feedback)

                            # Score by counting "passed" occurrences
                            reflections_scores.append(
                                cur_feedback.count("passed") + 1e-8)
                        else:
                            is_passing, cur_feedback, _ = exe.execute(
                                cur_func_impl, tests_i)
                            test_feedback.append(cur_feedback)
                            div_reflections_feedbacks.append(cur_feedback)

                            # measures total number of failed unit tests
                            reflections_scores.append(
    (len(tests_i) - cur_feedback.split("Tests failed:")[1].count('assert')) + 1e-8)

                        # increment ref-id counter
                        ref_id += 1
                        pbar.update(1)

                        # if solved, check if it passes the real tests, exit
                        # early
                        if is_passing or cur_iter == max_iters - 1:
                            # setting based on visible/synthetic tests
                            cur_prob_passed = True

                            if dataset_type == 'livecodebench':
                                is_passing = exe.evaluate_livecodebench(
    identifier, cur_func_impl, private_tests, timeout=20)
                            else:
                                is_passing = exe.evaluate(
    identifier, cur_func_impl, test_code, timeout=10)

                            if is_passing:
                                item["solution"] = cur_func_impl
                                is_solved = True
                                num_success += int(is_passing)

                            break

                    pbar.close()

                    # log reflection scores and given level implementations
                    all_levels_reflections_scores.append(reflections_scores)
                    all_levels_implementations.append(temp_implementations)

                    # sample likely implementation
                    print(reflections_scores)
                    sampled_impl_idx = random.choices(
    range(
        len(temp_implementations)),
        weights=reflections_scores,
         k=1)[0]
                    cur_func_impl = temp_implementations[sampled_impl_idx]

                    # set cur_feedback to the corresponding sampled
                    # div-reflection
                    cur_feedback = div_reflections_feedbacks[sampled_impl_idx]

                    # populate memory bank
                    visible_tests_status, _, _ = exe.execute(
                        cur_func_impl, tests_i)

                    if cur_iter == max_iters - 1 or visible_tests_status:
                        trajectory = {
                                        "task_id": item[primary_key],
                                        "prompt": prompt,
                                        "gen_solution": cur_func_impl,
                                        "reflection": cur_feedback,
                                        "test_feedback": test_feedback[sampled_impl_idx],
                                        "prev_implementation": cur_func_impl_copy,
                                        "prompt_embedding": get_openai_embedding([prompt]),
                                        "refection_embedding": get_openai_embedding([cur_feedback]),
                                    }
                        if visible_tests_status:
                            cur_prob_passed = True
                            memory_bank['positive_trajectories'].append(
                                trajectory)
                        else:
                            memory_bank['negative_trajectories'].append(
                                trajectory)

                    if is_solved:
                        break

                    cur_iter += 1
                cur_pass += 1

        except Exception as e:
            print(
                f"Exception occurred for problem {i}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            # Set is_solved to False when exception occurs
            is_solved = False
            llm_cost = gpt_usage(backend=model_name)
            print(llm_cost)
            item["is_solved"] = is_solved
            item["diverse_reflections"] = diverse_reflections
            item["implementations"] = implementations
            item["test_feedback"] = test_feedback
            item["solution"] = cur_func_impl
            item['all_levels_reflections_scores'] = all_levels_reflections_scores
            item['all_levels_implementations'] = all_levels_implementations
            item['cost'] = llm_cost['cost']
            item['completion_tokens'] = llm_cost['completion_tokens']
            item['prompt_tokens'] = llm_cost['prompt_tokens']
            write_jsonl(log_path, [item], append=True)
            continue

        llm_cost = gpt_usage(backend=model_name)
        print(llm_cost)

        if cur_prob_passed:
            item["is_solved"] = is_solved
            item["diverse_reflections"] = diverse_reflections
            item["implementations"] = implementations
            item["test_feedback"] = test_feedback
            item["solution"] = cur_func_impl
            item['all_levels_reflections_scores'] = all_levels_reflections_scores
            item['all_levels_implementations'] = all_levels_implementations
            item['cost'] = llm_cost['cost']
            item['completion_tokens'] = llm_cost['completion_tokens']
            item['prompt_tokens'] = llm_cost['prompt_tokens']
            write_jsonl(log_path, [item], append=True)

            print_v(
                f'completed {i+1}/{num_items}: acc = {round(num_success/(i+1), 2)}')

        else:
            failed_problems.append(item)

        # write mem-bank to file
        with open(mem_bank_file_path, 'wb') as f:
            pkl.dump(memory_bank, f)

        # update failed_probs.pkl
        with open(failed_probs_path, 'wb') as f:
            pkl.dump(failed_problems, f)

    print("finished first pass!")

    memory_bank = pkl.load(open(mem_bank_file_path, 'rb'))
    logs = read_jsonl(log_path)

    # Snapshot the entire first-stage log to a new JSONL
    root_path = '/'.join(log_path.split('/')[:-1])
    first_stage_json = root_path + '/first_stage_log.jsonl'
    second_stage_json = root_path + '/second_stage_log.jsonl'
    write_jsonl(first_stage_json, logs, append=False, key=None, stage2=False)
    print(
        f"[info] First-stage log saved to: {first_stage_json} (n={len(logs)})")

    print(logs[0].keys())
    # Filter out items that have stage2=True (keep only first-pass failures)
    failed_problems = [rec for rec in logs
                    if not rec.get("is_solved", False)
                    and not rec.get("stage2", False)]
    print(f"number of failed problems: {len(failed_problems)}")

    # Load the complete logs and maintain a copy for updates
    logs_copy = read_jsonl(log_path)

    # reset num_items and num_success for 2nd pass
    num_items = len(failed_problems)
    num_success = 0

    # # Second pass
    for i, item in enumerate(failed_problems):
        if 'stage2' in item:
            print(f"skip {i}; stage2 exists")
            continue

        # Normalize field access based on dataset type (SAME as Stage 1)
        if dataset_type == 'livecodebench':
            prompt = format_livecodebench_prompt(
    item, language="python", include_public_tests=True)
            identifier = item.get(
    "question_title", item.get(
        "question_id", f"problem_{i}"))
            public_tests = parse_public_tests(item)
            private_tests = parse_private_tests(item)
        else:
            prompt = item["prompt"]
            identifier = item["entry_point"]
            test_code = item["test"]

        try:

            cur_pass = 0
            is_solved = False
            diverse_reflections = []
            implementations = []
            test_feedback = []
            all_levels_reflections_scores = []
            all_levels_implementations = []
            cur_func_impl = ""

            visible_tests_status = False

            while cur_pass < pass_at_k and not is_solved:
                if dataset_type == 'livecodebench':
                    tests_i = public_tests
                elif is_leetcode:
                    tests_i = item['visible_tests']
                else:

                    if visible_tests:
                        # Use visible test cases
                        print("using visible test cases")
                        tests_i = visible_tests[identifier]['given_tests']

                    else:
                        print("generating synthetic test cases")
                        tests_i = gen.internal_tests(prompt, model, 1)

                # inject similar problems trajectory into context
                curr_emb = get_openai_embedding([prompt])

                if len(memory_bank['positive_trajectories']) > 0:
                    top_k_indices, cosine_similarities = get_top_k_closest(
                        memory_bank['positive_trajectories'], curr_emb[:, None], k=1)
                    closest_match = [
    memory_bank['positive_trajectories'][i] for i in top_k_indices]
                else:
                    print(
                        f"Warning: No positive trajectories available for Stage 2 problem {i}, using no examples")
                    closest_match = []

                PY_SIMPLE_CHAT_INSTRUCTION = (
                    "You are an AI that only responds with python code, NOT ENGLISH.\n"
                    "You will be given a function signature and its docstring by the user.\n"
                    "Write your full implementation (restate the function signature).\n"
                    + (f"Here are {len(closest_match)} problems and their solutions.\n\n"
                       + ''.join(
                           f"[Problem {i+1}]\n"
                           "```python\n"
                           f"{example['prompt']}\n"
                           "```\n"
                           f"[Solution {i+1}]\n"
                           "```python\n"
                           f"{example['gen_solution']}\n"
                           "```\n\n"
                           for i, example in enumerate(closest_match)
                       ) if len(closest_match) > 0 else "")
                )

                # first attempt
                cur_func_impl = gen_utils.generic_generate_func_impl(
                                            func_sig=prompt,
                                            model=model,
                                            strategy='simple',
                                            num_comps=1,
                                            temperature=1.,
                                            simple_chat_instruction=PY_SIMPLE_CHAT_INSTRUCTION,
                                            simple_completion_instruction=py_generate.PY_SIMPLE_COMPLETION_INSTRUCTION,
                                            code_block_instruction=py_generate.USE_PYTHON_CODEBLOCK_INSTRUCTION,
                                            parse_code_block=lambda x: parse_code_block(
                                                x, "python"),
                                            add_code_block=lambda x: add_code_block(
                                                x, "python"),
                                            prev_func_impl=None,
                                            feedback=None,
                                            self_reflection=None,
                                            reflexion_chat_instruction=None,
                                            reflexion_few_shot=None,
                                            reflexion_completion_instruction=None
                                        )

                implementations.append(cur_func_impl)
                assert isinstance(cur_func_impl, str)

                # Only postprocess for HumanEval (extract function only)
                if dataset_type != 'livecodebench':
                    entry_point = item.get('entry_point', None)
                    if entry_point:
                        cur_func_impl = _postprocess_impl(
                            cur_func_impl, entry_point)

                # Execute intermediate tests and get feedback
                if dataset_type == 'livecodebench':
                    is_passing, feedback = _evaluate_with_feedback_livecodebench(
                        exe, identifier, cur_func_impl, public_tests, timeout=20
                    )
                    test_feedback.append(feedback)
                else:
                    is_passing, feedback, _ = exe.execute(
                        cur_func_impl, tests_i)
                    test_feedback.append(feedback)

                print(gpt_usage(backend=model_name))

                # if solved, exit early
                if is_passing:
                    visible_tests_status = True  # passed visible/Synthetic test cases

                    # evaluate on hidden test cases
                    if dataset_type == 'livecodebench':
                        is_passing = exe.evaluate_livecodebench(
    identifier, cur_func_impl, private_tests, timeout=20)
                    else:
                        is_passing = exe.evaluate(
    identifier, cur_func_impl, test_code, timeout=20)

                    is_solved = is_passing
                    num_success += int(is_passing)
                    print(is_solved, num_success)
                    break

                # conditional sampling on prior reflections to promote
                # diversity
                cur_iter = 1
                cur_feedback = feedback
                while cur_iter < max_iters:

                    # iterative sampling
                    # # get self-reflection-diverse
                    # reflection = gen.self_reflection_diverse(
                    #     cur_func_impl, cur_feedback, model, diverse_reflections)
                    # diverse_reflections += [reflection]

                    # one-shot sampling
                    # get multiple diverse reflections
                    div_reflections = gen.self_reflection_diverse_oneshot(
                        cur_func_impl, cur_feedback, model, diverse_reflections).split("\n\n")

                    # filter out reflections if they are less than few
                    # characters
                    div_reflections = [
    ref for ref in div_reflections if len(ref) > 10]

                    # revisit later
                    diverse_reflections += div_reflections
                    cur_func_impl_copy = deepcopy(cur_func_impl)

                    temp_implementations = []
                    reflections_scores = []
                    div_reflections_feedbacks = []

                    ref_id = 0
                    pbar = tqdm(total=len(div_reflections))
                    while ref_id < min(len(div_reflections), 2):
                        try:
                            # re-init executor
                            del exe
                            exe = executor_factory(
                                language, is_leet=is_leetcode)

                            reflection = div_reflections[ref_id]
                            print(f"Attempting reflection-{ref_id}:")
                            pprint(reflection)
                            print()

                            # inject similar problems trajectory into context
                            # based on similary in reflection
                            curr_emb = get_openai_embedding([reflection])
                            # Only use trajectories that have ALL required
                            # fields for reflection-based examples
                            filtered_trajectories = [traj for traj in memory_bank['positive_trajectories']
                                                     if all(k in traj for k in ["refection_embedding", "prev_implementation",
                                                                                 "test_feedback", "reflection"])]

                            if len(filtered_trajectories):
                                top_k_indices, cosine_similarities = get_top_k_closest(filtered_trajectories,
                                                                                        curr_emb[:,
                                                                                            None],
                                                                                        k=1,
                                                                                        similarity_axis="refection_embedding")
                                closest_match = filtered_trajectories[top_k_indices[0]]
                            else:
                                # Fall back to all positive_trajectories with prompt_embedding
                                if len(memory_bank['positive_trajectories']) > 0:
                                    top_k_indices, cosine_similarities = get_top_k_closest(memory_bank['positive_trajectories'],
                                                                                            curr_emb[:,
                                                                                                None],
                                                                                            k=1,
                                                                                            similarity_axis="prompt_embedding")
                                    if len(top_k_indices) > 0:
                                        closest_match = memory_bank['positive_trajectories'][top_k_indices[0]]
                                        # Validate the fallback trajectory has required fields for reflection-based examples
                                        if not all(k in closest_match for k in ["prev_implementation", "test_feedback", "reflection"]):
                                            print(
                                                f"Warning: Fallback trajectory missing required fields for problem {i}, skipping reflection {ref_id}")
                                            continue
                                    else:
                                        print(
                                            f"Warning: No valid trajectories for problem {i}, skipping reflection {ref_id}")
                                        continue
                                else:
                                    print(f"Warning: No positive trajectories available for problem {i}, skipping reflection {ref_id}")
                                    continue
                            
                            # apply self-reflection in the next attempt
                            PY_FEW_SHOT = f'''Example 1:
    [previous impl]:
    ```python
    {closest_match['prev_implementation']}
    ```

    [unit test results from previous impl]:
    {closest_match["test_feedback"][0]}

    [reflection on previous impl]:
    {closest_match['reflection']}

    [improved impl]:
    ```python
    {closest_match['gen_solution']}
    ```
    '''
                            cur_func_impl = gen_utils.generic_generate_func_impl(
                                                    func_sig=prompt,
                                                    model=model,
                                                    strategy='reflexion',
                                                    num_comps=1,
                                                    temperature=1.,
                                                    simple_chat_instruction=PY_SIMPLE_CHAT_INSTRUCTION,
                                                    simple_completion_instruction=py_generate.PY_SIMPLE_COMPLETION_INSTRUCTION,
                                                    code_block_instruction=py_generate.USE_PYTHON_CODEBLOCK_INSTRUCTION,
                                                    parse_code_block=lambda x: parse_code_block(x, "python"),
                                                    add_code_block=lambda x: add_code_block(x, "python"),
                                                    prev_func_impl=cur_func_impl_copy,
                                                    feedback=cur_feedback,
                                                    self_reflection=reflection,
                                                    reflexion_chat_instruction=py_generate.PY_REFLEXION_CHAT_INSTRUCTION,
                                                    reflexion_few_shot=PY_FEW_SHOT,
                                                    reflexion_completion_instruction=py_generate.PY_REFLEXION_COMPLETION_INSTRUCTION
                                                    )

                            # Only postprocess for HumanEval (extract function only)
                            if dataset_type != 'livecodebench':
                                entry_point = item.get('entry_point', None)
                                if entry_point:
                                    cur_func_impl = _postprocess_impl(cur_func_impl, entry_point)

                            # Will be used later to sample a probable solution
                            temp_implementations.append(cur_func_impl)

                            # check if all internal unit tests pass
                            if dataset_type == 'livecodebench':
                                is_passing, cur_feedback = _evaluate_with_feedback_livecodebench(
                                    exe, identifier, cur_func_impl, public_tests, timeout=20
                                )
                                test_feedback.append(cur_feedback)
                                div_reflections_feedbacks.append(cur_feedback)

                                # Score by counting "passed" occurrences
                                reflections_scores.append(cur_feedback.count("passed") + 1e-8)
                            else:
                                is_passing, cur_feedback, _ = exe.execute(cur_func_impl, tests_i)
                                test_feedback.append(cur_feedback)
                                div_reflections_feedbacks.append(cur_feedback)

                                # measures total number of failed unit tests
                                reflections_scores.append((len(tests_i) - cur_feedback.split("Tests failed:")[1].count('assert')) + 1e-8)

                            # increment ref-id counter
                            ref_id += 1
                            pbar.update(1)

                            # if solved, check if it passes the real tests, exit early
                            if is_passing or cur_iter == max_iters - 1:
                                if dataset_type == 'livecodebench':
                                    is_passing = exe.evaluate_livecodebench(identifier, cur_func_impl, private_tests, timeout=20)
                                else:
                                    is_passing = exe.evaluate(identifier, cur_func_impl, test_code, timeout=10)

                                if is_passing:
                                    item["solution"] = cur_func_impl
                                    is_solved = True
                                    num_success += 1
                                break
                        except Exception as e:
                            print(f"Error in reflection {ref_id} for problem {i}: {e}")
                            import traceback
                            traceback.print_exc()
                            # Skip this reflection and try the next one
                            ref_id += 1
                            pbar.update(1)
                            continue

                    pbar.close()
                    
                    # log reflection scores and given level implementations
                    all_levels_reflections_scores.append(reflections_scores)
                    all_levels_implementations.append(temp_implementations)

                    # sample likely implementation
                    print(reflections_scores)

                    # Check if any implementations were generated
                    if len(temp_implementations) == 0:
                        print(f"Warning: No valid implementations generated for problem {i}, iter {cur_iter}. Breaking iteration loop.")
                        break  # Exit the cur_iter loop, will move to next problem

                    sampled_impl_idx = random.choices(range(len(temp_implementations)), weights=reflections_scores, k=1)[0]
                    cur_func_impl = temp_implementations[sampled_impl_idx]
                    
                    # set cur_feedback to the corresponding sampled div-reflection
                    cur_feedback = div_reflections_feedbacks[sampled_impl_idx]
                    
                    # populate memory bank
                    visible_tests_status, _, _ = exe.execute(cur_func_impl, tests_i)
                    if cur_iter == max_iters - 1:
                        trajectory = {
                                        "task_id": item[primary_key],
                                        "prompt": prompt,
                                        "gen_solution": cur_func_impl,
                                        "reflection": cur_feedback,
                                        "prompt_embedding": get_openai_embedding([prompt]),
                                        "refection_embedding": get_openai_embedding([cur_feedback]),
                                    }                           
                        if visible_tests_status:
                            memory_bank['positive_trajectories'].append(trajectory)
                        else:
                            memory_bank['negative_trajectories'].append(trajectory)
                    
                    if is_solved:
                        break
                    
                    cur_iter += 1
                cur_pass += 1
                
        except Exception as e:
            print(f"Exception in second pass example {i}: {e}")
            import traceback
            traceback.print_exc()
            # Set is_solved to False and continue to next problem
            is_solved = False
            continue
        
        llm_cost = gpt_usage(backend=model_name)
        print(llm_cost)
        
        # Find the corresponding item in logs_copy using task_id as key
        for log_item in logs_copy:
            if log_item.get(primary_key) == item.get(primary_key):
                # Update runtime and stage2 flag
                log_item['stage2'] = True
                
                # Update is_solved (may change or not)
                log_item["is_solved"] = is_solved
                
                # Accumulate the costs
                log_item["cost"] = log_item.get("cost", 0) + llm_cost["cost"]
                log_item["completion_tokens"] = log_item.get("completion_tokens", 0) + llm_cost["completion_tokens"]
                log_item["prompt_tokens"] = log_item.get("prompt_tokens", 0) + llm_cost["prompt_tokens"]
                
                # Concatenate the lists
                log_item["diverse_reflections"] = log_item.get("diverse_reflections", []) + diverse_reflections
                log_item["implementations"] = log_item.get("implementations", []) + implementations
                log_item["test_feedback"] = log_item.get("test_feedback", []) + test_feedback
                log_item["solution"] = cur_func_impl
                
                # Add new fields specific to this version
                log_item["all_levels_reflections_scores"] = log_item.get("all_levels_reflections_scores", []) + all_levels_reflections_scores
                log_item["all_levels_implementations"] = log_item.get("all_levels_implementations", []) + all_levels_implementations
                
                break

        # Write the updated logs to second_stage_json after each iteration
        write_jsonl(second_stage_json, logs_copy, append=False, key=None)

        print_v(f"second pass: completed {i+1}/{num_items}: acc = {round(num_success/(i+1), 4)}")

        # write memory bank to file after each item
        with open(mem_bank_file_path, 'wb') as f:
            pkl.dump(memory_bank, f)
        
    print(colored(gpt_usage(backend=model_name), 'blue'))
    
    