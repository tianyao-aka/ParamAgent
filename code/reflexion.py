from termcolor import colored
from time import time
from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count
from executors import executor_factory
from generators import generator_factory, model_factory
import sys
from typing import List
from gpt_usage import gpt_usage
import subprocess
import tempfile
import os


def _evaluate_with_feedback_livecodebench(exe, identifier, program, test_cases, timeout=20):
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
                feedback_lines.append(f"Test {i+1} FAILED - Runtime error:\n{result.stderr[:200]}")
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


def run_reflexion(
    dataset: List[dict],
    model_name: str,
    language: str,
    max_iters: int,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
    is_leetcode: bool = False,
    visible_tests: any = None,
    use_mistakes: bool = False,
    dataset_type: str = "humaneval",
    **kargs
) -> None:
    exe = executor_factory(language, is_leet=is_leetcode)
    gen = generator_factory(language)
    model = model_factory(model_name)
    print_v = make_printv(verbose)

    # Import LiveCodeBench utilities if needed
    if dataset_type == 'livecodebench':
        from generators.livecodebench_utils import (
            format_livecodebench_prompt,
            parse_public_tests,
            parse_private_tests
        )

    num_items = len(dataset)
    num_success = resume_success_count(dataset)
    print("Running Reflexion")
    
    for i, item in enumerate_resume(dataset, log_path):
        try:
            # Normalize field access based on dataset type
            if dataset_type == 'livecodebench':
                prompt = format_livecodebench_prompt(item, language="python", include_public_tests=True)
                identifier = item.get("question_title", item.get("question_id", f"problem_{i}"))
                public_tests = parse_public_tests(item)  # For intermediate feedback
                private_tests = parse_private_tests(item)  # For final evaluation
            else:
                prompt = item["prompt"]
                identifier = item["entry_point"]
                test_code = item["test"]

            cur_pass = 0
            is_solved = False
            reflections = []
            implementations = []
            test_feedback = []
            # cur_func_impl = ""
            cur_func_impl = None
            while cur_pass < pass_at_k and not is_solved:
                if dataset_type == 'livecodebench':
                    tests_i = public_tests  # Use public tests for reflexion loop
                elif is_leetcode:
                    tests_i = item['visible_tests']
                else:
                    if visible_tests and 'mbpp' not in log_path.lower():
                        # Use visible test cases
                        print("using visible test cases")
                        tests_i = visible_tests[identifier]['given_tests']

                    elif 'mbpp' not in log_path.lower():
                        print("using visible test cases for MBPP")
                        tests_i = item['visible_tests']
                    else:
                        print("generating synthetic test cases")
                        tests_i = gen.internal_tests(prompt, model, 1)

                        # Use original test cases
                        # print("using original test cases")
                        # tests_i = [case.lstrip().replace('candidate', item['entry_point']) for case in item['test'].split('\n')[1:-1] if 'assert' in case]

                # first attempt
                fail_cnt = 0
                start_time = time()
                while cur_func_impl is None:
                    cur_func_impl = gen.func_impl(prompt, model, "simple", temperature=0.2)
                    fail_cnt += 1
                    if fail_cnt > 1: break
                implementations.append(cur_func_impl)
                assert isinstance(cur_func_impl, str)

                # Execute intermediate tests and get feedback
                if dataset_type == 'livecodebench':
                    # Execute against public tests with detailed feedback
                    is_passing, feedback = _evaluate_with_feedback_livecodebench(
                        exe, identifier, cur_func_impl, public_tests, timeout=20
                    )
                    test_feedback.append(feedback)

                    # If passing public tests, evaluate with private tests
                    if is_passing:
                        is_passing = exe.evaluate_livecodebench(identifier, cur_func_impl, private_tests, timeout=20)
                        is_solved = is_passing
                        num_success += int(is_passing)
                        if is_passing:
                            break
                else:
                    # Original HumanEval/MBPP logic
                    is_passing, feedback, _ = exe.execute(cur_func_impl, tests_i)
                    test_feedback.append(feedback)

                    # if solved, exit early
                    if is_passing:
                        is_passing = exe.evaluate(identifier, cur_func_impl, test_code, timeout=10)
                        is_solved = is_passing
                        num_success += int(is_passing)
                        break

                # use self-reflection to iteratively improve
                cur_iter = 1
                cur_feedback = feedback
                # cur_feedback = "incorrect implementation"
                while cur_iter < max_iters:
                    # get self-reflection
                    reflection = gen.self_reflection(
                        cur_func_impl, cur_feedback, model)
                    reflections += [reflection]

                    # apply self-reflection in the next attempt
                    if isinstance(model, tuple):
                        model = model[0]
                    
                    new_func_impl = None
                    fail_cnt = 0
                    while new_func_impl is None:
                        new_func_impl = gen.func_impl(
                            func_sig=prompt,
                            model=model,
                            strategy="reflexion",
                            prev_func_impl=cur_func_impl,
                            feedback=cur_feedback,
                            self_reflection=reflection,
                            temperature=0.2
                        )
                        fail_cnt += 1
                        if fail_cnt > 3:
                            break
                    cur_func_impl = new_func_impl

                    implementations.append(cur_func_impl)
                    assert isinstance(cur_func_impl, str)

                    # check if all internal unit tests pass
                    if dataset_type == 'livecodebench':
                        # Test against public tests with feedback
                        is_passing, cur_feedback = _evaluate_with_feedback_livecodebench(
                            exe, identifier, cur_func_impl, public_tests, timeout=20
                        )
                        test_feedback.append(cur_feedback)

                        # On last iteration or if passing, check private tests
                        if is_passing or cur_iter == max_iters - 1:
                            is_passing = exe.evaluate_livecodebench(identifier, cur_func_impl, private_tests, timeout=20)
                            if is_passing:
                                item["solution"] = cur_func_impl
                                is_solved = True
                                num_success += 1
                            break
                    else:
                        # Original HumanEval/MBPP logic
                        is_passing, cur_feedback, _ = exe.execute(cur_func_impl, tests_i)
                        test_feedback.append(cur_feedback)

                        # if solved, check if it passes the real tests, exit early
                        if is_passing or cur_iter == max_iters - 1:
                            is_passing = exe.evaluate(identifier, cur_func_impl, test_code, timeout=10)
                            if is_passing:
                                item["solution"] = cur_func_impl
                                is_solved = True
                                num_success += 1
                            break

                    cur_iter += 1
                cur_pass += 1
                
        except:
            continue
        end_time = time()
        llm_cost = gpt_usage(backend=model_name)

        item["runtime"] = end_time - start_time
        item["is_solved"] = is_solved
        item["reflections"] = reflections
        item["implementations"] = implementations
        item["test_feedback"] = test_feedback
        item["solution"] = cur_func_impl
        item['cost'] = llm_cost['cost']
        item['completion_tokens'] = llm_cost['completion_tokens']
        item['prompt_tokens'] = llm_cost['prompt_tokens']
        write_jsonl(log_path, [item], append=True)

        print_v(
            f'completed {i+1}/{num_items}: acc = {round(num_success/(i+1), 2)}')

    print(colored(gpt_usage(backend=model_name), 'blue'))
    
    