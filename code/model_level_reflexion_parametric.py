from termcolor import colored

from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count
from executors import executor_factory
from generators import generator_factory, model_factory
import sys
from typing import List
from gpt_usage import gpt_usage
from time import time
import random

# LiveCodeBench utilities - imported conditionally when needed
try:
    from generators.livecodebench_utils import (
        format_livecodebench_prompt,
        parse_private_tests,
        parse_public_tests,
        TestCase
    )
    from reflexion import _evaluate_with_feedback_livecodebench
    LIVECODEBENCH_AVAILABLE = True
except ImportError:
    LIVECODEBENCH_AVAILABLE = False
def run_simple(
    dataset: List[dict],
    model_name: str,
    language: str,
    max_iters: int,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
    is_leetcode: bool = False,
    visible_tests: any = None,
    use_mistakes: bool = True,
    is_game24: bool = False,
    pitfall_agent=None,
    mistake_json_file = None,
    inner_iter: int = 5,
    dataset_type: str = 'humaneval',
) -> None:
    exe = executor_factory(language, is_leet=is_leetcode)
    gen = generator_factory(language)
    model = model_factory(model_name)
    print_v = make_printv(verbose)
    
    num_items = len(dataset)
    num_success = resume_success_count(dataset)
    print("Running Reflexion with parametric knowledge")
    
    for i, item in enumerate_resume(dataset, log_path):
        try:
            # Normalize fields based on dataset type
            if dataset_type == 'livecodebench':
                prompt = format_livecodebench_prompt(item, language="python", include_public_tests=True)
                identifier = item.get("question_title", item.get("question_id", f"problem_{i}"))
                public_tests = parse_public_tests(item)  # For intermediate feedback
                private_tests = parse_private_tests(item)  # For final evaluation
                test_code = None  # Not used for LiveCodeBench
            else:
                prompt = item["prompt"]
                identifier = item["entry_point"]
                test_code = item["test"]
                public_tests = None
                private_tests = None

            refined_insights = None
            cur_pass = 0
            is_solved = False
            reflections = []
            implementations = []
            test_feedback = []
            # cur_func_impl = ""
            cur_func_impl = None
            while cur_pass < pass_at_k and not is_solved:
                if dataset_type == 'livecodebench':
                    # LiveCodeBench: use public tests for intermediate feedback
                    print("using public test cases for LiveCodeBench")
                    tests_i = public_tests
                elif is_leetcode:
                    tests_i = item['visible_tests']
                else:
                    if visible_tests and 'mbpp' not in log_path.lower():
                        # Use visible test cases
                        print("using visible test cases")
                        tests_i = visible_tests[identifier]['given_tests']

                    elif 'mbpp' not in log_path.lower():
                        tests_i = item['visible_tests']
                    else:
                        print("generating synthetic test cases")
                        tests_i = gen.internal_tests(prompt, model, 1)

                # first attempt
                lst = list(range(8))
                random.shuffle(lst)
                fail_cnt = 0
                while cur_func_impl is None:
                    if use_mistakes:
                        if mistake_json_file is not None:
                            # LiveCodeBench uses 'pitfalls' key, HumanEval/MBPP use 'pitfall'
                            pitfall_key = 'pitfalls' if dataset_type == 'livecodebench' else 'pitfall'
                            refined_insights = mistake_json_file[i][pitfall_key]
                        elif pitfall_agent is not None:
                            refined_insights = pitfall_agent.generate(prompt)
                        else:
                            # use gpt-4o-mini to generate pitfalls
                            refined_insights = gen.generate_pre_insights(prompt)
                        item['refined_insights'] = refined_insights
                        start_time = time()
                        cur_func_impl = gen.func_impl(prompt, model, "simple",mistake_insights=refined_insights,temperature=0.2)
                    else:
                        cur_func_impl = gen.func_impl(prompt, model, "simple")
                    if cur_func_impl is None:
                        fail_cnt += 1
                    if fail_cnt>1:
                        break
                implementations.append(cur_func_impl)
                assert isinstance(cur_func_impl, str)

                # Test on intermediate tests (public for LiveCodeBench, visible for HumanEval)
                if dataset_type == 'livecodebench':
                    is_passing, feedback = _evaluate_with_feedback_livecodebench(
                        exe, identifier, cur_func_impl, tests_i, timeout=20
                    )
                    test_feedback.append(feedback)
                else:
                    is_passing, feedback, _ = exe.execute(cur_func_impl, tests_i)
                    test_feedback.append(feedback)

                # If intermediate tests pass, evaluate on final tests
                if is_passing:
                    if dataset_type == 'livecodebench':
                        is_passing = exe.evaluate_livecodebench(identifier, cur_func_impl, private_tests, timeout=20)
                    else:
                        is_passing = exe.evaluate(identifier, cur_func_impl, test_code, timeout=10)
                    is_solved = is_passing
                    num_success += int(is_passing)
                    break

                # use self-reflection to iteratively improve
                cur_iter = 0
                cur_feedback = feedback
                # cur_feedback = "incorrect implementation"
                
                while cur_iter < max_iters:
                    # get self-reflection (refined insights for next attempt)
                    if mistake_json_file and 'high_temp_pitfall' in mistake_json_file[i]:
                        refined_insights = mistake_json_file[i]['high_temp_pitfall'][lst[cur_iter]]
                    elif pitfall_agent:
                        refined_insights = pitfall_agent.generate(prompt, temperature=1.0)
                    else:
                        refined_insights = None

                    new_func_impl = None
                    fail_cnt = 0
                    while new_func_impl is None:
                        new_func_impl = gen.func_impl(prompt, model, "simple", mistake_insights=refined_insights, temperature=0.2)
                        fail_cnt += 1
                        if fail_cnt > 1:
                            break
                    cur_func_impl = new_func_impl

                    implementations.append(cur_func_impl)
                    assert isinstance(cur_func_impl, str)

                    # check if all internal unit tests pass
                    if dataset_type == 'livecodebench':
                        is_passing, cur_feedback = _evaluate_with_feedback_livecodebench(
                            exe, identifier, cur_func_impl, tests_i, timeout=20
                        )
                        test_feedback.append(cur_feedback)
                    else:
                        is_passing, cur_feedback, _ = exe.execute(cur_func_impl, tests_i)
                        test_feedback.append(cur_feedback)

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
                    # cur_feedback = "incorrect implementation"

                    cur_iter += 1
                cur_pass += 1
        except Exception as e:
            print (colored(f"Error processing item {i}: {e}", 'red'))
            import traceback
            traceback.print_exc()
            continue
        end_time = time()
        llm_cost = gpt_usage(backend=model_name)
        item['runtime'] = end_time - start_time
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
