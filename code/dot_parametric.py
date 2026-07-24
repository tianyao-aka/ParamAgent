import random
from copy import deepcopy
from pprint import pprint
from tqdm import tqdm
from termcolor import colored

from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count
from executors import executor_factory
from generators import generator_factory, model_factory
from time import time
from typing import List
import sys
from gpt_usage import gpt_usage

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

def run_dot(
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
    inner_iter: int =5,
    dataset_type: str = 'humaneval',
) -> None:
    exe = executor_factory(language, is_leet=is_leetcode)
    gen = generator_factory(language)
    model = model_factory(model_name)

    print_v = make_printv(verbose)
    
    primary_key = 'entry_point' #"task_id" if "task_id" in dataset[0].keys() else "name" #'entry_point' for HumanEval
    print("Running DoT with parametric knowledge")
    num_items = len(dataset)
    num_success = resume_success_count(dataset)
    for i, item in enumerate_resume(dataset, log_path):
        print (i)
        start_time = time()
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

            cur_pass = 0
            is_solved = False
            diverse_reflections = []
            implementations = []
            test_feedback = []
            all_levels_reflections_scores = []
            all_levels_implementations = []
            cur_func_impl = None
            while cur_pass < pass_at_k and not is_solved:
                if dataset_type == 'livecodebench':
                    # LiveCodeBench: use public tests for intermediate feedback
                    print("using public test cases for LiveCodeBench")
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


                # use self-reflection to iteratively improve
                init_iter = 0
                lst = list(range(8))
                random.shuffle(lst)
                while init_iter < inner_iter:
                    # get self-reflection
                    # LiveCodeBench uses 'pitfalls' key, HumanEval/MBPP use 'pitfall'
                    pitfall_key = 'pitfalls' if dataset_type == 'livecodebench' else 'pitfall'
                    if init_iter>0:
                        if 'high_temp_pitfall' in mistake_json_file[i]:
                            refined_insights = mistake_json_file[i]['high_temp_pitfall'][lst[init_iter]]
                        else:
                            refined_insights = pitfall_agent.generate(prompt, temperature=1.0)
                    else:
                        if 'high_temp_pitfall' in mistake_json_file[i]:
                            refined_insights = mistake_json_file[i][pitfall_key][lst[init_iter]] if isinstance(mistake_json_file[i][pitfall_key], list) else mistake_json_file[i][pitfall_key]
                        else:
                            refined_insights = pitfall_agent.generate(prompt, temperature=0.1)

                    new_func_impl = None
                    MAX_RETRIES = 3
                    for attempt in range(MAX_RETRIES):
                        new_func_impl = gen.func_impl(prompt, model, "simple", mistake_insights=refined_insights, temperature=0.2)
                        if new_func_impl is not None:
                            break
                        print(f"WARNING: Attempt {attempt+1}/{MAX_RETRIES} failed: gen.func_impl returned None")

                    if new_func_impl is None:
                        print(f"ERROR: Failed to generate implementation after {MAX_RETRIES} attempts for {identifier}, skipping init_iter {init_iter}")
                        # Use a placeholder to continue, will fail tests and trigger next iteration
                        new_func_impl = "def placeholder():\n    pass"

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
                    if is_passing or init_iter == inner_iter - 1:
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

                    init_iter += 1
                if is_solved:
                    break

                # conditional sampling on prior reflections to promote diversity
                cur_iter = 0
                while cur_iter < max_iters:
                    # one-shot sampling
                    # get multiple diverse reflections
                    if 'high_temp_pitfall' in mistake_json_file[i]:
                        refined_insights = mistake_json_file[i]['high_temp_pitfall'][lst[cur_iter]]
                    else:
                        refined_insights = pitfall_agent.generate(prompt, temperature=1.0)
                    div_reflections = gen.self_reflection_diverse_oneshot_parametric(
                        cur_func_impl, cur_feedback, model, diverse_reflections,refined_insights).split("\n\n")
                    
                    # filter out reflections if they are less than few characters
                    div_reflections = [ref for ref in div_reflections if len(ref) > 10]
                    diverse_reflections += div_reflections
                    
                    cur_func_impl_copy = deepcopy(cur_func_impl)
                    
                    temp_implementations = []
                    reflections_scores = []
                    div_reflections_feedbacks = []
                    
                    ref_id = 0    #! change back if needed
                    pbar = tqdm(total=len(div_reflections))
                    while ref_id < min(len(div_reflections), 2):  #! only perform one reflection
                        #re-init executor
                        del exe
                        exe = executor_factory(language, is_leet=is_leetcode)

                        reflection = div_reflections[ref_id]
                        print(f"Attempting reflection-{ref_id}:")
                        pprint(reflection)
                        print()
                        
                        # apply self-reflection in the next attempt
                        new_func_impl = None
                        MAX_RETRIES = 3
                        for attempt in range(MAX_RETRIES):
                            new_func_impl = gen.func_impl(
                                func_sig=prompt,
                                model=model,
                                strategy="reflexion",
                                prev_func_impl=cur_func_impl_copy,
                                feedback=cur_feedback,
                                self_reflection=reflection,
                                temperature=0.2,
                                ref_chat_instruction='dot',
                                mistake_insights=None,
                            )
                            if new_func_impl is not None:
                                break
                            print(f"WARNING: Reflexion attempt {attempt+1}/{MAX_RETRIES} failed: gen.func_impl returned None")

                        if new_func_impl is None:
                            print(f"ERROR: Failed to generate reflexion-based implementation after {MAX_RETRIES} attempts for {identifier}, skipping reflection {ref_id}")
                            # Skip this reflection and try the next one
                            ref_id += 1
                            pbar.update(1)
                            continue

                        cur_func_impl = new_func_impl

                        try:
                            assert isinstance(cur_func_impl, str)
                        except:
                            print("ERROR: cur_func_impl is not a string, regenerating func impl.")
                            continue

                        # Will be used later to sample a probable solution
                        temp_implementations.append(cur_func_impl)
                    
                        # check if all internal unit tests pass
                        is_passing, cur_feedback, _ = exe.execute(
                            cur_func_impl, tests_i)
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
                    
                    pbar.close()
                    
                    #log reflection scores and given level implementations
                    all_levels_reflections_scores.append(reflections_scores)
                    all_levels_implementations.append(temp_implementations)
                    
                    if is_solved:
                        break
                    
                    #sample likely implementation
                    sampled_impl_idx = random.choices(range(len(temp_implementations)), weights=reflections_scores, k=1)[0]
                    cur_func_impl = temp_implementations[sampled_impl_idx]
                    
                    # set cur_feedback to the corresponding sampled div-reflection
                    cur_feedback = div_reflections_feedbacks[sampled_impl_idx]
                    
                    cur_iter += 1
                cur_pass += 1
                
            
        except Exception as e:
            print(colored(f"Error: {e}", 'red'))
            print('-----------------')
        end_time = time()
            
        llm_cost = gpt_usage(backend=model_name)
        print(llm_cost)
        item["runtime"] = end_time - start_time
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
    print(colored(gpt_usage(backend=model_name), 'blue'))

