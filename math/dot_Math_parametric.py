from termcolor import colored

from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count
from executors import executor_factory
from generators import generator_factory, model_factory
import sys
from gpt_usage import gpt_usage
from time import time,sleep
from copy import deepcopy
from pprint import pprint
from LoRA_Llama3_QA_inference import QADecomposer
import random
from typing import Any, Dict, List


def filter_pitfalls_high_temp(record):
    """
    Return a new dict where the 'pitfalls_high_temp' list has been filtered to only include
    strings that mention 'mistake' (case‑insensitive) and are at least 10 characters long.

    Args:
        record (Dict[str, Any]):
            Input dictionary expected to have a key 'pitfalls_high_temp' mapping to List[str].

    Returns:
        Dict[str, Any]:
            A shallow copy of `record` with 'pitfalls_high_temp' replaced by the filtered list.
            If the key is missing or not a list, returns the original dict unchanged.
    """
    # Make a shallow copy so we don't modify the original
    new_record = record.copy()

    # Retrieve the original list, if present
    pitfalls = new_record.get("pitfalls_high_temp")

    # Only proceed if it's actually a list of strings
    if isinstance(pitfalls, list):
        filtered: List[str] = []
        for item in pitfalls:
            if isinstance(item, str):
                text = item.lower()
                # Keep only strings containing 'mistake' and with length >= 10
                if "mistake" in text and len(item) >= 10:
                    filtered.append(item)
        # Update the copy with the filtered list
        new_record["pitfalls_high_temp"] = filtered

    return new_record


def run_dot(
    dataset: List[dict],
    model_name: str,
    language: str,
    max_iters: int,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
    is_leetcode: bool = False,
    use_parsing = True,
    insight_json_file = None,
    device = 'cuda:0',Math_agent = None, 
    **kargs
) -> None:
    exe = executor_factory(lang='math', is_leet=is_leetcode)
    gen = generator_factory(lang='math')
    model = model_factory(model_name)
    print_v = make_printv(verbose)
    
    num_items = len(dataset)
    num_success = resume_success_count(dataset)
    print("Running DoT parametric for Maths")

    
    for i, item in enumerate_resume(dataset, log_path):
        try:
            cur_pass = 0
            is_solved = False
            diverse_reflections = []
            implementations = []
            test_feedback = []
            # cur_func_impl = ""
            cur_func_impl = None
            lst = list(range(8))
            random.shuffle(lst)
            while cur_pass < pass_at_k and not is_solved:
                init_iter = 0
                # first attempt
                start_time = time()
                while init_iter < 5:
                    print (colored(f"init iter: {init_iter}", 'red'))
                    if use_parsing:
                        if Math_agent is not None:
                            if init_iter==0:
                                refined_insights = Math_agent.generate(item["problem"],temperature=0.1)
                            else:
                                refined_insights = Math_agent.generate(item["problem"],temperature=1.0)
                        else:
                            if init_iter==0:
                                refined_insights = insight_json_file[i]['pitfalls']
                            else:
                                new_insight_file = filter_pitfalls_high_temp(insight_json_file[i])
                                N = len(new_insight_file['pitfalls_high_temp'])
                                refined_insights = new_insight_file['pitfalls_high_temp'][lst[init_iter-1]%N]
                    item['mistake_insights'] = refined_insights
                    init_iter += 1
                    # print (colored("refined insights:", refined_insights, 'blue'))
                    fail_cnt = 0
                    while cur_func_impl is None:
                        cur_func_impl = gen.func_impl(item["problem"], model, "simple",temperature=0.2,mistake_insights= refined_insights)
                        fail_cnt += 1
                        if fail_cnt > 3:
                            break
                        # print (colored("First solution", 'red'))
                        # print (colored("answer:", cur_func_impl, 'green'))
                        # print (colored("golden-truth:", item['answer'], 'green'))
                    implementations.append(cur_func_impl)
                    assert isinstance(cur_func_impl, str)
                    is_passing = exe.evaluate(cur_func_impl,item['answer'],timeout=10)
                    # print (colored(f"Test feedback: {is_passing}", 'cyan'))
                    # print (colored(f"predict:{cur_func_impl} \n\n golden:{item['answer']}",'yellow'))
                    # if solved, exit early
                    if is_passing:
                        is_solved = is_passing
                        num_success += int(is_passing)
                        break
                if is_solved:
                    break
                init_iter += 1
                # use self-reflection to iteratively improve
                cur_iter = 0
                cur_feedback = "Incorrect answer"
                
                while cur_iter < max_iters:
                    # get self-reflection
                    if Math_agent is not None:
                        refined_insights = Math_agent.generate(item["problem"],temperature=1.0)
                    else:
                        refined_insights = new_insight_file['pitfalls_high_temp'][lst[cur_iter]%N]
                    div_reflections = gen.self_reflection_diverse_parametric(
                        item["problem"], cur_func_impl, cur_feedback, model,diverse_reflections,refined_insights).split('\n')
                    div_reflections = [ref for ref in div_reflections if len(ref) > 7]
                    diverse_reflections += div_reflections
                    cur_func_impl_copy = deepcopy(cur_func_impl)
                    # apply self-reflection in the next attempt
                    if isinstance(model, tuple):
                        model = model[0]

                    # temp_implementations = []
                    # div_reflections_feedbacks = []
                    
                    ref_id = 0    #! change back if needed
                    while ref_id < min(len(div_reflections), 3):
                        del exe
                        exe = executor_factory(lang='math', is_leet=is_leetcode)
                        reflection = div_reflections[ref_id]
                        
                        new_func_impl = None
                        fail_cnt = 0    
                        while new_func_impl is None:
                            new_func_impl = gen.func_impl(
                                item["problem"],model, "reflexion",prev_answers=cur_func_impl_copy,feedback=cur_feedback,
                                self_reflection=reflection, temperature = 0.2, mistake_insights=refined_insights
                            )
                            fail_cnt += 1
                            if fail_cnt > 3:
                                break
                            
                        print (colored(f"new answer:\n {new_func_impl}; golden: {item['answer']}", 'cyan'))
                        cur_func_impl = new_func_impl
                        
                        implementations.append(cur_func_impl)
                        assert isinstance(cur_func_impl, str)
                        
                        # check if all internal unit tests pass
                        is_passing = exe.evaluate(cur_func_impl,item['answer'],timeout=5)
                        print (colored(f"Test feedback: {is_passing}", 'cyan'))
                        test_feedback.append("Incorrect answer" if not is_passing else "Correct answer")

                        # if solved, check if it passes the real tests, exit early
                        if is_passing or cur_iter == max_iters - 1:
                            if is_passing:
                                item["solution"] = cur_func_impl
                                is_solved = True
                                num_success += 1
                            break
                        
                        ref_id += 1
                        if is_solved:
                            break

                    cur_iter += 1
                    if is_solved:
                        break
                cur_pass += 1
        except Exception as e:
            print (colored(f"Error in item {i}: {e}", 'red'))
            continue
        end_time = time()
        llm_cost = gpt_usage(backend=model_name)
        print(llm_cost)

        item['runtime'] = end_time - start_time
        item["is_solved"] = is_solved
        print (colored(f"Item {i} solved: {is_solved}", 'green' if is_solved else 'green'))
        item["reflections"] = diverse_reflections
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
    