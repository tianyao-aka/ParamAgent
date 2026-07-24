from termcolor import colored

from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count
from executors import executor_factory
from generators import generator_factory, model_factory
import sys
from typing import List
from gpt_usage import gpt_usage
from time import time
from copy import deepcopy
from pprint import pprint
import os

def run_dot(
    dataset: List[dict],
    model_name: str,
    language: str,
    max_iters: int,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
    is_leetcode: bool = False,
    use_parsing = False,
    device = 'cuda:0', **kargs
) -> None:
    exe = executor_factory(lang='QA', is_leet=is_leetcode)
    gen = generator_factory(lang='QA')
    model = model_factory(model_name)
    print_v = make_printv(verbose)
    
    num_items = len(dataset)
    num_success = resume_success_count(dataset)
    qa_agent = None
    print("Running DoT with QA")

    for i, item in enumerate_resume(dataset, log_path):
        try:
            cur_pass = 0
            is_solved = False
            diverse_reflections = []
            implementations = []
            test_feedback = []
            # cur_func_impl = ""
            cur_func_impl = None
            while cur_pass < pass_at_k and not is_solved:
                # first attempt
                start_time = time()
                fail_cnt = 0
                while cur_func_impl is None:
                    cur_func_impl = gen.func_impl(item["question"],item["context"], model, "simple",temperature=0.2)
                    fail_cnt += 1
                    if fail_cnt > 3:
                        break
                    # print (colored("First solution", 'red'))
                    # print (colored("answer:", cur_func_impl, 'green'))
                    # print (colored("golden-truth:", item['answer'], 'green'))
                implementations.append(cur_func_impl)
                assert isinstance(cur_func_impl, str)
                is_passing = exe.evaluate(cur_func_impl,item['answer'],timeout=5)
                # if solved, exit early
                if is_passing:
                    is_solved = is_passing
                    num_success += int(is_passing)
                    break

                # use self-reflection to iteratively improve
                cur_iter = 1
                cur_feedback = "Incorrect answer"
                while cur_iter < max_iters:
                    # get self-reflection
                    div_reflections = gen.self_reflection_diverse(
                        item["question"], cur_func_impl,item["context"], cur_feedback, model,diverse_reflections).split('\n\n')
                    
                    # print ('@@@@@@@@@@@@@@')
                    # print (div_reflections)
                    div_reflections = [ref for ref in div_reflections if len(ref) > 10]
                    diverse_reflections += div_reflections
                    cur_func_impl_copy = deepcopy(cur_func_impl)
                    # apply self-reflection in the next attempt
                    if isinstance(model, tuple):
                        model = model[0]

                    temp_implementations = []
                    div_reflections_feedbacks = []
                    
                    ref_id = 0    #! change back if needed
                    del exe
                    exe = executor_factory(lang='QA', is_leet=is_leetcode)
                    reflection = div_reflections[ref_id]
                    # print(f"Attempting reflection-{ref_id}:")
                    # pprint(reflection)
                    
                    new_func_impl = None
                    fail_cnt = 0
                    while new_func_impl is None:
                        new_func_impl = gen.func_impl(
                            item["question"],item["context"], model, "reflexion",prev_answers=cur_func_impl_copy,feedback=cur_feedback,
                            self_reflection=reflection, temperature = 0.2
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

                    cur_iter += 1
                cur_pass += 1
                
        except Exception as e:
            print ('error here:',e)
            continue
        end_time = time()
        llm_cost = gpt_usage(backend=model_name)
        print(llm_cost)

        item['runtime'] = end_time - start_time
        item["is_solved"] = is_solved
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
    
