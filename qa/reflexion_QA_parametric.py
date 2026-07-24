from termcolor import colored

from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count
from executors import executor_factory
from generators import generator_factory, model_factory
import sys
from typing import List
from gpt_usage import gpt_usage
from time import time
from LoRA_Llama3_QA_inference import QADecomposer

def run_reflexion(
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
    device = 'cuda:0', **kargs
) -> None:
    exe = executor_factory(lang='QA', is_leet=is_leetcode)
    gen = generator_factory(lang='QA')
    model = model_factory(model_name)
    print_v = make_printv(verbose)
    
    num_items = len(dataset)
    num_success = resume_success_count(dataset)
    qa_agent = None
    
    print("Running Reflexion with QA parsing")
    if use_parsing:
        if not insight_json_file:
            print ("Using question decomposition model for parsing insights")
            qa_agent = QADecomposer(device=device,adapter_dir="./LoRA/qa/")
        print ('use existing json file for qa insights')
    for i, item in enumerate_resume(dataset, log_path):
        refined_insights = None
        try:
            cur_pass = 0
            is_solved = False
            reflections = []
            implementations = []
            test_feedback = []
            # cur_func_impl = ""
            cur_func_impl = None
            if use_parsing:
                if not insight_json_file:
                    refined_insights = qa_agent.generate(item["question"])
                    # refined_insights = gen.generate_pre_insights(item["question"])
                    # print ('***********************refined insights**********************')
                    # print (refined_insights)
                    item['refined_insights'] = refined_insights
                    
                else:
                    refined_insights = insight_json_file[i]['refined_insights']
                    item['refined_insights'] = refined_insights
            while cur_pass < pass_at_k and not is_solved:
                # first attempt
                start_time = time()
                while cur_func_impl is None:
                    cur_func_impl = gen.func_impl(item["question"],item["context"], model, "simple",question_decomposition= refined_insights)
                    print ('***********************generated answer**********************')
                    print ('answer:',cur_func_impl)
                    print ('golden truth:',item['answer'])
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
                    reflection = gen.self_reflection_parametric(
                        item["question"], cur_func_impl,item["context"], cur_feedback,refined_insights, model)
                    reflections += [reflection]
                    # print (colored(f'reflection:\n {reflection}','cyan'))
                    # apply self-reflection in the next attempt
                    if isinstance(model, tuple):
                        model = model[0]
                    
                    new_func_impl = None
                    while new_func_impl is None:
                        new_func_impl = gen.func_impl(
                            item["question"],item["context"], model, "reflexion",prev_answers=cur_func_impl,feedback=cur_feedback,
                            self_reflection=reflection, question_decomposition=refined_insights, temperature = 0.2
                        )
                    
                    cur_func_impl = new_func_impl
                    print (colored(f'answer: {cur_func_impl}','red'))
                    print (colored(f'golden truth: {item["answer"]}','red'))
                    implementations.append(cur_func_impl)
                    assert isinstance(cur_func_impl, str)
                    
                    # check if all internal unit tests pass
                    is_passing = exe.evaluate(cur_func_impl,item['answer'],timeout=5)
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
                
        except:
            continue
        end_time = time()
        llm_cost = gpt_usage(backend=model_name)

        
        llm_cost = gpt_usage(backend=model_name)
        print(llm_cost)

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
    
    