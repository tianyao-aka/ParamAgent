from pprint import pprint
from utils import enumerate_resume, make_printv, write_jsonl
from executors import executor_factory
from generators import generator_factory, model_factory
import os
from typing import List
import textwrap
from gpt_usage import gpt_usage
import sys
import json
# from LoRA_Llama3_QA_inference import QADecomposer  # Commented out - import handled in main script
from time import time


def run_simple(
        dataset: List[dict],
        model_name: str,
        language: str,
        pass_at_k: int,
        log_path: str,
        verbose: bool,
        is_leetcode: bool = False,
        use_parsing = False,
        qa_agent = None,
        insight_json_file = None,
        device = 'cuda:0',
        **kargs
    ) -> None:
    # load json

    exe = executor_factory(lang='QA', is_leet=is_leetcode)
    gen = generator_factory(lang='QA')
    model = model_factory(model_name)
    print_v = make_printv(verbose)
    failed_probs = []
    num_items = len(dataset)
    num_success = 0
    print (f"use question decomposition model: {use_parsing}")
    print_v("Using question decomposition model for parsing insights")
    for i, item in enumerate_resume(dataset, log_path):
        refined_insights = None
        try:
            refined_insights = None
            cur_pass = 0
            is_solved = False
            cur_func_impl = ""
            
            if use_parsing:
                if insight_json_file is not None:
                    refined_insights = insight_json_file[i]['insight']
                elif qa_agent is not None:
                    refined_insights = qa_agent.generate(item["question"],temperature=0.1)
                # refined_insights = gen.generate_pre_insights(item["question"])
                # print ('***********************refined insights**********************')
                # print (refined_insights)
                item['refined_insights'] = refined_insights
            
            while cur_pass < pass_at_k:
                start_time = time()
                cur_func_impl = gen.func_impl(item["question"],item["context"], model, "simple",question_decomposition= refined_insights)
                print ('***********************generated answer**********************')
                print ('answer:',cur_func_impl)
                print ('golden truth:',item['answer'])
                
                assert isinstance(cur_func_impl, str)
                # print (cur_func_impl)
                is_passing = exe.evaluate(cur_func_impl,item['answer'],timeout=5)
                if is_passing:
                    is_solved = True
                    num_success += 1
                    break
                cur_pass += 1
            end_time = time()
            item["runtime"] = end_time - start_time
            item["solution"] = cur_func_impl
            
            llm_cost = gpt_usage(backend=model_name)
            print(llm_cost)
            
            item["is_solved"] = is_solved
            item['cost'] = llm_cost['cost']
            item['completion_tokens'] = llm_cost['completion_tokens']
            item['prompt_tokens'] = llm_cost['prompt_tokens']
            # for k in item:
            #     print (k, item[k])
            write_jsonl(log_path, [item], append=True)
        
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            failed_probs.append(item['task_id'])
            continue

        print("Failed problems:")
        pprint(failed_probs)
        print_v(f'completed {i+1}/{num_items}: acc = {round(num_success/(i+1), 2)}')


