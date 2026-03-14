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
from time import time
import random
from termcolor import colored
SIMPLE_COMPLETION_INSTRUCTION = "# Write the body of this function only."
SIMPLE_CHAT_INSTRUCTION = "You are a programming assistant. You will be given a function signature and docstring. You should fill in the following text of the missing function body. For example, the first line of the completion should have 4 spaces for the indendation so that it fits syntactically with the preceding signature."


def normalize_test_block(raw: str) -> str:
    """
    Normalize indentation of a HumanEval-style test block:
    - METADATA lines flush-left
    - One blank line
    - `def check(candidate):` flush-left
    - All `assert` lines indented by 4 spaces
    """
    
    lines = raw.splitlines()
    metadata = []
    asserts = []
    in_assert = False

    for line in lines:
        stripped = line.lstrip()
        if not in_assert:
            if stripped.startswith("def check"):
                in_assert = True
                # finalize metadata
                metadata.append("")  # blank line
                metadata.append(stripped)
            elif stripped:
                metadata.append(stripped)
        else:
            if stripped.startswith("assert"):
                asserts.append("    " + stripped)

    # Combine metadata and asserts
    normalized = metadata + asserts
    return "\n".join(normalized) + "\n"

def run_simple(
        dataset: List[dict],
        model_name: str,
        language: str,
        pass_at_k: int,
        log_path: str,
        verbose: bool,
        is_leetcode: bool = False,
        use_mistakes = False,
        is_game24= False,
        pitfall_agent = None,
        mistake_json_file = None,
        dataset_type: str = "humaneval",
        inner_iter=5, **kargs
    ) -> None:
    # load json
    # with open("retrieved_insights/mbpp/IIDModel/pre_insights.json", "r") as fp:
    #     insights_dict = json.load(fp)
    if is_game24:
        print (f"is_game24: {is_game24}")
        dataset = Game24Dataset("benchmarks/game24.csv")
        # dataset = dataset[:1]
        print (f"dataset length of game24: {len(dataset)}")
        exe = executor_factory("game24")
        gen = generator_factory("game24")
        model = model_factory(model_name)
        print_v = make_printv(verbose)
    else:
        exe = executor_factory(language, is_leet=is_leetcode)
        gen = generator_factory(language)
        model = model_factory(model_name)
        print_v = make_printv(verbose)
    failed_probs = []
    num_items = len(dataset)
    num_success = 0
    print (f"use mistake model: {use_mistakes}")

    # Import LiveCodeBench utilities if needed
    if dataset_type == 'livecodebench':
        from generators.livecodebench_utils import (
            format_livecodebench_prompt,
            parse_private_tests
        )

    for i, item in enumerate_resume(dataset, log_path):
        try:
            item['rand_insights'] = []
            refined_insights = None
            cur_pass = 0
            is_solved = False
            cur_func_impl = ""

            # Normalize field access based on dataset type
            if dataset_type == 'livecodebench':
                prompt = format_livecodebench_prompt(item, language="python", include_public_tests=True)
                identifier = item.get("question_title", item.get("question_id", f"problem_{i}"))
                test_cases = parse_private_tests(item)
            else:
                prompt = item["prompt"]
                identifier = item["entry_point"]
                test_code = item["test"]

            if use_mistakes:
                if mistake_json_file is not None:
                    refined_insights = mistake_json_file[i]['pitfall']
                elif pitfall_agent is not None:
                    refined_insights = pitfall_agent.generate(prompt, temperature=0.1)
                else:
                    refined_insights = gen.generate_pre_insights(prompt)
            # print (refined_insights)
            item['refined_insights'] = refined_insights
            start_time = time()
            fail_cnt = 0
            while cur_pass < pass_at_k:
                cur_func_impl = None
                while cur_func_impl is None:
                    cur_func_impl = gen.func_impl(prompt, model, "simple", mistake_insights=refined_insights, temperature=0.2)
                    # print ('h'*50)
                    # print ('***********************cur expr**********************')
                    print (cur_func_impl)
                    fail_cnt += 1
                    if fail_cnt > 3: break

                assert isinstance(cur_func_impl, str)
                # print (cur_func_impl)

                # Evaluate based on dataset type
                if dataset_type == 'livecodebench':
                    is_passing = exe.evaluate_livecodebench(identifier, cur_func_impl, test_cases, timeout=20)
                else:
                    is_passing = exe.evaluate(identifier, cur_func_impl, test_code, timeout=20 if is_leetcode else 6)
                if is_passing:
                    is_solved = True
                    num_success += 1
                    break
                cur_pass +=1
            
            end_time = time()
            item["runtime"] = end_time - start_time
            item["solution"] = cur_func_impl
            
            llm_cost = gpt_usage(backend=model_name)
            
            item["is_solved"] = is_solved
            item['cost'] = llm_cost['cost']
            item['completion_tokens'] = llm_cost['completion_tokens']
            item['prompt_tokens'] = llm_cost['prompt_tokens']
            # for k in item:
            #     print (k, item[k])
            write_jsonl(log_path, [item], append=True)
        
        except Exception as e:
            print(colored(f"Error processing item {i}: {e}",'red'))
            continue

        print("Failed problems:")
        pprint(failed_probs)
        print_v(f'completed {i+1}/{num_items}: acc = {round(num_success/(i+1), 2)}')
