from termcolor import colored

from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count, read_jsonl
from executors import executor_factory
from generators import generator_factory, model_factory
import sys
from gpt_usage import gpt_usage
from time import time,sleep
from copy import deepcopy
from pprint import pprint
# from LoRA_Llama3_QA_inference import QADecomposer  # Commented out - unused import
import random
from typing import Any, Dict, List
from tqdm import tqdm
import pickle as pkl
import os

# memory bank imports
from memory_utils import (
    get_openai_embedding,
    get_top_k_closest,
)


def _math_prompt_string(problem: str) -> str:
    """
    Compose a compact text for embedding retrieval in math QA.

    Args:
        problem (str): The math problem text.

    Returns:
        str: Normalized text used for embedding-based retrieval.
    """
    return f"Problem:\n{problem}"


def _build_augmented_problem_from_examples(examples: list, current_problem: str) -> str:
    """
    Build a few-shot augmented input by inlining similar solved math exemplars,
    then appending the current problem statement.

    Args:
        examples (list): List of trajectories from memory bank (each must contain
                         'problem', 'gen_solution').
        current_problem (str): The current problem statement.

    Returns:
        str: An augmented problem string fed to the math generator.
    """
    header = "Below are similar solved math examples. Learn their solution style.\n\n"
    blocks = []
    for i, ex in enumerate(examples, start=1):
        p = ex.get("problem", "")
        s = ex.get("gen_solution", "")
        blocks.append(
            f"[Example {i}]\n"
            f"Problem:\n{p}\n\n"
            f"Solution:\n{s}\n\n"
        )
    trailer = (
        "Now solve the NEW problem. Provide the final numeric/symbolic answer at the end.\n\n"
        "[CURRENT PROBLEM]\n"
        f"{current_problem}"
    )
    return header + "".join(blocks) + trailer


def _compose_reflexion_few_shot_from_trajectory(traj: dict) -> str:
    """
    Compose a small 'few-shot' style reflexion hint from a past positive trajectory.

    Args:
        traj (dict): A memory bank trajectory containing the keys:
                     'prev_solution', 'reflection', 'gen_solution'.

    Returns:
        str: A textual block that shows previous solution, reflection, and improved solution.
    """
    prev_sol = traj.get("prev_solution", "")
    refl = traj.get("reflection", "")
    improved = traj.get("gen_solution", "")
    return (
        "Example 1 (Reflexion Improvement):\n"
        "[previous solution]:\n"
        f"{prev_sol}\n\n"
        "[reflection]:\n"
        f"{refl}\n\n"
        "[improved solution]:\n"
        f"{improved}\n"
    )


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
    device = 'cuda:0',
    Math_agent = None, 
    **kargs
) -> None:
    exe = executor_factory(lang='math', is_leet=is_leetcode)
    gen = generator_factory(lang='math')
    model = model_factory(model_name)
    print_v = make_printv(verbose)
    
    num_items = len(dataset)
    num_success = resume_success_count(dataset)
    print("Running DoT parametric for Maths")

    # init memory bank related file paths
    root_path = "/".join(log_path.split("/")[:-1])
    mem_bank_file_path = os.path.join(root_path, "mem_bank.pkl")
    failed_probs_path = os.path.join(root_path, "failed_probs.pkl")

    # check if memory-bank already exists
    if os.path.exists(mem_bank_file_path):
        with open(mem_bank_file_path, "rb") as f:
            memory_bank = pkl.load(f)
    else:
        memory_bank = {
            "positive_trajectories": [],
            "negative_trajectories": [],
        }

    if os.path.exists(failed_probs_path):
        with open(failed_probs_path, "rb") as f:
            failed_problems = pkl.load(f)
    else:
        failed_problems = []


    # Snapshot the entire first-stage log to a new JSONL
    root_path = '/'.join(log_path.split('/')[:-1])
    first_stage_json = root_path + '/first_stage_log.jsonl'
    second_stage_json = root_path + '/second_stage_log.jsonl'
    if os.path.exists(second_stage_json):
        skip_first = True
    else:
        skip_first = False
    # --------------------------
    # First Pass
    # --------------------------
    for i, item in enumerate_resume(dataset, log_path):
        if skip_first: break
        try:
            cur_pass = 0
            is_solved = False
            diverse_reflections = []
            implementations = []
            test_feedback = []
            all_levels_reflections_scores = []
            all_levels_implementations = []
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
                                # Check if pitfalls_high_temp exists, otherwise use pitfalls
                                if 'pitfalls_high_temp' in insight_json_file[i]:
                                    new_insight_file = filter_pitfalls_high_temp(insight_json_file[i])
                                    N = len(new_insight_file['pitfalls_high_temp'])
                                    refined_insights = new_insight_file['pitfalls_high_temp'][lst[init_iter-1]%N]
                                else:
                                    refined_insights = insight_json_file[i]['pitfalls']
                    item['mistake_insights'] = refined_insights
                    init_iter += 1
                    fail_cnt = 0
                    while cur_func_impl is None:
                        cur_func_impl = gen.func_impl(item["problem"], model, "simple",temperature=0.2,mistake_insights= refined_insights)
                        fail_cnt += 1
                        if fail_cnt > 1:
                            break
                    implementations.append(cur_func_impl)
                    assert isinstance(cur_func_impl, str)
                    is_passing = exe.evaluate(cur_func_impl,item['answer'],timeout=10)
                    test_feedback.append("Correct answer" if is_passing else "Incorrect answer")
                    
                    # if solved, store positive trajectory & exit early
                    if is_passing:
                        trajectory = {
                            "problem": item["problem"],
                            "gen_solution": cur_func_impl,
                            "prompt_embedding": get_openai_embedding(
                                [_math_prompt_string(item["problem"])]
                            ),
                            "mistake_insights": refined_insights
                        }
                        memory_bank["positive_trajectories"].append(trajectory)
                        
                        is_solved = is_passing
                        num_success += int(is_passing)
                        item["solution"] = cur_func_impl
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
                        # Check if pitfalls_high_temp exists, otherwise use pitfalls
                        if 'pitfalls_high_temp' in insight_json_file[i]:
                            new_insight_file = filter_pitfalls_high_temp(insight_json_file[i])
                            N = len(new_insight_file['pitfalls_high_temp'])
                            refined_insights = new_insight_file['pitfalls_high_temp'][lst[cur_iter]%N]
                        else:
                            refined_insights = insight_json_file[i]['pitfalls']
                    div_reflections = gen.self_reflection_diverse_parametric(
                        item["problem"], cur_func_impl, cur_feedback, model,diverse_reflections,refined_insights).split('\n')
                    div_reflections = [ref for ref in div_reflections if len(ref) > 7]
                    diverse_reflections += div_reflections
                    cur_func_impl_copy = deepcopy(cur_func_impl)
                    # apply self-reflection in the next attempt
                    if isinstance(model, tuple):
                        model = model[0]

                    temp_solutions = []
                    reflections_scores = []
                    div_reflections_feedbacks = []
                    
                    ref_id = 0
                    while ref_id < min(len(div_reflections), 2):
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
                            if fail_cnt > 1:
                                break
                            
                        print (colored(f"new answer:\n {new_func_impl}; golden: {item['answer']}", 'cyan'))
                        cur_func_impl = new_func_impl
                        temp_solutions.append(cur_func_impl)
                        
                        implementations.append(cur_func_impl)
                        assert isinstance(cur_func_impl, str)
                        
                        # check if all internal unit tests pass
                        is_passing = exe.evaluate(cur_func_impl,item['answer'],timeout=5)
                        print (colored(f"Test feedback: {is_passing}", 'cyan'))
                        fb = "Correct answer" if is_passing else "Incorrect answer"
                        test_feedback.append(fb)
                        div_reflections_feedbacks.append(fb)
                        
                        # binary success score for weighted sampling
                        reflections_scores.append((1.0 if is_passing else 0.0) + 1e-8)

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

                    # bookkeeping
                    all_levels_reflections_scores.append(reflections_scores)
                    all_levels_implementations.append(temp_solutions)

                    # memory bank update at end-of-iter or success
                    if (cur_iter == max_iters - 1) or is_passing:
                        sampled_idx = (
                            random.choices(
                                range(len(temp_solutions)),
                                weights=reflections_scores,
                                k=1,
                            )[0]
                            if temp_solutions
                            else None
                        )
                        chosen_fb = (
                            div_reflections_feedbacks[sampled_idx]
                            if sampled_idx is not None
                            else cur_feedback
                        )
                        chosen_reflection = (
                            div_reflections[sampled_idx]
                            if (sampled_idx is not None and sampled_idx < len(div_reflections))
                            else (div_reflections[-1] if div_reflections else "")
                        )
                        chosen_solution = (
                            temp_solutions[sampled_idx] if sampled_idx is not None else cur_func_impl
                        )

                        trajectory = {
                            "problem": item["problem"],
                            "gen_solution": chosen_solution,
                            "reflection": chosen_reflection,
                            "test_feedback": chosen_fb,
                            "prev_solution": cur_func_impl_copy,
                            "prompt_embedding": get_openai_embedding(
                                [_math_prompt_string(item["problem"])]
                            ),
                            "refection_embedding": get_openai_embedding([chosen_reflection]),
                            "mistake_insights": refined_insights
                        }

                        if is_passing:
                            memory_bank["positive_trajectories"].append(trajectory)
                        else:
                            memory_bank["negative_trajectories"].append(trajectory)

                    cur_iter += 1
                    if is_solved:
                        break
                        
                    # sample likely solution for next iteration
                    if temp_solutions:
                        sampled_impl_idx = random.choices(
                            range(len(temp_solutions)),
                            weights=reflections_scores,
                            k=1,
                        )[0]
                        cur_func_impl = temp_solutions[sampled_impl_idx]
                        cur_feedback = div_reflections_feedbacks[sampled_impl_idx]
                        
                cur_pass += 1
                
        except Exception as e:
            print (colored(f"Error in item {i}: {e}", 'red'))
            llm_cost = gpt_usage(backend=model_name)
            print(llm_cost)

            item['runtime'] = end_time - start_time
            item["is_solved"] = is_solved
            print (colored(f"Item {i} solved: {is_solved}", 'green' if is_solved else 'green'))
            item["reflections"] = diverse_reflections
            item["implementations"] = implementations
            item["test_feedback"] = test_feedback
            item["solution"] = cur_func_impl
            item["all_levels_reflections_scores"] = all_levels_reflections_scores
            item["all_levels_implementations"] = all_levels_implementations
            item['cost'] = llm_cost['cost']
            item['completion_tokens'] = llm_cost['completion_tokens']
            item['prompt_tokens'] = llm_cost['prompt_tokens']
            write_jsonl(log_path, [item], append=True)
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
        item["all_levels_reflections_scores"] = all_levels_reflections_scores
        item["all_levels_implementations"] = all_levels_implementations
        item['cost'] = llm_cost['cost']
        item['completion_tokens'] = llm_cost['completion_tokens']
        item['prompt_tokens'] = llm_cost['prompt_tokens']
        write_jsonl(log_path, [item], append=True)

        print_v(
            f'completed {i+1}/{num_items}: acc = {round(num_success/(i+1), 2)}')
        
        if not is_solved:
            failed_problems.append(item)

        # write memory bank / failed list to disk after each item
        with open(mem_bank_file_path, "wb") as f:
            pkl.dump(memory_bank, f)
        with open(failed_probs_path, "wb") as f:
            pkl.dump(failed_problems, f)

    print("Finished first pass")
    
    # --------------------------
    # Second Pass (memory-augmented)
    # --------------------------
    memory_bank = pkl.load(open(mem_bank_file_path, "rb"))
    logs = read_jsonl(log_path)
    
    # Snapshot the entire first-stage log to a new JSONL
    root_path = '/'.join(log_path.split('/')[:-1])
    first_stage_json = root_path + '/first_stage_log.jsonl'
    second_stage_json = root_path + '/second_stage_log.jsonl'
    write_jsonl(first_stage_json, logs, append=False)
    print(f"[info] First-stage log saved to: {first_stage_json} (n={len(logs)})")
    
    print(logs[0].keys())
    # Filter out items that have stage2=True (keep only first-pass failures)
    failed_problems = [rec for rec in logs 
                    if not rec.get("is_solved", False) 
                    and not rec.get("stage2", False)]
    print(f"number of failed problems: {len(failed_problems)}")

    # Load the complete logs and maintain a copy for updates
    logs_copy = read_jsonl(log_path)

    num_items = len(failed_problems)
    num_success = 0

    for i, item in enumerate(failed_problems):
        if 'stage2' in item: 
            print (f"skip {i}, stage2 exists")
            continue
        try:
            cur_pass = 0
            is_solved = False
            diverse_reflections = []
            implementations = []
            test_feedback = []
            all_levels_reflections_scores = []
            all_levels_implementations = []
            cur_func_impl = None
            lst = list(range(8))
            random.shuffle(lst)

            start_time = time()

            while cur_pass < pass_at_k and not is_solved:
                # Retrieve similar positive trajectories based on problem embedding
                curr_emb = get_openai_embedding([_math_prompt_string(item["problem"])])
                top_k_indices, _ = get_top_k_closest(
                    memory_bank["positive_trajectories"],
                    curr_emb[:, None],
                    k=1,
                )
                closest = [memory_bank["positive_trajectories"][idx] for idx in top_k_indices] if len(top_k_indices) else []

                # First attempt with memory-augmented problem (few-shot exemplars inline)
                augmented_problem = (
                    _build_augmented_problem_from_examples(closest, item["problem"])
                    if closest else item["problem"]
                )

                # Generate initial insights for second pass
                if use_parsing:
                    if Math_agent is not None:
                        refined_insights = Math_agent.generate(item["problem"], temperature=0.1)
                    else:
                        # Use insights from closest trajectory if available
                        if closest and "mistake_insights" in closest[0]:
                            refined_insights = closest[0]["mistake_insights"]
                        else:
                            # Fallback to original approach if no insights in memory
                            if 'mistake_insights' in item:
                                refined_insights = item['mistake_insights']
                            else:
                                refined_insights = ""

                cur_func_impl = gen.func_impl(
                    augmented_problem,
                    model,
                    "simple",
                    temperature=1.0,
                    mistake_insights=refined_insights
                )
                implementations.append(cur_func_impl)
                assert isinstance(cur_func_impl, str)

                is_passing = exe.evaluate(cur_func_impl, item["answer"], timeout=10)
                test_feedback.append("Correct answer" if is_passing else "Incorrect answer")

                print(gpt_usage(backend=model_name))

                if is_passing:
                    is_solved = True
                    num_success += 1
                    item["solution"] = cur_func_impl
                    break

                # Reflexion iterations with reflection-conditioned retrieval
                cur_iter = 0
                cur_feedback = "Incorrect answer"
                while cur_iter < max_iters:
                    # Generate new insights
                    if Math_agent is not None:
                        refined_insights = Math_agent.generate(item["problem"], temperature=1.0)
                    else:
                        new_insight_file = filter_pitfalls_high_temp(insight_json_file[i])
                        N = len(new_insight_file['pitfalls_high_temp'])
                        refined_insights = new_insight_file['pitfalls_high_temp'][lst[cur_iter]%N]

                    div_reflections = gen.self_reflection_diverse_parametric(
                        item["problem"], cur_func_impl, cur_feedback, model, diverse_reflections, refined_insights
                    ).split('\n')

                    div_reflections = [ref for ref in div_reflections if len(ref) > 7]
                    diverse_reflections += div_reflections
                    cur_func_impl_copy = deepcopy(cur_func_impl)

                    temp_solutions = []
                    reflections_scores = []
                    div_reflections_feedbacks = []

                    ref_id = 0
                    pbar = tqdm(total=len(div_reflections))
                    while ref_id < min(len(div_reflections), 2):
                        del exe
                        exe = executor_factory(lang='math', is_leet=is_leetcode)

                        reflection = div_reflections[ref_id]
                        print(f"Attempting reflection-{ref_id} (second pass):")
                        pprint(reflection)
                        print()

                        # reflection-conditioned retrieval
                        filtered_trajs = [
                            traj for traj in memory_bank["positive_trajectories"]
                            if "refection_embedding" in traj.keys()
                        ]
                        if len(filtered_trajs) > 0:
                            ref_emb = get_openai_embedding([reflection])
                            top_idx, _ = get_top_k_closest(
                                filtered_trajs,
                                ref_emb[:, None],
                                k=1,
                                similarity_axis="refection_embedding",
                            )
                            closest_ref_traj = filtered_trajs[top_idx[0]] if len(top_idx) else None
                        else:
                            # fallback to prompt similarity
                            top_idx, _ = get_top_k_closest(
                                memory_bank["positive_trajectories"],
                                curr_emb[:, None],
                                k=1,
                            )
                            closest_ref_traj = (
                                memory_bank["positive_trajectories"][top_idx[0]] if len(top_idx) else None
                            )

                        few_shot_reflexion_block = (
                            _compose_reflexion_few_shot_from_trajectory(closest_ref_traj)
                            if closest_ref_traj is not None
                            else ""
                        )

                        # Compose self-reflection by appending a small few-shot reflexion example
                        composed_self_reflection = (
                            reflection
                            + (
                                ("\n\n" + few_shot_reflexion_block)
                                if len(few_shot_reflexion_block) > 0
                                else ""
                            )
                        )

                        new_func_impl = gen.func_impl(
                            augmented_problem,
                            model,
                            "reflexion",
                            prev_answers=cur_func_impl_copy,
                            feedback=cur_feedback,
                            self_reflection=composed_self_reflection,
                            fewshot_example=few_shot_reflexion_block,
                            temperature=1.0,
                            mistake_insights=refined_insights
                        )

                        try:
                            assert isinstance(new_func_impl, str)
                        except Exception:
                            print("skipping solution generation due to invalid type.")
                            ref_id += 1
                            continue

                        cur_func_impl = new_func_impl
                        temp_solutions.append(cur_func_impl)

                        is_passing = exe.evaluate(cur_func_impl, item["answer"], timeout=5)
                        fb = "Correct answer" if is_passing else "Incorrect answer"
                        test_feedback.append(fb)
                        div_reflections_feedbacks.append(fb)

                        reflections_scores.append((1.0 if is_passing else 0.0) + 1e-8)

                        ref_id += 1
                        pbar.update(1)

                        if is_passing or cur_iter == max_iters - 1:
                            if is_passing:
                                item["solution"] = cur_func_impl
                                is_solved = True
                                num_success += 1
                            break

                    pbar.close()

                    all_levels_reflections_scores.append(reflections_scores)
                    all_levels_implementations.append(temp_solutions)

                    # memory bank update at end-of-iter
                    if (cur_iter == max_iters - 1) or is_passing:
                        sampled_idx = (
                            random.choices(
                                range(len(temp_solutions)),
                                weights=reflections_scores,
                                k=1,
                            )[0]
                            if temp_solutions
                            else None
                        )
                        chosen_fb = (
                            div_reflections_feedbacks[sampled_idx]
                            if sampled_idx is not None
                            else cur_feedback
                        )
                        chosen_reflection = (
                            div_reflections[sampled_idx]
                            if (sampled_idx is not None and sampled_idx < len(div_reflections))
                            else (div_reflections[-1] if div_reflections else "")
                        )
                        chosen_solution = (
                            temp_solutions[sampled_idx] if sampled_idx is not None else cur_func_impl
                        )

                        trajectory = {
                            "problem": item["problem"],
                            "gen_solution": chosen_solution,
                            "reflection": chosen_reflection,
                            "test_feedback": chosen_fb,
                            "prev_solution": cur_func_impl_copy,
                            "prompt_embedding": get_openai_embedding(
                                [_math_prompt_string(item["problem"])]
                            ),
                            "refection_embedding": get_openai_embedding([chosen_reflection]),
                            "mistake_insights": refined_insights
                        }

                        if is_passing:
                            memory_bank["positive_trajectories"].append(trajectory)
                        else:
                            memory_bank["negative_trajectories"].append(trajectory)

                    if is_solved:
                        break

                    if temp_solutions:
                        sampled_impl_idx = random.choices(
                            range(len(temp_solutions)),
                            weights=reflections_scores,
                            k=1,
                        )[0]
                        cur_func_impl = temp_solutions[sampled_impl_idx]
                        cur_feedback = div_reflections_feedbacks[sampled_impl_idx]
                    cur_iter += 1
                cur_pass += 1

        except Exception as e:
            print("Exception in second pass example:", e)
            continue

        llm_cost = gpt_usage(backend=model_name)
        print(llm_cost)

        # Find the corresponding item in logs_copy using unique_id as key
        for log_item in logs_copy:
            if log_item.get('unique_id') == item.get('unique_id'):
                # Update runtime and stage2 flag
                log_item["runtime"] = time() - start_time
                log_item['stage2'] = True
                
                # Update is_solved (may change or not)
                log_item["is_solved"] = is_solved
                
                # Add up the costs
                log_item["cost"] = log_item.get("cost", 0) + llm_cost["cost"]
                log_item["completion_tokens"] = log_item.get("completion_tokens", 0) + llm_cost["completion_tokens"]
                log_item["prompt_tokens"] = log_item.get("prompt_tokens", 0) + llm_cost["prompt_tokens"]
                
                # Concatenate the lists
                log_item["reflections"] = log_item.get("reflections", []) + diverse_reflections
                log_item["implementations"] = log_item.get("implementations", []) + implementations
                log_item["test_feedback"] = log_item.get("test_feedback", []) + test_feedback
                log_item["solution"] = cur_func_impl
                
                # Add new fields specific to this version
                log_item["all_levels_reflections_scores"] = log_item.get("all_levels_reflections_scores", []) + all_levels_reflections_scores
                log_item["all_levels_implementations"] = log_item.get("all_levels_implementations", []) + all_levels_implementations
                
                break

        # Write the updated logs to second_stage_json after each iteration
        write_jsonl(second_stage_json, logs_copy, append=False)

        print_v(f"second pass: completed {i+1}/{num_items}: acc = {round(num_success/(i+1), 4)}")

    print(colored(gpt_usage(backend=model_name), 'blue'))
    