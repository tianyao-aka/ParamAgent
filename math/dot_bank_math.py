# dot_bank_math.py

import random
import pickle as pkl
from copy import deepcopy
from pprint import pprint
from tqdm import tqdm
from termcolor import colored
from time import time

from utils import (
    enumerate_resume,
    make_printv,
    write_jsonl,
    resume_success_count,read_jsonl
)
from executors import executor_factory
from generators import generator_factory, model_factory
from typing import List

from gpt_usage import gpt_usage

# memory bank imports
from memory_utils import (
    get_openai_embedding,
    get_top_k_closest,
)

import os


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


def run_dot_bank(
    dataset: List[dict],
    model_name: str,
    language: str,  # kept for signature parity; not used for math
    max_iters: int,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
    is_leetcode: bool = False,  # unused for math; kept for API compatibility
    use_parsing: bool = False,  # unused; compatibility with dot_math
    device: str = "cuda:0",     # unused here; embeddings use OpenAI endpoints inside memory_utils
    **kargs,
) -> None:
    """
    DoT-Bank for math reasoning (two-pass, memory-augmented).

    Each dataset item must contain:
      - 'problem': str
      - 'answer' : str (gold/reference)

    This adapts the code-generation DoT-Bank logic:
      - First pass attempts solution + reflexion; logs solved items and stores trajectories.
      - Unsolved items are saved to failed list for a second pass.
      - Second pass injects nearest solved exemplars and uses reflection-conditioned retrieval.

    Outputs:
      - Appends per-item results to `log_path` (jsonl).
      - Saves/updates `mem_bank.pkl` and `failed_probs.pkl` next to the log.
    """
    # Executors/generators/models for math
    exe = executor_factory(lang="math", is_leet=is_leetcode)
    gen = generator_factory(lang="math")
    model = model_factory(model_name)

    print_v = make_printv(verbose)

    num_items = len(dataset)
    num_success = resume_success_count(dataset)

    print("Running DoT-Bank for Math")

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
    
    
    primary_key = "unique_id"
    # --------------------------
    # First Pass
    # --------------------------
    for i, item in enumerate_resume(dataset, log_path):
        if skip_first: break
        cur_pass = 0
        is_solved = False
        diverse_reflections = []
        implementations = []  # attempted solutions (strings)
        test_feedback = []    # "Correct answer" or "Incorrect answer"
        all_levels_reflections_scores = []
        all_levels_implementations = []

        cur_solution = None  # the current best/last solution
        start_time = time()

        try:
            while cur_pass < pass_at_k and not is_solved:
                # First attempt (simple)
                fail_cnt = 0
                while cur_solution is None:
                    cur_solution = gen.func_impl(
                        item["problem"], model, "simple", temperature=0.2
                    )
                    fail_cnt += 1
                    if fail_cnt > 1:
                        break

                implementations.append(cur_solution)
                assert isinstance(cur_solution, str)

                # Evaluate against gold
                is_passing = exe.evaluate(cur_solution, item["answer"], timeout=10)
                test_feedback.append("Correct answer" if is_passing else "Incorrect answer")

                print(gpt_usage(backend=model_name))

                # If solved, store positive trajectory & exit early
                if is_passing:
                    trajectory = {
                        "problem": item["problem"],
                        "gen_solution": cur_solution,
                        "prompt_embedding": get_openai_embedding(
                            [_math_prompt_string(item["problem"])]
                        ),
                    }
                    memory_bank["positive_trajectories"].append(trajectory)

                    is_solved = True
                    num_success += 1
                    item["solution"] = cur_solution
                    break

                # Self-improvement with diverse reflections (math-style)
                cur_iter = 1
                cur_feedback = "Incorrect answer"
                while cur_iter < max_iters:
                    # produce multiple reflections (split on '\n' like dot_math)
                    div_reflections = gen.self_reflection_diverse(
                        item["problem"],
                        cur_solution,
                        cur_feedback,
                        model,
                        diverse_reflections,
                    ).split("\n")

                    # filter out short reflections
                    div_reflections = [ref for ref in div_reflections if len(ref) > 7]

                    diverse_reflections += div_reflections
                    cur_solution_copy = deepcopy(cur_solution)

                    temp_solutions = []
                    reflections_scores = []
                    div_reflections_feedbacks = []

                    ref_id = 0
                    pbar = tqdm(total=len(div_reflections))
                    # try up to 3 diverse reflections per iteration
                    while ref_id < min(len(div_reflections), 2):
                        del exe
                        exe = executor_factory(lang="math", is_leet=is_leetcode)

                        reflection = div_reflections[ref_id]
                        print(f"Attempting reflection-{ref_id}:")
                        pprint(reflection)
                        print()

                        new_solution = None
                        fail_cnt = 0
                        while new_solution is None:
                            new_solution = gen.func_impl(
                                item["problem"],
                                model,
                                "reflexion",
                                prev_answers=cur_solution_copy,
                                feedback=cur_feedback,
                                self_reflection=reflection,
                                temperature=0.2,
                            )
                            fail_cnt += 1
                            if fail_cnt > 1:
                                break

                        cur_solution = new_solution

                        try:
                            assert isinstance(cur_solution, str)
                        except Exception:
                            print("skipping solution generation due to invalid type.")
                            ref_id += 1
                            continue

                        temp_solutions.append(cur_solution)

                        is_passing = exe.evaluate(cur_solution, item["answer"], timeout=5)
                        fb = "Correct answer" if is_passing else "Incorrect answer"
                        test_feedback.append(fb)
                        div_reflections_feedbacks.append(fb)

                        # binary success score for weighted sampling
                        reflections_scores.append((1.0 if is_passing else 0.0) + 1e-8)

                        ref_id += 1
                        pbar.update(1)

                        if is_passing or cur_iter == max_iters - 1:
                            if is_passing:
                                item["solution"] = cur_solution
                                is_solved = True
                                num_success += 1
                            break

                    pbar.close()

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
                            temp_solutions[sampled_idx] if sampled_idx is not None else cur_solution
                        )

                        trajectory = {
                            "problem": item["problem"],
                            "gen_solution": chosen_solution,
                            "reflection": chosen_reflection,
                            "test_feedback": chosen_fb,
                            "prev_solution": cur_solution_copy,
                            "prompt_embedding": get_openai_embedding(
                                [_math_prompt_string(item["problem"])]
                            ),
                            # keep compatibility with 'refection_embedding' key
                            "refection_embedding": get_openai_embedding([chosen_reflection]),
                        }

                        if is_passing:
                            memory_bank["positive_trajectories"].append(trajectory)
                        else:
                            memory_bank["negative_trajectories"].append(trajectory)

                    if is_solved:
                        break

                    # sample likely solution for next iteration
                    if temp_solutions:
                        sampled_impl_idx = random.choices(
                            range(len(temp_solutions)),
                            weights=reflections_scores,
                            k=1,
                        )[0]
                        cur_solution = temp_solutions[sampled_impl_idx]
                        cur_feedback = div_reflections_feedbacks[sampled_impl_idx]

                    cur_iter += 1

                cur_pass += 1

        except Exception as e:
            print("Exception in first pass example:", e)
            continue

        # persist usage + per-item logs / memory bank
        end_time = time()
        llm_cost = gpt_usage(backend=model_name)
        print(llm_cost)


        item["runtime"] = end_time - start_time
        item["is_solved"] = is_solved
        item["reflections"] = diverse_reflections
        item["implementations"] = implementations
        item["test_feedback"] = test_feedback
        item["solution"] = cur_solution
        item["all_levels_reflections_scores"] = all_levels_reflections_scores
        item["all_levels_implementations"] = all_levels_implementations
        item["cost"] = llm_cost["cost"]
        item["completion_tokens"] = llm_cost["completion_tokens"]
        item["prompt_tokens"] = llm_cost["prompt_tokens"]
        write_jsonl(log_path, [item], append=True)

        print_v(f"completed {i+1}/{num_items}: acc = {round(num_success/(i+1), 4)}")
        if not is_solved:
            failed_problems.append(item)

        # write memory bank / failed list to disk after each item
        with open(mem_bank_file_path, "wb") as f:
            pkl.dump(memory_bank, f)
        with open(failed_probs_path, "wb") as f:
            pkl.dump(failed_problems, f)

    print("Finished first pass")

    
    memory_bank = pkl.load(open(mem_bank_file_path, 'rb'))
    logs = read_jsonl(log_path)
    
    # Snapshot the entire first-stage log to a new JSONL
    root_path = '/'.join(log_path.split('/')[:-1])
    first_stage_json = root_path + '/first_stage_log.jsonl'
    second_stage_json = root_path + '/second_stage_log.jsonl'
    write_jsonl(first_stage_json, logs, append=False, key=None, stage2=False)
    print(f"[info] First-stage log saved to: {first_stage_json} (n={len(logs)})")
    
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
            cur_solution = None

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

                cur_solution = gen.func_impl(
                    augmented_problem,
                    model,
                    "simple",
                    temperature=1.0,
                )
                implementations.append(cur_solution)
                assert isinstance(cur_solution, str)

                is_passing = exe.evaluate(cur_solution, item["answer"], timeout=10)
                test_feedback.append("Correct answer" if is_passing else "Incorrect answer")

                print(gpt_usage(backend=model_name))

                if is_passing:
                    is_solved = True
                    num_success += 1
                    item["solution"] = cur_solution
                    break

                # Reflexion iterations with reflection-conditioned retrieval
                cur_iter = 1
                cur_feedback = "Incorrect answer"
                while cur_iter < max_iters:
                    div_reflections = gen.self_reflection_diverse(
                        item["problem"],
                        cur_solution,
                        cur_feedback,
                        model,
                        diverse_reflections,
                    ).split("\n")

                    div_reflections = [ref for ref in div_reflections if len(ref) > 7]
                    diverse_reflections += div_reflections
                    cur_solution_copy = deepcopy(cur_solution)

                    temp_solutions = []
                    reflections_scores = []
                    div_reflections_feedbacks = []

                    ref_id = 0
                    pbar = tqdm(total=len(div_reflections))
                    while ref_id < min(len(div_reflections), 2):
                        del exe
                        exe = executor_factory(lang="math", is_leet=is_leetcode)

                        reflection = div_reflections[ref_id]
                        print(f"Attempting reflection-{ref_id} (second pass):")
                        pprint(reflection)
                        print()

                        # reflection-conditioned retrieval (prefer 'refection_embedding' for compatibility)
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

                        new_solution = gen.func_impl(
                            augmented_problem,
                            model,
                            "reflexion",
                            prev_answers=cur_solution_copy,
                            feedback=cur_feedback,
                            self_reflection=composed_self_reflection,
                            fewshot_example  = few_shot_reflexion_block, 
                            temperature=1.0,
                        )

                        try:
                            assert isinstance(new_solution, str)
                        except Exception:
                            print("skipping solution generation due to invalid type.")
                            continue

                        cur_solution = new_solution
                        temp_solutions.append(cur_solution)

                        is_passing = exe.evaluate(cur_solution, item["answer"], timeout=5)
                        fb = "Correct answer" if is_passing else "Incorrect answer"
                        test_feedback.append(fb)
                        div_reflections_feedbacks.append(fb)

                        reflections_scores.append((1.0 if is_passing else 0.0) + 1e-8)

                        ref_id += 1
                        pbar.update(1)

                        if is_passing or cur_iter == max_iters - 1:
                            if is_passing:
                                item["solution"] = cur_solution
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
                            temp_solutions[sampled_idx] if sampled_idx is not None else cur_solution
                        )

                        trajectory = {
                            "problem": item["problem"],
                            "gen_solution": chosen_solution,
                            "reflection": chosen_reflection,
                            "test_feedback": chosen_fb,
                            "prev_solution": cur_solution_copy,
                            "prompt_embedding": get_openai_embedding(
                                [_math_prompt_string(item["problem"])]
                            ),
                            "refection_embedding": get_openai_embedding([chosen_reflection]),
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
                        cur_solution = temp_solutions[sampled_impl_idx]
                        cur_feedback = div_reflections_feedbacks[sampled_impl_idx]

                    cur_iter += 1

                cur_pass += 1

        except Exception as e:
            print("Exception in second pass example:", e)
            continue

        llm_cost = gpt_usage(backend=model_name)
        print(llm_cost)
        
        # Find the corresponding item in logs_copy using task_id as key
        for log_item in logs_copy:
            if log_item.get(primary_key) == item.get(primary_key):
                # Update runtime and stage2 flag
                log_item["runtime"] = time() - start_time
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
                log_item["solution"] = cur_solution
                
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