import random
from copy import deepcopy
from pprint import pprint
from tqdm import tqdm
from termcolor import colored
import pickle as pkl
import os
from typing import List, Any, Dict

from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count, read_jsonl
from executors import executor_factory
from generators import generator_factory, model_factory
from time import time
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

# memory bank imports
from memory_utils import (
    get_openai_embedding,
    get_top_k_closest,
)


def _programming_prompt_string(prompt: str) -> str:
    """
    Compose a compact text for embedding retrieval in programming.
    
    Args:
        prompt (str): The programming problem/function signature.
    
    Returns:
        str: Normalized text used for embedding-based retrieval.
    """
    return f"Problem:\n{prompt}"


def _build_augmented_prompt_from_examples(examples: list, current_prompt: str) -> str:
    """
    Build a few-shot augmented input by inlining similar solved programming exemplars,
    then appending the current problem.
    
    Args:
        examples (list): List of trajectories from memory bank.
        current_prompt (str): The current programming prompt.
    
    Returns:
        str: An augmented prompt string fed to the code generator.
    """
    header = "Below are similar solved programming examples. Learn their solution style.\n\n"
    blocks = []
    for i, ex in enumerate(examples, start=1):
        p = ex.get("prompt", "")
        s = ex.get("gen_solution", "")
        blocks.append(
            f"[Example {i}]\n"
            f"Problem:\n{p}\n\n"
            f"Solution:\n{s}\n\n"
        )
    
    trailer = (
        "Now solve the NEW problem. Provide clean, efficient code.\n\n"
        "[CURRENT PROBLEM]\n"
        f"{current_prompt}"
    )
    
    return header + "".join(blocks) + trailer


def _compose_programming_reflexion_few_shot(traj: dict) -> str:
    """
    Compose a small 'few-shot' style reflexion hint from a past positive programming trajectory.
    
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
    inner_iter: int = 5,
    dataset_type: str = 'humaneval',
    fix_stage1_indices: bool = False,
) -> None:
    exe = executor_factory(language, is_leet=is_leetcode)
    gen = generator_factory(language)
    model = model_factory(model_name)

    print_v = make_printv(verbose)

    # Determine primary key based on dataset type (FIXED: was hardcoded as 'entry_point')
    if dataset_type == 'livecodebench':
        primary_key = "question_id"
    elif "task_id" in dataset[0].keys():
        primary_key = "task_id"
    else:
        primary_key = "entry_point"  # fallback for other datasets

    print(f"Running DoT with parametric knowledge and memory bank (primary_key: {primary_key})")
    num_items = len(dataset)
    num_success = resume_success_count(dataset)
    
    # Init memory bank related file paths
    root_path = "/".join(log_path.split("/")[:-1])
    mem_bank_file_path = os.path.join(root_path, "mem_bank.pkl")
    failed_probs_path = os.path.join(root_path, "failed_probs.pkl")

    # Check if memory-bank already exists
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

    # Check if second stage already exists
    first_stage_json = root_path + '/first_stage_log.jsonl'
    second_stage_json = root_path + '/second_stage_log.jsonl'

    # If flag enabled and first_stage_log.jsonl exists, fix indices and skip Stage 1
    if fix_stage1_indices and os.path.exists(first_stage_json) and not os.path.exists(second_stage_json):
        print("[INFO] Fixing original_index in first_stage_log.jsonl...")

        # Load existing Stage 1 results
        stage1_logs = read_jsonl(first_stage_json)

        # Create mapping from dataset item to its index
        # Try multiple keys for matching: question_id, question_title, entry_point, prompt
        dataset_map = {}
        for idx, item in enumerate(dataset):
            # For LiveCodeBench
            if 'question_id' in item:
                dataset_map[item['question_id']] = idx
            if 'question_title' in item:
                dataset_map[item['question_title']] = idx
            # For HumanEval
            if 'entry_point' in item:
                dataset_map[item['entry_point']] = idx
            # Fallback: use prompt hash
            if 'prompt' in item:
                dataset_map[item['prompt']] = idx

        # Add original_index to each entry
        fixed_count = 0
        for log_entry in stage1_logs:
            # Try to match by multiple keys
            matched_idx = None
            for key in ['question_id', 'question_title', 'entry_point', 'prompt']:
                if key in log_entry and log_entry[key] in dataset_map:
                    matched_idx = dataset_map[log_entry[key]]
                    break

            if matched_idx is not None:
                log_entry['original_index'] = matched_idx
                fixed_count += 1
            else:
                print(f"WARNING: Could not match entry with keys: {list(log_entry.keys())[:5]}")

        # Save fixed first_stage_log.jsonl
        write_jsonl(first_stage_json, stage1_logs, append=False)
        print(f"[INFO] Fixed {fixed_count}/{len(stage1_logs)} entries with original_index")

        # Copy to main log_path so Stage 2 can read it
        write_jsonl(log_path, stage1_logs, append=False)

        # Skip Stage 1
        skip_first = True
        print(f"[INFO] Loaded {len(stage1_logs)} items from first_stage_log.jsonl, skipping Stage 1")
    elif os.path.exists(second_stage_json):
        skip_first = True
    else:
        skip_first = False

    # --------------------------
    # First Pass
    # --------------------------
    for i, item in enumerate_resume(dataset, log_path):
        if skip_first: break
        print(i)
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
                        print("using visible test cases")
                        tests_i = visible_tests[identifier]['given_tests']
                    else:
                        print("generating synthetic test cases")
                        tests_i = gen.internal_tests(prompt, model, 1)

                # Use self-reflection to iteratively improve
                init_iter = 0
                lst = list(range(8))
                random.shuffle(lst)
                
                while init_iter < inner_iter:
                    # Get self-reflection
                    # LiveCodeBench uses 'pitfalls' key, HumanEval/MBPP use 'pitfall'
                    pitfall_key = 'pitfalls' if dataset_type == 'livecodebench' else 'pitfall'
                    if init_iter > 0:
                        if 'high_temp_pitfall' in mistake_json_file[i]:
                            refined_insights = mistake_json_file[i]['high_temp_pitfall'][lst[init_iter]]
                        elif pitfall_agent is not None:
                            refined_insights = pitfall_agent.generate(prompt, temperature=1.0)
                        else:
                            refined_insights = ""
                    else:
                        if 'high_temp_pitfall' in mistake_json_file[i]:
                            refined_insights = mistake_json_file[i][pitfall_key][lst[init_iter]] if isinstance(mistake_json_file[i][pitfall_key], list) else mistake_json_file[i][pitfall_key]
                        elif pitfall_agent is not None:
                            refined_insights = pitfall_agent.generate(prompt, temperature=0.1)
                        else:
                            refined_insights = ""

                    item['mistake_insights'] = refined_insights
                    new_func_impl = None
                    fail_cnt = 0
                    while new_func_impl is None:
                        new_func_impl = gen.func_impl(prompt, model, "simple",
                                                     mistake_insights=refined_insights, temperature=0.2)
                        fail_cnt += 1
                        if fail_cnt > 1:
                            break
                    cur_func_impl = new_func_impl
                    
                    implementations.append(cur_func_impl)
                    assert isinstance(cur_func_impl, str)

                    # Check if all internal unit tests pass
                    if dataset_type == 'livecodebench':
                        is_passing, cur_feedback = _evaluate_with_feedback_livecodebench(
                            exe, identifier, cur_func_impl, tests_i, timeout=20
                        )
                        test_feedback.append(cur_feedback)
                    else:
                        is_passing, cur_feedback, _ = exe.execute(cur_func_impl, tests_i)
                        test_feedback.append(cur_feedback)

                    # If solved, check if it passes the real tests, exit early
                    if is_passing or init_iter == inner_iter - 1:
                        if dataset_type == 'livecodebench':
                            is_passing = exe.evaluate_livecodebench(identifier, cur_func_impl, private_tests, timeout=20)
                        else:
                            is_passing = exe.evaluate(identifier, cur_func_impl, test_code, timeout=10)
                        if is_passing:
                            # Store positive trajectory
                            trajectory = {
                                "prompt": prompt,
                                "gen_solution": cur_func_impl,
                                "prompt_embedding": get_openai_embedding(
                                    [_programming_prompt_string(prompt)]
                                ),
                                "mistake_insights": refined_insights,
                                "entry_point": identifier
                            }
                            memory_bank["positive_trajectories"].append(trajectory)

                            item["solution"] = cur_func_impl
                            is_solved = True
                            num_success += 1
                        break

                    init_iter += 1
                    
                if is_solved:
                    break

                # Conditional sampling on prior reflections to promote diversity
                cur_iter = 0
                while cur_iter < max_iters:
                    # Get multiple diverse reflections
                    if 'high_temp_pitfall' in mistake_json_file[i]:
                        refined_insights = mistake_json_file[i]['high_temp_pitfall'][lst[cur_iter]]
                    elif pitfall_agent is not None:
                        refined_insights = pitfall_agent.generate(prompt, temperature=1.0)
                    else:
                        refined_insights = ""

                    div_reflections = gen.self_reflection_diverse_oneshot_parametric(
                        cur_func_impl, cur_feedback, model, diverse_reflections, refined_insights
                    ).split("\n\n")
                    
                    # Filter out reflections if they are less than few characters
                    div_reflections = [ref for ref in div_reflections if len(ref) > 10]
                    diverse_reflections += div_reflections
                    
                    cur_func_impl_copy = deepcopy(cur_func_impl)
                    
                    temp_implementations = []
                    reflections_scores = []
                    div_reflections_feedbacks = []
                    
                    ref_id = 0
                    pbar = tqdm(total=min(len(div_reflections), 2))
                    while ref_id < min(len(div_reflections), 2):
                        # Re-init executor
                        del exe
                        exe = executor_factory(language, is_leet=is_leetcode)

                        reflection = div_reflections[ref_id]
                        print(f"Attempting reflection-{ref_id}:")
                        pprint(reflection)
                        print()
                        
                        # Apply self-reflection in the next attempt
                        new_func_impl = None
                        fail_cnt = 0
                        while new_func_impl is None:
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
                            fail_cnt += 1
                            if fail_cnt > 1:
                                break
                        cur_func_impl = new_func_impl

                        try:
                            assert isinstance(cur_func_impl, str)
                        except:
                            print("regenerating func impl.")
                            ref_id += 1
                            pbar.update(1)
                            continue

                        # Will be used later to sample a probable solution
                        temp_implementations.append(cur_func_impl)

                        # Check if all internal unit tests pass
                        if dataset_type == 'livecodebench':
                            is_passing, cur_feedback_new = _evaluate_with_feedback_livecodebench(
                                exe, identifier, cur_func_impl, tests_i, timeout=30
                            )
                            test_feedback.append(cur_feedback_new)
                            div_reflections_feedbacks.append(cur_feedback_new)
                            # For LiveCodeBench, count passing tests
                            reflections_scores.append(
                                cur_feedback_new.count("PASS") + 1e-8
                            )
                        else:
                            is_passing, cur_feedback_new, _ = exe.execute(cur_func_impl, tests_i)
                            test_feedback.append(cur_feedback_new)
                            div_reflections_feedbacks.append(cur_feedback_new)
                            # Measures total number of failed unit tests
                            reflections_scores.append(
                                (len(tests_i) - cur_feedback_new.split("Tests failed:")[1].count('assert')) + 1e-8
                            )

                        # Increment ref-id counter
                        ref_id += 1
                        pbar.update(1)

                        # If solved, check if it passes the real tests, exit early
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
                    
                    # Log reflection scores and given level implementations
                    all_levels_reflections_scores.append(reflections_scores)
                    all_levels_implementations.append(temp_implementations)
                    
                    # Memory bank update at end-of-iter or success
                    if (cur_iter == max_iters - 1) or is_passing:
                        if temp_implementations:
                            sampled_idx = random.choices(
                                range(len(temp_implementations)),
                                weights=reflections_scores,
                                k=1
                            )[0]
                            chosen_fb = div_reflections_feedbacks[sampled_idx]
                            chosen_reflection = (
                                div_reflections[sampled_idx] 
                                if sampled_idx < len(div_reflections) 
                                else div_reflections[-1]
                            )
                            chosen_solution = temp_implementations[sampled_idx]
                        else:
                            chosen_fb = cur_feedback
                            chosen_reflection = div_reflections[-1] if div_reflections else ""
                            chosen_solution = cur_func_impl

                        trajectory = {
                            "prompt": prompt,
                            "gen_solution": chosen_solution,
                            "reflection": chosen_reflection,
                            "test_feedback": chosen_fb,
                            "prev_solution": cur_func_impl_copy,
                            "prompt_embedding": get_openai_embedding(
                                [_programming_prompt_string(prompt)]
                            ),
                            "reflection_embedding": get_openai_embedding([chosen_reflection]),
                            "mistake_insights": refined_insights,
                            "entry_point": identifier
                        }

                        if is_passing:
                            memory_bank["positive_trajectories"].append(trajectory)
                        else:
                            memory_bank["negative_trajectories"].append(trajectory)
                    
                    if is_solved:
                        break
                    
                    # Sample likely implementation
                    if temp_implementations:
                        sampled_impl_idx = random.choices(
                            range(len(temp_implementations)), 
                            weights=reflections_scores, 
                            k=1
                        )[0]
                        cur_func_impl = temp_implementations[sampled_impl_idx]
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
        item['original_index'] = i  # Store original dataset index for Stage 2
        write_jsonl(log_path, [item], append=True)

        print_v(f'completed {i+1}/{num_items}: acc = {round(num_success/(i+1), 2)}')
        
        if not is_solved:
            failed_problems.append(item)

        # Write memory bank / failed list to disk after each item
        with open(mem_bank_file_path, "wb") as f:
            pkl.dump(memory_bank, f)
        with open(failed_probs_path, "wb") as f:
            pkl.dump(failed_problems, f)
    print("Finished first pass")
    print(colored(gpt_usage(backend=model_name), 'blue'))
    # --------------------------
    # Second Pass (memory-augmented)
    # --------------------------
    memory_bank = pkl.load(open(mem_bank_file_path, "rb"))
    logs = read_jsonl(log_path)
    
    # Snapshot the entire first-stage log to a new JSONL
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
            print(f"skip {i}, stage2 exists")
            continue
        print(f"Second pass: item {i}")
        start_time = time()

        # Get original dataset index for accessing mistake_json_file
        original_idx = item.get('original_index', i)

        try:
            # Normalize fields based on dataset type
            if dataset_type == 'livecodebench':
                prompt = format_livecodebench_prompt(item, language="python", include_public_tests=True)
                identifier = item.get("question_title", item.get("question_id", f"problem_{i}"))
                public_tests = parse_public_tests(item)
                private_tests = parse_private_tests(item)
                test_code = None
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
            lst = list(range(8))
            random.shuffle(lst)

            while cur_pass < pass_at_k and not is_solved:
                # Generate test cases
                if dataset_type == 'livecodebench':
                    print("using public test cases for LiveCodeBench (second pass)")
                    tests_i = public_tests
                elif is_leetcode:
                    tests_i = item['visible_tests']
                else:
                    if visible_tests:
                        tests_i = visible_tests[identifier]['given_tests']
                    else:
                        tests_i = gen.internal_tests(prompt, model, 1)

                # Retrieve similar positive trajectories based on prompt embedding
                curr_emb = get_openai_embedding([_programming_prompt_string(prompt)])
                top_k_indices, _ = get_top_k_closest(
                    memory_bank["positive_trajectories"],
                    curr_emb[:, None],
                    k=1,
                )
                closest = [memory_bank["positive_trajectories"][idx] for idx in top_k_indices] if len(top_k_indices) else []

                # First attempt with memory-augmented prompt
                augmented_prompt = (
                    _build_augmented_prompt_from_examples(closest, prompt)
                    if closest else prompt
                )

                # Generate initial insights for second pass
                if use_mistakes:
                    if pitfall_agent is not None:
                        refined_insights = pitfall_agent.generate(prompt, temperature=0.1)
                    else:
                        # Use insights from closest trajectory if available
                        if closest and "mistake_insights" in closest[0]:
                            refined_insights = closest[0]["mistake_insights"]
                        else:
                            if 'mistake_insights' in item:
                                refined_insights = item['mistake_insights']
                            else:
                                refined_insights = ""
                else:
                    refined_insights = ""

                cur_func_impl = gen.func_impl(
                    augmented_prompt,
                    model,
                    "simple",
                    temperature=1.0,
                    mistake_insights=refined_insights
                )
                implementations.append(cur_func_impl)
                assert isinstance(cur_func_impl, str)

                if dataset_type == 'livecodebench':
                    is_passing, cur_feedback = _evaluate_with_feedback_livecodebench(
                        exe, identifier, cur_func_impl, tests_i, timeout=30
                    )
                    test_feedback.append(cur_feedback)
                else:
                    is_passing, cur_feedback, _ = exe.execute(cur_func_impl, tests_i)
                    test_feedback.append(cur_feedback)

                # Check real tests
                if is_passing:
                    if dataset_type == 'livecodebench':
                        is_passing = exe.evaluate_livecodebench(identifier, cur_func_impl, private_tests, timeout=30)
                    else:
                        is_passing = exe.evaluate(identifier, cur_func_impl, test_code, timeout=10)
                    if is_passing:
                        # Store positive trajectory
                        trajectory = {
                            "prompt": prompt,
                            "gen_solution": cur_func_impl,
                            "prompt_embedding": get_openai_embedding(
                                [_programming_prompt_string(prompt)]
                            ),
                            "mistake_insights": refined_insights,
                            "entry_point": identifier
                        }
                        memory_bank["positive_trajectories"].append(trajectory)

                        is_solved = True
                        num_success += 1
                        item["solution"] = cur_func_impl
                        break

                # Reflexion iterations with reflection-conditioned retrieval
                cur_iter = 0
                while cur_iter < max_iters:
                    # Generate new insights
                    if use_mistakes:
                        if pitfall_agent is not None:
                            refined_insights = pitfall_agent.generate(prompt, temperature=1.0)
                        else:
                            if (mistake_json_file and
                                original_idx < len(mistake_json_file) and
                                mistake_json_file[original_idx] is not None and
                                'high_temp_pitfall' in mistake_json_file[original_idx]):
                                refined_insights = mistake_json_file[original_idx]['high_temp_pitfall'][lst[cur_iter]]
                            else:
                                refined_insights = ""
                    else:
                        refined_insights = ""

                    div_reflections = gen.self_reflection_diverse_oneshot_parametric(
                        cur_func_impl, cur_feedback, model, diverse_reflections, refined_insights
                    ).split("\n\n")

                    div_reflections = [ref for ref in div_reflections if len(ref) > 10]
                    diverse_reflections += div_reflections
                    cur_func_impl_copy = deepcopy(cur_func_impl)

                    temp_implementations = []
                    reflections_scores = []
                    div_reflections_feedbacks = []

                    ref_id = 0
                    pbar = tqdm(total=min(len(div_reflections), 2))
                    while ref_id < min(len(div_reflections), 2):
                        del exe
                        exe = executor_factory(language, is_leet=is_leetcode)

                        reflection = div_reflections[ref_id]
                        print(f"Attempting reflection-{ref_id} (second pass):")
                        pprint(reflection)
                        print()

                        # Reflection-conditioned retrieval
                        filtered_trajs = [
                            traj for traj in memory_bank["positive_trajectories"]
                            if "reflection_embedding" in traj.keys()
                        ]
                        
                        if len(filtered_trajs) > 0:
                            ref_emb = get_openai_embedding([reflection])
                            top_idx, _ = get_top_k_closest(
                                filtered_trajs,
                                ref_emb[:, None],
                                k=1,
                                similarity_axis="reflection_embedding",
                            )
                            closest_ref_traj = filtered_trajs[top_idx[0]] if len(top_idx) else None
                        else:
                            # Fallback to prompt similarity
                            top_idx, _ = get_top_k_closest(
                                memory_bank["positive_trajectories"],
                                curr_emb[:, None],
                                k=1,
                            )
                            closest_ref_traj = (
                                memory_bank["positive_trajectories"][top_idx[0]] if len(top_idx) else None
                            )

                        few_shot_reflexion_block = (
                            _compose_programming_reflexion_few_shot(closest_ref_traj)
                            if closest_ref_traj is not None
                            else ""
                        )

                        # Compose self-reflection
                        composed_self_reflection = (
                            reflection
                            + (("\n\n" + few_shot_reflexion_block) if len(few_shot_reflexion_block) > 0 else "")
                        )

                        new_func_impl = gen.func_impl(
                            func_sig=augmented_prompt,
                            model=model,
                            strategy="reflexion",
                            prev_func_impl=cur_func_impl_copy,
                            feedback=cur_feedback,
                            self_reflection=composed_self_reflection,
                            temperature=1.0,
                            ref_chat_instruction='dot',
                            mistake_insights=refined_insights,
                        )

                        try:
                            assert isinstance(new_func_impl, str)
                        except:
                            print("skipping solution generation due to invalid type.")
                            ref_id += 1
                            pbar.update(1)
                            continue

                        cur_func_impl = new_func_impl
                        temp_implementations.append(cur_func_impl)
                        implementations.append(cur_func_impl)

                        if dataset_type == 'livecodebench':
                            is_passing, cur_feedback_new = _evaluate_with_feedback_livecodebench(
                                exe, identifier, cur_func_impl, tests_i, timeout=30
                            )
                            test_feedback.append(cur_feedback_new)
                            div_reflections_feedbacks.append(cur_feedback_new)
                            reflections_scores.append(
                                cur_feedback_new.count("PASS") + 1e-8
                            )
                        else:
                            is_passing, cur_feedback_new, _ = exe.execute(cur_func_impl, tests_i)
                            test_feedback.append(cur_feedback_new)
                            div_reflections_feedbacks.append(cur_feedback_new)
                            reflections_scores.append(
                                (len(tests_i) - cur_feedback_new.split("Tests failed:")[1].count('assert')) + 1e-8
                            )

                        ref_id += 1
                        pbar.update(1)

                        if is_passing or cur_iter == max_iters - 1:
                            if is_passing:
                                if dataset_type == 'livecodebench':
                                    is_passing = exe.evaluate_livecodebench(identifier, cur_func_impl, private_tests, timeout=30)
                                else:
                                    is_passing = exe.evaluate(identifier, cur_func_impl, test_code, timeout=10)

                            # Update memory bank
                            if temp_implementations:
                                sampled_idx = random.choices(
                                    range(len(temp_implementations)),
                                    weights=reflections_scores,
                                    k=1
                                )[0]
                                chosen_fb = div_reflections_feedbacks[sampled_idx]
                                chosen_reflection = (
                                    div_reflections[sampled_idx]
                                    if sampled_idx < len(div_reflections)
                                    else div_reflections[-1]
                                )
                                chosen_solution = temp_implementations[sampled_idx]
                            else:
                                chosen_fb = cur_feedback
                                chosen_reflection = reflection
                                chosen_solution = cur_func_impl

                            trajectory = {
                                "prompt": prompt,
                                "gen_solution": chosen_solution,
                                "reflection": chosen_reflection,
                                "test_feedback": chosen_fb,
                                "prev_solution": cur_func_impl_copy,
                                "prompt_embedding": get_openai_embedding(
                                    [_programming_prompt_string(prompt)]
                                ),
                                "reflection_embedding": get_openai_embedding([chosen_reflection]),
                                "mistake_insights": refined_insights,
                                "entry_point": identifier
                            }

                            if is_passing:
                                memory_bank["positive_trajectories"].append(trajectory)
                                item["solution"] = cur_func_impl
                                is_solved = True
                                num_success += 1
                            else:
                                memory_bank["negative_trajectories"].append(trajectory)
                            break

                    pbar.close()
                    
                    all_levels_reflections_scores.append(reflections_scores)
                    all_levels_implementations.append(temp_implementations)

                    if is_solved:
                        break

                    if temp_implementations:
                        sampled_impl_idx = random.choices(
                            range(len(temp_implementations)),
                            weights=reflections_scores,
                            k=1
                        )[0]
                        cur_func_impl = temp_implementations[sampled_impl_idx]
                        cur_feedback = div_reflections_feedbacks[sampled_impl_idx]
                    
                    cur_iter += 1
                cur_pass += 1

        except Exception as e:
            import traceback
            print("Exception in second pass example:", e)
            print("Traceback:")
            traceback.print_exc()
            continue

        llm_cost = gpt_usage(backend=model_name)
        print(llm_cost)

        # Find the corresponding item in logs_copy using primary_key (FIXED: was using entry_point + prompt)
        matched = False
        for log_item in logs_copy:
            if log_item.get(primary_key) == item.get(primary_key):
                # Update runtime and stage2 flag
                log_item["runtime"] = time() - start_time
                log_item['stage2'] = True
                
                # Add defensive check: warn if overwriting a solved problem with unsolved
                if log_item.get('is_solved', False) and not is_solved:
                    print(f"WARNING: Stage 2 overwrote SOLVED problem {log_item.get(primary_key)} with UNSOLVED!")

                # Update is_solved
                log_item["is_solved"] = is_solved
                matched = True
                
                # Add up the costs
                log_item["cost"] = log_item.get("cost", 0) + llm_cost["cost"]
                log_item["completion_tokens"] = log_item.get("completion_tokens", 0) + llm_cost["completion_tokens"]
                log_item["prompt_tokens"] = log_item.get("prompt_tokens", 0) + llm_cost["prompt_tokens"]
                
                # Concatenate the lists
                log_item["diverse_reflections"] = log_item.get("diverse_reflections", []) + diverse_reflections
                log_item["implementations"] = log_item.get("implementations", []) + implementations
                log_item["test_feedback"] = log_item.get("test_feedback", []) + test_feedback
                log_item["solution"] = cur_func_impl
                log_item["all_levels_reflections_scores"] = log_item.get("all_levels_reflections_scores", []) + all_levels_reflections_scores
                log_item["all_levels_implementations"] = log_item.get("all_levels_implementations", []) + all_levels_implementations
                
                break

        # Add warning if matching failed
        if not matched:
            print(f"ERROR: Failed to match Stage 2 item with {primary_key}={item.get(primary_key)} in logs_copy!")

        # Write the updated logs to second_stage_json after each iteration
        write_jsonl(second_stage_json, logs_copy, append=False)
        
        # Update memory bank file
        with open(mem_bank_file_path, "wb") as f:
            pkl.dump(memory_bank, f)

        print_v(f"second pass: completed {i+1}/{num_items}: acc = {round(num_success/(i+1), 4)}")

    print(colored(gpt_usage(backend=model_name), 'blue'))