import random
import pickle as pkl
from copy import deepcopy
from pprint import pprint
from tqdm import tqdm
from termcolor import colored
from time import time

from utils import enumerate_resume_dotbank, \
                  make_printv, \
                  write_jsonl, \
                  resume_success_count,enumerate_resume,read_jsonl
from executors import executor_factory
from generators import generator_factory, model_factory
from typing import List

from gpt_usage import gpt_usage

#memory bank imports
from scipy.spatial import distance
from memory_utils import get_cohere_embedding, \
                         get_openai_embedding, \
                         get_top_k_closest, \
                         get_random_k_indices
import os

def run_dot_bank(
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
    
    print("Running DoT-Bank for QA")
    
    #init memory bank related file paths
    root_path = '/'.join(log_path.split('/')[:-1])
    mem_bank_file_path = root_path + '/mem_bank_qa.pkl'
    failed_probs_path = root_path + '/failed_probs_qa.pkl'
    
    # check if memory-bank already exists
    if os.path.exists(mem_bank_file_path):
        with open(mem_bank_file_path, 'rb') as f:
            memory_bank = pkl.load(f)
    else:
        # initialize memory bank
        memory_bank = {
            "positive_trajectories": [],  # correct answers
            "negative_trajectories": [],  # incorrect answers
        }
        
    if os.path.exists(failed_probs_path):
        with open(failed_probs_path, 'rb') as f:
            failed_problems = pkl.load(f)
    else:
        # store all problems that failed in the first pass
        failed_problems = []
    
    # Use question as primary key for QA tasks
    primary_key = "question"

    # Snapshot the entire first-stage log to a new JSONL
    root_path = '/'.join(log_path.split('/')[:-1])
    first_stage_json = root_path + '/first_stage_log.jsonl'
    second_stage_json = root_path + '/second_stage_log.jsonl'
    if os.path.exists(second_stage_json):
        skip_first = True
    else:
        skip_first = False
    # First Pass - Use regular enumerate_resume for QA (not dotbank version)
    for i, item in enumerate_resume(dataset, log_path):
        if skip_first: break
        cur_pass = 0
        is_solved = False
        diverse_reflections = []
        implementations = []
        test_feedback = []
        cur_func_impl = None
        
        cur_prob_passed = False
        
        try:
            start_time = time()
        
            while cur_pass < pass_at_k and not is_solved:
                
                # Generate initial answer
                fail_cnt = 0
                while cur_func_impl is None:
                    cur_func_impl = gen.func_impl(item["question"], item["context"], model, "simple", temperature=0.2)
                    fail_cnt += 1
                    if fail_cnt > 1:
                        break
                
                implementations.append(cur_func_impl)
                assert isinstance(cur_func_impl, str)
                
                # Check if answer is correct
                is_passing = exe.evaluate(cur_func_impl, item['answer'], timeout=5)
                test_feedback.append("Correct answer" if is_passing else "Incorrect answer")
                
                print(gpt_usage(backend=model_name))

                # if solved, exit early
                if is_passing:
                    
                    # populate memory bank if first attempt passed
                    trajectory = {
                        "question": item["question"],
                        "context": item["context"],
                        "gen_answer": cur_func_impl,
                        "golden_answer": item["answer"],
                        "question_embedding": get_openai_embedding([item["question"]]),
                        "context_embedding": get_openai_embedding([item["context"][:1000]]),  # limit context length for embedding
                    }
                
                    # update memory bank
                    cur_prob_passed = True
                    memory_bank['positive_trajectories'].append(trajectory)
                    
                    is_solved = is_passing
                    num_success += int(is_passing)
                    print(f"Solved: {is_solved}, Success count: {num_success}")
                    break

                # conditional sampling on prior reflections to promote diversity
                cur_iter = 1
                cur_feedback = "Incorrect answer"
                while cur_iter < max_iters:
                    
                    # one-shot sampling for multiple diverse reflections
                    div_reflections = gen.self_reflection_diverse(
                        item["question"], cur_func_impl, item["context"], cur_feedback, model, diverse_reflections).split('\n\n')
                    
                    # filter out reflections if they are less than few characters
                    div_reflections = [ref for ref in div_reflections if len(ref) > 10]
                    
                    diverse_reflections += div_reflections
                    cur_func_impl_copy = deepcopy(cur_func_impl)
                    
                    temp_implementations = []
                    reflections_scores = []
                    div_reflections_feedbacks = []
                    
                    ref_id = 0
                    pbar = tqdm(total=len(div_reflections))
                    while ref_id < min(len(div_reflections), 2):
                        
                        #re-init executor
                        del exe
                        exe = executor_factory(lang='QA', is_leet=is_leetcode)

                        reflection = div_reflections[ref_id]
                        print(f"Attempting reflection-{ref_id}:")
                        pprint(reflection)
                        print()
                        
                        # apply self-reflection in the next attempt
                        new_func_impl = None
                        fail_cnt = 0
                        while new_func_impl is None:
                            new_func_impl = gen.func_impl(
                                item["question"],
                                item["context"],
                                model,
                                "reflexion",
                                prev_answers=cur_func_impl_copy,
                                feedback=cur_feedback,
                                self_reflection=reflection,
                                temperature=0.2
                            )
                            fail_cnt += 1
                            if fail_cnt > 1:
                                break
                        cur_func_impl = new_func_impl

                        try:
                            assert isinstance(cur_func_impl, str)
                        except:                            
                            print("skipping answer impl.")
                            ref_id += 1
                            continue

                        # Will be used later to sample a probable solution
                        temp_implementations.append(cur_func_impl)
                    
                        # check if answer is correct
                        is_passing = exe.evaluate(cur_func_impl, item['answer'], timeout=5)
                        current_feedback = "Correct answer" if is_passing else "Incorrect answer"
                        test_feedback.append(current_feedback)
                        div_reflections_feedbacks.append(current_feedback)
                        
                        # Score based on correctness (1 for correct, small value for incorrect)
                        reflections_scores.append(1.0 if is_passing else 0.1)

                        # increment ref-id counter
                        ref_id += 1
                        pbar.update(1)

                        # if solved, check if it passes, exit early
                        if is_passing or cur_iter == max_iters - 1:
                            # setting based on correctness
                            cur_prob_passed = is_passing
                            
                            if is_passing:
                                item["solution"] = cur_func_impl
                                is_solved = True
                                num_success += int(is_passing)
                                
                            break
                    
                    pbar.close()

                    #sample likely implementation
                    print(reflections_scores)
                    if temp_implementations:
                        sampled_impl_idx = random.choices(range(len(temp_implementations)), weights=reflections_scores, k=1)[0]
                        cur_func_impl = temp_implementations[sampled_impl_idx]
                        
                        # set cur_feedback to the corresponding sampled div-reflection
                        cur_feedback = div_reflections_feedbacks[sampled_impl_idx]
                    
                    # populate memory bank
                    if cur_iter == max_iters - 1 or is_passing:
                        trajectory = {
                            "question": item["question"],
                            "context": item["context"],
                            "gen_answer": cur_func_impl,
                            "golden_answer": item["answer"],
                            "reflection": reflection,
                            "feedback": cur_feedback,
                            "prev_answer": cur_func_impl_copy,
                            "question_embedding": get_openai_embedding([item["question"]]),
                            "context_embedding": get_openai_embedding([item["context"][:1000]]),
                            "reflection_embedding": get_openai_embedding([reflection]),
                        }                           
                        if is_passing:
                            cur_prob_passed = True
                            memory_bank['positive_trajectories'].append(trajectory)
                        else:
                            memory_bank['negative_trajectories'].append(trajectory)
                    
                    if is_solved:                        
                        break
                    
                    cur_iter += 1
                cur_pass += 1
                
        except Exception as e:
            print(f"Error in first pass: {e}")
            continue
        
        end_time = time()
            
        llm_cost = gpt_usage(backend=model_name)
        print(llm_cost)
        
        # Always save the item to log file
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
        
        # Add to failed problems list if not solved
        if not cur_prob_passed:
            failed_problems.append(item)
            
        #write mem-bank to file
        with open(mem_bank_file_path, 'wb') as f:
            pkl.dump(memory_bank, f)
            
        #update failed_probs.pkl
        with open(failed_probs_path, 'wb') as f:
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

    print (logs[0].keys())
    # Filter out items that have stage2=True (keep only first-pass failures)
    failed_problems = [rec for rec in logs 
                    if not rec.get("is_solved", False) 
                    and not rec.get("stage2", False)]
    print (f"number of failed problems: {len(failed_problems)}")

    # Load the complete logs and maintain a copy for updates
    logs_copy = read_jsonl(log_path)
    
    # reset num_items and num_success for 2nd pass
    num_items = len(failed_problems)
    num_success = 0
    
    # Second pass - using memory bank to help solve failed problems
    for i, item in enumerate(failed_problems):
        if 'stage2' in item: 
            print (f"skip {i}, stage2 exists")
            continue
        try:
            start_time = time()
        
            cur_pass = 0
            is_solved = False
            diverse_reflections = []
            implementations = []
            test_feedback = []
            cur_func_impl = ""
            
            while cur_pass < pass_at_k and not is_solved:
                
                # inject similar problems trajectory into context
                curr_emb = get_openai_embedding([item['question']]) 
                        
                top_k_indices, cosine_similarities = get_top_k_closest(
                    memory_bank['positive_trajectories'], 
                    curr_emb[:, None], 
                    k=1,
                    similarity_axis='question_embedding'
                )

                closest_match = [memory_bank['positive_trajectories'][i] for i in top_k_indices]

                # first attempt with memory bank guidance
                cur_func_impl = gen.func_impl(
                    item["question"], 
                    item["context"], 
                    model, 
                    "simple",
                    temperature=0.2
                )
                
                implementations.append(cur_func_impl)
                assert isinstance(cur_func_impl, str)
                is_passing = exe.evaluate(cur_func_impl, item['answer'], timeout=5)
                test_feedback.append("Correct answer" if is_passing else "Incorrect answer")
                
                print(gpt_usage(backend=model_name))

                # if solved, exit early
                if is_passing:
                    is_solved = is_passing
                    num_success += int(is_passing)
                    print(f"Solved: {is_solved}, Success count: {num_success}")
                    break

                # conditional sampling on prior reflections to promote diversity
                cur_iter = 1
                cur_feedback = "Incorrect answer"
                while cur_iter < max_iters:
                    
                    # one-shot sampling for multiple diverse reflections
                    div_reflections = gen.self_reflection_diverse(
                        item["question"], cur_func_impl, item["context"], cur_feedback, model, diverse_reflections).split('\n\n')
                    
                    # filter out reflections if they are less than few characters
                    div_reflections = [ref for ref in div_reflections if len(ref) > 10]
                    
                    diverse_reflections += div_reflections
                    cur_func_impl_copy = deepcopy(cur_func_impl)
                    
                    temp_implementations = []
                    reflections_scores = []
                    div_reflections_feedbacks = []
                    
                    ref_id = 0
                    pbar = tqdm(total=len(div_reflections))
                    while ref_id < min(len(div_reflections), 2):
                        
                        #re-init executor
                        del exe
                        exe = executor_factory(lang='QA', is_leet=is_leetcode)

                        reflection = div_reflections[ref_id]
                        print(f"Attempting reflection-{ref_id}:")
                        pprint(reflection)
                        print()
                        
                        # inject similar problems trajectory into context based on similarity in reflection
                        curr_emb = get_openai_embedding([reflection])
                        filtered_trajectories = [traj for traj in memory_bank['positive_trajectories'] if "reflection_embedding" in traj.keys()]
                        
                        if len(filtered_trajectories):
                            top_k_indices, cosine_similarities = get_top_k_closest(
                                filtered_trajectories, 
                                curr_emb[:, None], 
                                k=1, 
                                similarity_axis="reflection_embedding"
                            )
                            closest_match = filtered_trajectories[top_k_indices[0]]
                        else:
                            top_k_indices, cosine_similarities = get_top_k_closest(
                                memory_bank['positive_trajectories'], 
                                curr_emb[:, None], 
                                k=1, 
                                similarity_axis="question_embedding"
                            )
                            if top_k_indices:
                                closest_match = memory_bank['positive_trajectories'][top_k_indices[0]]
                            else:
                                closest_match = None
                        
                        # apply self-reflection in the next attempt with memory bank examples
                        if closest_match and 'prev_answer' in closest_match:
                            QA_FEW_SHOT = f'''Example:
[Question]: {closest_match['question']}
[Previous Answer]: {closest_match['prev_answer']}
[Reflection]: {closest_match.get('reflection', 'Need to reconsider the approach')}
[Improved Answer]: {closest_match['gen_answer']}
'''
                        else:
                            QA_FEW_SHOT = None
                            
                        new_func_impl = None
                        fail_cnt = 0
                        while new_func_impl is None:
                            new_func_impl = gen.func_impl(
                                item["question"],
                                item["context"],
                                model,
                                "reflexion",
                                prev_answers=cur_func_impl_copy,
                                feedback=cur_feedback,
                                self_reflection=reflection,
                                temperature=0.2,
                                fewshot_example=QA_FEW_SHOT  # Pass the few-shot example if available
                            )
                            fail_cnt += 1
                            if fail_cnt > 1:
                                break
                        cur_func_impl = new_func_impl

                        try:
                            assert isinstance(cur_func_impl, str)
                        except:
                            print("skipping answer impl.")
                            ref_id += 1
                            continue

                        # Will be used later to sample a probable solution
                        temp_implementations.append(cur_func_impl)
                    
                        # check if answer is correct
                        is_passing = exe.evaluate(cur_func_impl, item['answer'], timeout=5)
                        current_feedback = "Correct answer" if is_passing else "Incorrect answer"
                        test_feedback.append(current_feedback)
                        div_reflections_feedbacks.append(current_feedback)
                        
                        # Score based on correctness
                        reflections_scores.append(1.0 if is_passing else 0.1)

                        # increment ref-id counter
                        ref_id += 1
                        pbar.update(1)

                        # if solved, check if it passes, exit early
                        if is_passing or cur_iter == max_iters - 1:  
                            if is_passing:
                                item["solution"] = cur_func_impl
                                is_solved = True
                                num_success += 1
                            break
                    
                    pbar.close()

                    #sample likely implementation
                    print(reflections_scores)
                    if temp_implementations:
                        sampled_impl_idx = random.choices(range(len(temp_implementations)), weights=reflections_scores, k=1)[0]
                        cur_func_impl = temp_implementations[sampled_impl_idx]
                        
                        # set cur_feedback to the corresponding sampled div-reflection
                        cur_feedback = div_reflections_feedbacks[sampled_impl_idx]
                    
                    # populate memory bank for second pass
                    if cur_iter == max_iters - 1:
                        trajectory = {
                            "question": item["question"],
                            "context": item["context"],
                            "gen_answer": cur_func_impl,
                            "golden_answer": item["answer"],
                            "reflection": reflection if 'reflection' in locals() else "",
                            "feedback": cur_feedback,
                            "question_embedding": get_openai_embedding([item["question"]]),
                            "context_embedding": get_openai_embedding([item["context"][:1000]]),
                            "reflection_embedding": get_openai_embedding([reflection]) if 'reflection' in locals() else None,
                        }                           
                        if is_passing:
                            memory_bank['positive_trajectories'].append(trajectory)
                        else:
                            memory_bank['negative_trajectories'].append(trajectory)
                    
                    if is_solved:
                        break
                    
                    cur_iter += 1
                cur_pass += 1
                
        except Exception as e:
            print(f"Error in second pass: {e}")
            continue
        
        end_time = time()
            
        llm_cost = gpt_usage(backend=model_name)
        print(llm_cost)
        
        # Find the corresponding item in logs_copy using the question as key
        for log_item in logs_copy:
            if log_item['question'] == item['question']:
                # Update runtime and stage2 flag
                log_item['runtime'] = end_time - start_time
                log_item['stage2'] = True
                
                # Update is_solved (may change or not)
                log_item["is_solved"] = is_solved
                
                # Add up the costs
                log_item['cost'] = log_item.get('cost', 0) + llm_cost['cost']
                log_item['completion_tokens'] = log_item.get('completion_tokens', 0) + llm_cost['completion_tokens']
                log_item['prompt_tokens'] = log_item.get('prompt_tokens', 0) + llm_cost['prompt_tokens']
                
                # Concatenate the lists
                log_item["reflections"] = log_item.get("reflections", []) + diverse_reflections
                log_item["implementations"] = log_item.get("implementations", []) + implementations
                log_item["test_feedback"] = log_item.get("test_feedback", []) + test_feedback
                log_item["solution"] = cur_func_impl
                
                break

        # Write the updated logs to second_stage_json after each iteration
        write_jsonl(second_stage_json, logs_copy, append=False, key=None)

        print_v(
            f'completed {i+1}/{num_items}: acc = {round(num_success/(i+1), 4)}')

        
    print(colored(gpt_usage(backend=model_name), 'blue'))