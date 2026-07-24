from termcolor import colored
from time import time
from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count
from executors import executor_factory
from generators import generator_factory, model_factory
import sys
from typing import List
from gpt_usage import gpt_usage
import json
import re


def parse_gpt_solution(solution_text: str) -> str:
    """
    Parse GPT solution to extract the reasoning and code.

    Args:
        solution_text: Raw solution text from high_temp_solution

    Returns:
        Cleaned solution text with reasoning and code
    """
    if not solution_text:
        return ""

    # Try to extract content between <Solution> tags
    solution_match = re.search(r'<Solution>(.*?)(?:</Solution>|$)', solution_text, re.DOTALL)
    if solution_match:
        content = solution_match.group(1).strip()
    else:
        # If no <Solution> tags, use the whole text
        content = solution_text

    # Extract code blocks if present
    code_blocks = re.findall(r'```python\n(.*?)```', content, re.DOTALL)

    # Clean up the content
    # Remove excessive tags and formatting
    content = re.sub(r'<[/]?(?:SYS|INST|s)>', '', content)
    content = re.sub(r'\[/?INST\]', '', content)
    content = re.sub(r'<<SYS>>', '', content)
    content = re.sub(r'<</SYS>>', '', content)

    # If we have code blocks, include them prominently
    if code_blocks:
        reasoning = re.sub(r'```python.*?```', '', content, flags=re.DOTALL).strip()
        code = code_blocks[0].strip()
        return f"{reasoning}\n\nCode:\n```python\n{code}\n```"

    return content.strip()


def load_gpt_solutions(jsonl_path: str) -> dict:
    """
    Load GPT solutions from humaneval_full_solutions.jsonl

    Returns:
        dict: Mapping from task_id to list of GPT solutions
    """
    gpt_solutions_map = {}
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                task_id = item.get('task_id')
                high_temp_solutions = item.get('high_temp_solution', [])
                if task_id and high_temp_solutions:
                    gpt_solutions_map[task_id] = high_temp_solutions
        print(f"Loaded GPT solutions for {len(gpt_solutions_map)} problems")
    except FileNotFoundError:
        print(f"Warning: GPT solutions file not found at {jsonl_path}")
    except Exception as e:
        print(f"Warning: Error loading GPT solutions: {e}")

    return gpt_solutions_map


def run_reflexion_gpt4o_mini(
    dataset: List[dict],
    model_name: str = "llama3_1_8b",
    language: str = "python",
    log_path: str = "reflexion_gpt4o_mini.jsonl",
    verbose: bool = True,
    gpt_solutions_path: str = "benchmarks/code_solutions/humaneval_full_solutions.jsonl",
    **kwargs
) -> None:
    """
    Run reflexion algorithm with GPT solution guidance for HumanEval.

    Phase 1: 4 attempts using different GPT solutions as guidance
    Phase 2: If all 4 fail, 3 more reflexion attempts with saved reflections

    Args:
        dataset: List of HumanEval problems
        model_name: Model to use (default: "llama3_1_8b")
        language: Programming language (default: "python")
        log_path: Path to save results
        verbose: Whether to print progress
        gpt_solutions_path: Path to GPT solutions file
    """
    exe = executor_factory(language, is_leet=False)
    gen = generator_factory(language)
    model = model_factory(model_name)
    print_v = make_printv(verbose)

    # Load GPT solutions
    gpt_solutions_map = load_gpt_solutions(gpt_solutions_path)

    num_items = len(dataset)
    num_success = resume_success_count(dataset)
    print(f"Running Reflexion with GPT-4o-mini approach using model: {model_name}")

    for i, item in enumerate_resume(dataset, log_path):
        try:
            # Extract problem information
            prompt = item["prompt"]
            identifier = item["entry_point"]
            test_code = item["test"]
            task_id = item.get("task_id", None)

            # Get GPT solutions for this problem
            gpt_solutions = gpt_solutions_map.get(task_id, [])
            if not gpt_solutions:
                print(f"Warning: No GPT solutions found for {task_id}")
                continue

            # Extract public tests (individual assertions)
            tests_i = [
                case.lstrip().replace('candidate', item['entry_point'])
                for case in item['test'].split('\n')[1:-1]
                if 'assert' in case
            ]

            # Initialize tracking variables
            is_solved = False
            phase_solved = None
            all_attempts = []
            reflections = []
            start_time = time()

            # ===== PHASE 1: 4 Guided Attempts =====
            print_v(f"\n{'='*60}")
            print_v(f"Problem {i+1}/{num_items}: {task_id}")
            print_v(f"{'='*60}")
            print_v("PHASE 1: 4 guided attempts with GPT solutions")

            for attempt_num in range(4):
                print_v(f"\n--- Attempt {attempt_num + 1}/4 ---")

                # Select GPT solution for this attempt
                gpt_solution_idx = min(attempt_num, len(gpt_solutions) - 1)
                gpt_solution_raw = gpt_solutions[gpt_solution_idx]
                gpt_solution_parsed = parse_gpt_solution(gpt_solution_raw)

                print_v(f"Using GPT solution {gpt_solution_idx + 1}")

                # Generate implementation with GPT solution as guidance
                # We'll modify the prompt to include the GPT solution as a reference
                augmented_prompt = f"""Here is a reference solution approach for inspiration:

{gpt_solution_parsed}

Now, please implement the following function:

{prompt}"""

                # Generate solution
                cur_func_impl = None
                fail_cnt = 0
                while cur_func_impl is None and fail_cnt < 3:
                    cur_func_impl = gen.func_impl(
                        augmented_prompt,
                        model,
                        "simple",
                        temperature=0.7
                    )
                    fail_cnt += 1

                if cur_func_impl is None:
                    print_v("Failed to generate implementation")
                    all_attempts.append({
                        "phase": 1,
                        "attempt_num": attempt_num + 1,
                        "implementation": None,
                        "gpt_solution_index": gpt_solution_idx,
                        "passed_public": False,
                        "passed_private": False,
                        "feedback": "Generation failed"
                    })
                    continue

                # Test against public tests
                is_passing_public, feedback, _ = exe.execute(cur_func_impl, tests_i, timeout=10)

                print_v(f"Public tests: {'PASS' if is_passing_public else 'FAIL'}")

                # If passes public, test against private (full test suite)
                passed_private = False
                if is_passing_public:
                    passed_private = exe.evaluate(identifier, cur_func_impl, test_code, timeout=10)
                    print_v(f"Private tests: {'PASS' if passed_private else 'FAIL'}")

                    if passed_private:
                        is_solved = True
                        phase_solved = 1
                        num_success += 1
                        print_v(colored(f"✓ Solved in Phase 1, Attempt {attempt_num + 1}!", "green"))

                # Record attempt
                all_attempts.append({
                    "phase": 1,
                    "attempt_num": attempt_num + 1,
                    "implementation": cur_func_impl,
                    "gpt_solution_index": gpt_solution_idx,
                    "passed_public": is_passing_public,
                    "passed_private": passed_private,
                    "feedback": feedback
                })

                # If solved, we can stop (but we could continue all 4 for data collection)
                if is_solved:
                    break

            # ===== PHASE 2: 3 Reflexion Attempts (only if all 4 failed) =====
            if not is_solved:
                print_v("\n" + "="*60)
                print_v("PHASE 2: 3 reflexion attempts with saved reflections")
                print_v("="*60)

                # Generate reflections from failed attempts
                print_v("\nGenerating reflections from failed attempts...")
                for attempt in all_attempts:
                    if attempt["implementation"] and not attempt["passed_private"]:
                        reflection = gen.self_reflection(
                            attempt["implementation"],
                            attempt["feedback"],
                            model
                        )
                        reflections.append(reflection)
                        print_v(f"  - Reflection {len(reflections)}: {reflection[:100]}...")

                # Combine all reflections
                combined_reflections = "\n\n".join([
                    f"Reflection {i+1}: {refl}"
                    for i, refl in enumerate(reflections)
                ])

                # 3 reflexion attempts
                for attempt_num in range(3):
                    print_v(f"\n--- Reflexion Attempt {attempt_num + 1}/3 ---")

                    # Get the last failed implementation and feedback
                    last_attempt = all_attempts[-1]
                    prev_impl = last_attempt["implementation"]
                    prev_feedback = last_attempt["feedback"]

                    # Phase 2: Use ONLY reflections, NO GPT solutions
                    augmented_feedback = f"""{prev_feedback}

[Previous Reflections]:
{combined_reflections}"""

                    # Generate improved implementation using reflexion
                    new_func_impl = None
                    fail_cnt = 0
                    while new_func_impl is None and fail_cnt < 3:
                        new_func_impl = gen.func_impl(
                            func_sig=prompt,
                            model=model,
                            strategy="reflexion",
                            prev_func_impl=prev_impl,
                            feedback=augmented_feedback,
                            self_reflection=combined_reflections,
                            temperature=0.7
                        )
                        fail_cnt += 1

                    if new_func_impl is None:
                        print_v("Failed to generate implementation")
                        all_attempts.append({
                            "phase": 2,
                            "attempt_num": attempt_num + 1,
                            "implementation": None,
                            "gpt_solution_index": None,  # Phase 2 uses only reflections
                            "passed_public": False,
                            "passed_private": False,
                            "feedback": "Generation failed"
                        })
                        continue

                    # Test against public tests
                    is_passing_public, feedback, _ = exe.execute(new_func_impl, tests_i, timeout=10)

                    print_v(f"Public tests: {'PASS' if is_passing_public else 'FAIL'}")

                    # If passes public, test against private
                    passed_private = False
                    if is_passing_public:
                        passed_private = exe.evaluate(identifier, new_func_impl, test_code, timeout=10)
                        print_v(f"Private tests: {'PASS' if passed_private else 'FAIL'}")

                        if passed_private:
                            is_solved = True
                            phase_solved = 2
                            num_success += 1
                            print_v(colored(f"✓ Solved in Phase 2, Attempt {attempt_num + 1}!", "green"))

                    # Record attempt
                    all_attempts.append({
                        "phase": 2,
                        "attempt_num": attempt_num + 1,
                        "implementation": new_func_impl,
                        "gpt_solution_index": None,  # Phase 2 uses only reflections
                        "passed_public": is_passing_public,
                        "passed_private": passed_private,
                        "feedback": feedback
                    })

                    # If solved, stop
                    if is_solved:
                        break

        except Exception as e:
            print(f"Error processing problem {i}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Record results
        end_time = time()
        llm_cost = gpt_usage(backend=model_name)

        # Get final solution (last implementation or best one)
        final_solution = None
        for attempt in reversed(all_attempts):
            if attempt["passed_private"]:
                final_solution = attempt["implementation"]
                break
        if final_solution is None and all_attempts:
            final_solution = all_attempts[-1]["implementation"]

        # Save results
        item["runtime"] = end_time - start_time
        item["is_solved"] = is_solved
        item["phase_solved"] = phase_solved
        item["attempts"] = all_attempts
        item["reflections"] = reflections
        item["solution"] = final_solution
        item['cost'] = llm_cost['cost']
        item['completion_tokens'] = llm_cost['completion_tokens']
        item['prompt_tokens'] = llm_cost['prompt_tokens']

        write_jsonl(log_path, [item], append=True)

        print_v(f'\nCompleted {i+1}/{num_items}: acc = {round(num_success/(i+1), 2)}')
        print_v(f"Cost so far: ${llm_cost['cost']:.4f}")

    print(colored(f"\n{'='*60}", 'blue'))
    print(colored(f"Final Results:", 'blue'))
    print(colored(f"{'='*60}", 'blue'))
    print(colored(f"Total solved: {num_success}/{num_items} ({100*num_success/num_items:.1f}%)", 'blue'))
    print(colored(gpt_usage(backend=model_name), 'blue'))
