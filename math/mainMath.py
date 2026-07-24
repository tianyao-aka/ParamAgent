import os
import argparse
import sys
from simple_Math import run_simple
# from reflexion_QA_parametric import run_reflexion
from reflexion_Math import run_reflexion
# from test_acc import run_test_acc
from utils import read_jsonl, read_jsonl_gz, read_jsonl_map
import json
# from dot_QA import run_dot
from dot_Math import run_dot
from dot_bank_math import run_dot_bank
import torch
# from LoRA_Llama3_Math_Inference import MathPitfallAgent
import gc

MathPitfallAgent = None

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, help="The name of the run")
    parser.add_argument("--root_dir", type=str,
                        help="The root logging directory", default="root")
    parser.add_argument("--dataset_path", type=str,
                        help="The path to the benchmark dataset", default="root")
    parser.add_argument("--strategy", type=str,
                        help="Strategy: `simple`, `reflexion`")
    parser.add_argument("--language", type=str, help="Strategy: `py` or `rs`")
    parser.add_argument(
        "--model", type=str, help="OpenAI models only for now. For best results, use GPT-4")
    parser.add_argument("--pass_at_k", type=int,
                        help="Pass@k metric", default=1)
    parser.add_argument("--max_iters", type=int,
                        help="The maximum number of self-improvement iterations", default=3)

    parser.add_argument("--is_leetcode", action='store_true',
                        help="To run the leetcode benchmark")
    parser.add_argument("--use_expert_module", action='store_true',
                        help="To enable using expert module")
    parser.add_argument("--insight_json_path", type=str, default='',
                        help="load the pre-calc code pitfalls")
    parser.add_argument("--verbose", action='store_true',
                        help="To print live logs")
    parser.add_argument("--device", type=str,
                        help="device", default='cuda:0')
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Limit the number of samples to process (for testing)")
    args = parser.parse_args()
    return args


def strategy_factory(strategy: str):
    def kwargs_wrapper_gen(func, delete_keys=[]):
        def kwargs_wrapper(**kwargs):
            for key in delete_keys:
                del kwargs[key]
            return func(**kwargs)
        return kwargs_wrapper


    if strategy == "simple":
        return kwargs_wrapper_gen(run_simple, delete_keys=["expansion_factor", "max_iters"])
    elif strategy == "dot":
        return kwargs_wrapper_gen(run_dot, delete_keys=["expansion_factor"])
    elif strategy == "dot_bank":
        return kwargs_wrapper_gen(run_dot_bank, delete_keys=["expansion_factor"])
    elif strategy == "reflexion":
        return kwargs_wrapper_gen(run_reflexion, delete_keys=["expansion_factor"])
    elif strategy == "test-acc":
        return kwargs_wrapper_gen(run_test_acc, delete_keys=["expansion_factor", "max_iters"])
    else:
        raise ValueError(f"Strategy `{strategy}` is not supported")


def main(args):

    if not os.path.exists(args.root_dir):
        os.makedirs(args.root_dir)

    # get the dataset name
    dataset_name = os.path.basename(args.dataset_path).replace("json", "")
    # check if log path already exists
    log_dir = os.path.join(args.root_dir, args.run_name)
    log_path = os.path.join(
        log_dir, f"{dataset_name}_{args.strategy}_{args.max_iters}_{args.model}_pass_at_k_{args.pass_at_k}_{args.language}.jsonl")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # check if the strategy is valid
    run_strategy = strategy_factory(args.strategy)
    
    # print starting message
    if args.verbose:
        print(f"""
Starting run with the following parameters:
strategy: {args.strategy}
pass@k: {args.pass_at_k}
""")
    else:
        print(f"Logs will be saved in `{log_dir}`")
    # load the dataset
    dataset=None
    # load the dataset
    print(f'Loading the dataset...')
    if args.dataset_path.endswith(".json"):
        # load json
        with open(args.dataset_path) as f:
            dataset = json.load(f)
    else:
        raise ValueError(
            f"Dataset path `{args.dataset_path}` is not supported")

    print(f"Loaded {len(dataset)} examples")

    # Limit dataset size if num_samples is specified
    if args.num_samples is not None and args.num_samples > 0:
        dataset = dataset[:args.num_samples]
        print(f"Limited to first {len(dataset)} samples for testing")

    Math_agent = None
    if args.use_expert_module:
        if len(args.insight_json_path)>3:
            print (f'Loading the insights from {args.insight_json_path}')
            insight_json_file = read_jsonl(args.insight_json_path)
        else:
            Math_agent = MathPitfallAgent(
                        adapter_dir="./LoRA/math/",
                        base_model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        device="cuda:0",
                        max_length=1024,
                        max_new_tokens=1024)
    
    # for visible test cases for HumanEval
    visible_tests = read_jsonl_map("benchmarks/humaneval_visible_tests.jsonl", primary_key='entry_point') 
    # print (run_strategy)
    
    # sys.exit(0)
    # start the run
    # evaluate with pass@k
    if args.strategy != 'simple': 
        run_strategy(
            dataset=dataset,
            model_name=args.model,
            language=args.language,
            max_iters=args.max_iters,
            pass_at_k=args.pass_at_k,
            log_path=log_path,
            verbose=args.verbose,
            expansion_factor=1,
            use_parsing = args.use_expert_module,
            insight_json_file = insight_json_file if len(args.insight_json_path)>3 else None,
            Math_agent = Math_agent
        )
    else:
        run_strategy(
            dataset=dataset,
            model_name=args.model,
            language="math",
            max_iters=args.max_iters,
            pass_at_k=args.pass_at_k,
            log_path=log_path,
            verbose=args.verbose,
            expansion_factor=1,
            use_parsing = args.use_expert_module,
            device = args.device,
            insight_json_file = insight_json_file if len(args.insight_json_path)>3 else None,
            Math_agent = Math_agent
        )     

    print(f"Done! Check out the logs in `{log_path}`")

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()  # clean inter-process shared memory

    # Python-level cleanup
    gc.collect()

    # Flush logs if needed
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

if __name__ == "__main__":
    args = get_args()
    main(args)

