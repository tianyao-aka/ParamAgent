import os
import argparse
import sys
from simple import run_simple
# from reflexion_parametric import run_reflexion
from reflexion import run_reflexion
# from test_acc import run_test_acc
from utils import read_jsonl, read_jsonl_gz, read_jsonl_map
import json
from dot import run_dot
# from dot_parametric import run_dot
from dot_bank import run_dot_bank
try:
    from LoRA_Llama3_Code_Inference import CodePitfallAgent
except ModuleNotFoundError:
    CodePitfallAgent = None  # Module not available
import torch
import gc

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
    parser.add_argument("--inner_iter", type=int,
                        help="inner iterations", default=5)
    parser.add_argument("--is_leetcode", action='store_true',
                        help="To run the leetcode benchmark")
    parser.add_argument("--is_game24", action='store_true',
                        help="To enable mistakes using expert module")
    parser.add_argument("--is_QA", action='store_true',
                        help="Use QA dataset")
    parser.add_argument("--use_mistakes", action='store_true',
                        help="To enable mistakes using expert module")
    parser.add_argument("--mistake_json_path", type=str, default='',
                        help="load the pre-calc code pitfalls")
    parser.add_argument("--verbose", action='store_true',
                        help="To print live logs")
    parser.add_argument("--device", type=str,
                        help="Pass@k metric", default='cuda:0')
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate (for testing)")
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
    # check if the root dir exists and create it if not
    if not os.path.exists(args.root_dir):
        os.makedirs(args.root_dir)

    # get the dataset name
    dataset_name = os.path.basename(args.dataset_path).replace("jsonl", "")

    # check if log path already exists
    log_dir = os.path.join(args.root_dir, args.run_name)
    log_path = os.path.join(
        log_dir, f"{dataset_name}_{args.strategy}_{args.max_iters}_{args.model}_pass_at_k_{args.pass_at_k}_{args.language}.jsonl")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # print starting message
    if args.verbose:
        print(f"""
Starting run with the following parameters:
strategy: {args.strategy}
pass@k: {args.pass_at_k}
""")
    else:
        print(f"Logs will be saved in `{log_dir}`")
    dataset=None
    # load the dataset
    print(f'Loading the dataset...')
    if args.dataset_path.endswith(".jsonl"):
        dataset = read_jsonl(args.dataset_path)
    elif args.dataset_path.endswith(".jsonl.gz"):
        dataset = read_jsonl_gz(args.dataset_path)
    else:
        raise ValueError(
            f"Dataset path `{args.dataset_path}` is not supported")

    print(f"Loaded {len(dataset)} examples")

    # Detect dataset type
    if 'livecodebench' in args.dataset_path.lower():
        dataset_type = 'livecodebench'
    elif 'mbpp' in args.dataset_path.lower():
        dataset_type = 'mbpp'
    else:
        dataset_type = 'humaneval'

    # Limit dataset if max_samples is specified
    if args.max_samples is not None and args.max_samples < len(dataset):
        dataset = dataset[:args.max_samples]
        print(f"Limited to first {args.max_samples} examples")

    run_strategy = strategy_factory(args.strategy)
    # for visible test cases for HumanEval
    if dataset_type == 'humaneval':
        visible_tests = read_jsonl_map("benchmarks/humaneval_visible_tests.jsonl", primary_key='entry_point')
    else:
        visible_tests = None
    # print (run_strategy)
    PitfallAgent = None
    mistake_json_file = None
    if args.use_mistakes:
        if len(args.mistake_json_path)>3:
            mistake_json_file = read_jsonl(args.mistake_json_path)
            if 'high_temp_pitfall' in mistake_json_file[0]:
                PitfallAgent = None
            else:
                if CodePitfallAgent is not None:
                    PitfallAgent = CodePitfallAgent(base_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                lora_path="LoRA/code_r128/checkpoint-290/",
                device=args.device)
                else:
                    print("WARNING: CodePitfallAgent not available, setting to None")
                    PitfallAgent = None
        else:
            if CodePitfallAgent is not None:
                PitfallAgent = CodePitfallAgent(base_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                lora_path="LoRA/code_r128/checkpoint-290/",
                device=args.device)
                print ('using pitfall agent (Llama3.1-8b)')
            else:
                print("WARNING: CodePitfallAgent not available, setting to None")
                PitfallAgent = None
    # sys.exit(0)
    # start the run
    # evaluate with pass@k
    if args.strategy != 'simple' and 'mbpp' not in args.dataset_path.lower():
        run_strategy(
            dataset=dataset,
            dataset_type=dataset_type,
            model_name=args.model,
            language=args.language,
            max_iters=args.max_iters,
            pass_at_k=args.pass_at_k,
            log_path=log_path,
            verbose=args.verbose,
            expansion_factor=1,
            is_leetcode=args.is_leetcode,
            is_game24=args.is_game24,
            visible_tests=visible_tests,
            use_mistakes = args.use_mistakes,
            pitfall_agent=PitfallAgent,
            mistake_json_file=mistake_json_file
        )
    else:
        run_strategy(
            dataset=dataset,
            dataset_type=dataset_type,
            model_name=args.model,
            language=args.language,
            max_iters=args.max_iters,
            pass_at_k=args.pass_at_k,
            log_path=log_path,
            verbose=args.verbose,
            expansion_factor=1,
            is_leetcode=args.is_leetcode,
            is_game24=args.is_game24,
            use_mistakes = args.use_mistakes,
            pitfall_agent=PitfallAgent,
            mistake_json_file=mistake_json_file,
        )        

    print(f"Done! Check out the logs in `{log_path}`")
    
    # If using GPU
    # torch.cuda.empty_cache()
    # torch.cuda.ipc_collect()  # clean inter-process shared memory

    # Python-level cleanup
    gc.collect()

    # Flush logs if needed
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

if __name__ == "__main__":
    args = get_args()
    main(args)
