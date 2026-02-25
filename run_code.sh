# HumanEval Benchmark Examples
#

# Simple baseline (single attempt, no reflection)
python main.py \
    --run_name "simple_humaneval" \
    --root_dir ./results/humaneval/simple/ \
    --dataset_path benchmarks/humaneval_full.jsonl \
    --strategy "simple" \
    --language "py" \
    --model "llama3_1_8b" \
    --pass_at_k "1" \
    --max_iters "1" \
    --verbose


# Reflexion baseline (single-path iterative refinement)
python main.py \
    --run_name "reflexion_humaneval" \
    --root_dir ./results/humaneval/reflexion/ \
    --dataset_path benchmarks/humaneval_full.jsonl \
    --strategy "reflexion" \
    --language "py" \
    --model "llama3_1_8b" \
    --pass_at_k "1" \
    --max_iters "5" \
    --verbose


# DoT (Diversity of Thoughts)
python main.py \
    --run_name "dot_humaneval" \
    --root_dir ./results/humaneval/dot/ \
    --dataset_path benchmarks/humaneval_full.jsonl \
    --strategy "dot" \
    --language "py" \
    --model "llama3_1_8b" \
    --pass_at_k "1" \
    --max_iters "5" \
    --verbose


# DoT-bank
python main.py \
    --run_name "dot_bank_humaneval" \
    --root_dir ./results/humaneval/dot_bank/ \
    --dataset_path benchmarks/humaneval_full.jsonl \
    --strategy "dot_bank" \
    --language "py" \
    --model "llama3_1_8b" \
    --pass_at_k "1" \
    --max_iters "5" \
    --verbose


# ParamAgent
python main_param.py \
    --run_name "paramAgent_humaneval" \
    --root_dir ./results/humaneval/paramAgent/ \
    --dataset_path benchmarks/humaneval_full.jsonl \
    --strategy "dot" \
    --language "py" \
    --model "llama3_1_8b" \
    --pass_at_k "1" \
    --max_iters "5" \
    --use_mistakes \
    --mistake_json_path ./benchmarks/code_pitfalls/humaneval_full_pitfalls.jsonl \
    --verbose
