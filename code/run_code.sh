#!/bin/bash
set -e

# This runs the code baselines and then ParamAgent if reflection data exists.

cd "$(dirname "$0")/.."
export PYTHONPATH=.

MODEL="llama3_1_8b"
DATASET_PATH="benchmarks/humaneval_full.jsonl"
ROOT_DIR="results/code"
REFLECTION_JSONL="generated_reflections/code/humaneval_pitfalls_llama3_8b.jsonl"

python code/main.py \
  --run_name "simple_$MODEL" \
  --root_dir "$ROOT_DIR/simple/" \
  --dataset_path "$DATASET_PATH" \
  --strategy simple \
  --language py \
  --model "$MODEL" \
  --pass_at_k 1 \
  --max_iters 1 \
  --verbose

python code/main.py \
  --run_name "reflexion_$MODEL" \
  --root_dir "$ROOT_DIR/reflexion/" \
  --dataset_path "$DATASET_PATH" \
  --strategy reflexion \
  --language py \
  --model "$MODEL" \
  --pass_at_k 1 \
  --max_iters 5 \
  --verbose

python code/main.py \
  --run_name "dot_$MODEL" \
  --root_dir "$ROOT_DIR/dot/" \
  --dataset_path "$DATASET_PATH" \
  --strategy dot \
  --language py \
  --model "$MODEL" \
  --pass_at_k 1 \
  --max_iters 5 \
  --verbose

python code/main.py \
  --run_name "dot_bank_$MODEL" \
  --root_dir "$ROOT_DIR/dot_bank/" \
  --dataset_path "$DATASET_PATH" \
  --strategy dot_bank \
  --language py \
  --model "$MODEL" \
  --pass_at_k 1 \
  --max_iters 5 \
  --verbose

if [ -f "$REFLECTION_JSONL" ]; then
  python code/main_param.py \
    --run_name "paramagent_$MODEL" \
    --root_dir "$ROOT_DIR/paramagent/" \
    --dataset_path "$DATASET_PATH" \
    --strategy dot \
    --language py \
    --model "$MODEL" \
    --pass_at_k 1 \
    --max_iters 5 \
    --inner_iter 5 \
    --use_mistakes \
    --mistake_json_path "$REFLECTION_JSONL" \
    --device cuda:0 \
    --verbose
else
  echo "Skip ParamAgent because $REFLECTION_JSONL does not exist yet."
  echo "Run: bash code/infer_reflections_llama3_8b.sh"
fi
