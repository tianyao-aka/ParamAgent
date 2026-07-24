#!/bin/bash
set -e

# This uses the finetuned LoRA adapter to generate per-sample math pitfalls.
# If you only want a quick test, add --test to the python command below.

cd "$(dirname "$0")/.."
export PYTHONPATH=.

BASE_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
LORA_PATH="./lora-llama3-8b-math"
INPUT_JSONL="benchmarks/math/testset.jsonl"
OUTPUT_DIR="generated_reflections/math"
OUTPUT_JSONL="$OUTPUT_DIR/math_pitfalls_llama3_8b.jsonl"

mkdir -p "$OUTPUT_DIR"

python math/LoRA_Llama3_Math_multigpu_inference.py \
  --base_model "$BASE_MODEL" \
  --lora_path "$LORA_PATH" \
  --input_jsonl "$INPUT_JSONL" \
  --output_jsonl "$OUTPUT_JSONL" \
  --output_key pitfalls_high_temp \
  --num_gpus 1 \
  --batch_size 1 \
  --num_versions 8
