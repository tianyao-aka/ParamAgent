#!/bin/bash
set -e

# This uses the finetuned LoRA adapter to generate per-sample QA insights.
# If you only want a quick test, add --test to the python command below.

cd "$(dirname "$0")/.."
export PYTHONPATH=.

BASE_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
LORA_PATH="./lora-llama3-8b-qa"
INPUT_JSON="benchmarks/light_hotpotqa.json"
OUTPUT_DIR="generated_reflections/qa"
OUTPUT_JSON="$OUTPUT_DIR/hotpotqa_insights_llama3_8b.jsonl"

mkdir -p "$OUTPUT_DIR"

python qa/LoRA_Llama3_QA_multigpu_inference.py \
  --base_model "$BASE_MODEL" \
  --lora_path "$LORA_PATH" \
  --input_json "$INPUT_JSON" \
  --output_json "$OUTPUT_JSON" \
  --output_key high_temp_insight \
  --num_gpus 1 \
  --batch_size 1 \
  --num_versions 8
