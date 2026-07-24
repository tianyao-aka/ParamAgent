#!/bin/bash
set -e

# Change NUM_GPUS to the number of GPUs you have.
# This trains a LoRA adapter for code reflection generation.

cd "$(dirname "$0")/.."
export PYTHONPATH=.

NUM_GPUS=8
BASE_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
DATASET1_PATH="benchmarks/APP_code_datasets.json"
DATASET2_PATH="benchmarks/augmented_coding_datasets_llama3_8b.json"
OUTPUT_DIR="./lora-llama3-8b-code"

torchrun --nproc_per_node "$NUM_GPUS" code/LoRA_Llama3_Code_multigpu.py \
  --dataset1_path "$DATASET1_PATH" \
  --dataset2_path "$DATASET2_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --base_model "$BASE_MODEL" \
  --num_epochs 3 \
  --per_device_batch_size 1 \
  --grad_accum_steps 12 \
  --lr 3e-5 \
  --max_seq_len 2048 \
  --seed 1
