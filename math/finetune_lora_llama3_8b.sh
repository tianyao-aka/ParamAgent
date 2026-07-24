#!/bin/bash
set -e

# Change NUM_GPUS to the number of GPUs you have.
# This trains a LoRA adapter for math reflection generation.

cd "$(dirname "$0")/.."
export PYTHONPATH=.

NUM_GPUS=8
BASE_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
DATASET_PATH="benchmarks/math_finetune_dataset/math_finetune_dataset.json"
OUTPUT_DIR="./lora-llama3-8b-math"

torchrun --nproc_per_node "$NUM_GPUS" math/LoRA_Llama3_Math_multigpu.py \
  --dataset_path "$DATASET_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --base_model "$BASE_MODEL" \
  --num_epochs 2 \
  --per_device_batch_size 1 \
  --grad_accum_steps 16 \
  --lr 1.5e-5 \
  --max_seq_len 2048 \
  --seed 1
