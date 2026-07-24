#!/bin/bash
set -e

# Change NUM_GPUS to the number of GPUs you have.
# This trains a LoRA adapter for QA decomposition/reflection generation.

cd "$(dirname "$0")/.."
export PYTHONPATH=.

NUM_GPUS=8
BASE_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
TRAIN_JSON="benchmarks/hotpotQA_decomposition_dataset_llama3.json"
OUTPUT_DIR="./lora-llama3-8b-qa"

torchrun --nproc_per_node "$NUM_GPUS" qa/LoRA_Llama3_QA_multigpu.py \
  --train_json "$TRAIN_JSON" \
  --output_dir "$OUTPUT_DIR" \
  --base_model "$BASE_MODEL" \
  --num_epochs 3 \
  --per_device_batch_size 1 \
  --grad_accum_steps 12 \
  --lr 3e-5 \
  --max_seq_len 2048 \
  --seed 1
