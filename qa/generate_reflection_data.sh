#!/bin/bash
set -e

# This step uses the Together API. Set TOGETHER_API_KEY before running.

cd "$(dirname "$0")/.."
export PYTHONPATH=.

python qa/gen_question_decomposition_data_llama3.py \
  --input_dataset benchmarks/hotpotQA_decomposition_dataset.json \
  --output_path benchmarks/hotpotQA_decomposition_dataset_llama3.json \
  --sample_size 15000 \
  --save_interval 500 \
  --random_seed 42
