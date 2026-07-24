#!/bin/bash
set -e

# Run this file from anywhere with:
#   bash code/generate_reflection_data.sh
#
# This step uses the Together API. Set TOGETHER_API_KEY before running.

cd "$(dirname "$0")/.."
export PYTHONPATH=.

python code/gen_code_from_llama3_8b.py \
  --dataset both \
  --app_input benchmarks/APP_code_datasets.json \
  --app_output benchmarks/APP_code_datasets_llama3_8b.json \
  --aug_input benchmarks/augmented_coding_datasets.json \
  --aug_output benchmarks/augmented_coding_datasets_llama3_8b.json
