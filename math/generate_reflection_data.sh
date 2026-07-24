#!/bin/bash
set -e

# This step uses the OpenAI API. Set OPENAI_API_KEY before running.

cd "$(dirname "$0")/.."
export PYTHONPATH=.

OUTPUT_JSON="benchmarks/math_finetune_dataset/math_finetune_dataset.json"

mkdir -p benchmarks/math_finetune_dataset

python math/make_math_finetune_dataset.py \
  --save_path "$OUTPUT_JSON"
