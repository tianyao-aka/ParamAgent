#!/bin/bash
set -e

# Run this once before using the project:
#   bash setup.sh

cd "$(dirname "$0")"

pip install -r requirements.txt

# Extra packages used by LoRA training and local Llama inference.
pip install \
  transformers \
  peft \
  accelerate \
  bitsandbytes \
  deepspeed \
  sentencepiece \
  safetensors \
  tqdm
