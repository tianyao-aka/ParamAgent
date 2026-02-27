# ParamAgent

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set OpenAI API key:
```bash
export OPENAI_API_KEY=<your key>
```

3. For Together AI models:
```bash
export TOGETHER_API_KEY=<your key>
```

## Usage

See `run_code.sh` for example commands. First-stage result log corresponds to ParamAgent, while second-stage result log corresponds to ParamAgent-plus, where memory bank is used to augment the reasoning ability.

## ParamMem

In `benchmarks/code_pitfalls/`, we provide model-based reflections from the ParamMem fine-tuned using a GPT-4o-mini labelled auxiliary dataset.

From [https://huggingface.co/TianJun1/lora-llama3-8b-code](https://huggingface.co/TianJun1/lora-llama3-8b-code), you can download the ParamMem LoRA adapter that uses a Llama-3.1-8B labelled dataset to try self-improving reflection generation.

Use `LoRA_Llama3_Code_multigpu_inference.py` to run inference with the LoRA adapter:
```bash
python LoRA_Llama3_Code_multigpu_inference.py \
    --base_model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --lora_path TianJun1/lora-llama3-8b-code \
    --input_jsonl benchmarks/humaneval_full.jsonl \
    --output_jsonl benchmarks/code_pitfalls/humaneval_pitfalls_llama3_8b.jsonl \
    --num_gpus 1 --batch_size 1
```

## Acknowledgements

Our codebase is adapted from [Diversity of Thoughts (DoT)](https://openreview.net/forum?id=ZsP3YbYeE9) (Lingam et al., ICLR 2025). We thank the authors for their open-source contribution.
