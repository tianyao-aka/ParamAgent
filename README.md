# ParamAgent Finalized

This repository contains the organized code for ParamAgent across three domains:
code generation, math reasoning, and multi-hop QA. It includes the runnable code,
the curated benchmark/train-test data, and simple shell scripts for the full
workflow.

For the detailed guide, see [HowToUse.md](HowToUse.md).

## Setup

Run all commands from the repository root:

```bash
cd ParamAgentFinalized
bash setup.sh
```

Set API keys when running API-based data generation:

```bash
export OPENAI_API_KEY=<your_openai_key>
export TOGETHER_API_KEY=<your_together_key>
```

LoRA training and local Llama inference need GPUs, Hugging Face model access,
and a Llama3-8B base model. To change GPU count, model path, data path, or
output path, edit the variable block near the top of the corresponding `.sh`
file.

## Workflow

Each domain follows the same four steps:

| Step | Purpose | Output |
| --- | --- | --- |
| 1. Generate reflection data | Use an API model to generate training reflections, pitfalls, or decompositions | `benchmarks/...` training JSON |
| 2. Finetune LoRA | SFT/LoRA finetune Llama3-8B on the generated data | `lora-llama3-8b-*` |
| 3. Infer reflections | Use the finetuned LLM to generate per-sample reflections for test data | `generated_reflections/.../*.jsonl` |
| 4. Run agents | Run Base, Reflexion, DoT, DoT-Bank, and ParamAgent | `results/...` |

`generated_reflections/`, `results/`, LoRA adapters, and model checkpoints are
runtime outputs and are ignored by git.

## Code Domain

Default data:

- Train: `benchmarks/APP_code_datasets.json`
- Train: `benchmarks/augmented_coding_datasets_llama3_8b.json`
- Test: `benchmarks/humaneval_full.jsonl`
- Other tests: `benchmarks/mbpp-py.jsonl`, `benchmarks/leetcode_humaneval.jsonl`

Run the full code workflow:

```bash
# 1. Generate API reflection data with Together
bash code/generate_reflection_data.sh

# 2. Finetune Llama3-8B LoRA
bash code/finetune_lora_llama3_8b.sh

# 3. Generate per-sample reflections with the finetuned LoRA
bash code/infer_reflections_llama3_8b.sh

# 4. Run Base, Reflexion, DoT, DoT-Bank, and ParamAgent
bash code/run_code.sh
```

Main Python files:

- Data generation: `code/gen_code_from_llama3_8b.py`
- LoRA finetune: `code/LoRA_Llama3_Code_multigpu.py`
- Reflection inference: `code/LoRA_Llama3_Code_multigpu_inference.py`
- Base / Reflexion / DoT / DoT-Bank: `code/main.py`
- ParamAgent: `code/main_param.py`, `code/paramAgent.py`

ParamAgent uses:

```text
generated_reflections/code/humaneval_pitfalls_llama3_8b.jsonl
```

## Math Domain

Default data:

- Train: `benchmarks/math_finetune_dataset/math_finetune_dataset.json`
- Test: `benchmarks/math/testset.json`
- Reflection inference test data: `benchmarks/math/testset.jsonl`

Run the full math workflow:

```bash
# 1. Generate API reflection data with OpenAI
bash math/generate_reflection_data.sh

# 2. Finetune Llama3-8B LoRA
bash math/finetune_lora_llama3_8b.sh

# 3. Generate per-sample reflections with the finetuned LoRA
bash math/infer_reflections_llama3_8b.sh

# 4. Run Base, Reflexion, DoT, DoT-Bank, and ParamAgent
bash math/run_math.sh
```

Main Python files:

- Data generation: `math/make_math_finetune_dataset.py`
- LoRA finetune: `math/LoRA_Llama3_Math_multigpu.py`
- Reflection inference: `math/LoRA_Llama3_Math_multigpu_inference.py`
- Base / Reflexion / DoT / DoT-Bank: `math/mainMath.py`
- ParamAgent: `math/mainMath_param.py`

ParamAgent uses:

```text
generated_reflections/math/math_pitfalls_llama3_8b.jsonl
```

## QA Domain

Default data:

- Train: `benchmarks/hotpotQA_decomposition_dataset_llama3.json`
- Test: `benchmarks/light_hotpotqa.json`
- Other test: `benchmarks/light_2wikimultihopqa.json`

Run the full QA workflow:

```bash
# 1. Generate API reflection/decomposition data with Together
bash qa/generate_reflection_data.sh

# 2. Finetune Llama3-8B LoRA
bash qa/finetune_lora_llama3_8b.sh

# 3. Generate per-sample reflections with the finetuned LoRA
bash qa/infer_reflections_llama3_8b.sh

# 4. Run Base, Reflexion, DoT, DoT-Bank, and ParamAgent
bash qa/run_qa.sh
```

Main Python files:

- Data generation: `qa/gen_question_decomposition_data_llama3.py`
- LoRA finetune: `qa/LoRA_Llama3_QA_multigpu.py`
- Reflection inference: `qa/LoRA_Llama3_QA_multigpu_inference.py`
- Base / Reflexion / DoT / DoT-Bank: `qa/mainQA.py`
- ParamAgent: `qa/mainQA_parametric.py`

ParamAgent uses:

```text
generated_reflections/qa/hotpotqa_insights_llama3_8b.jsonl
```

## Agents

| Agent | Strategy / script behavior | Entrypoint |
| --- | --- | --- |
| Base / Simple | `--strategy simple` | `main.py`, `mainMath.py`, `mainQA.py` |
| Reflexion | `--strategy reflexion` | `main.py`, `mainMath.py`, `mainQA.py` |
| DoT | `--strategy dot` | `main.py`, `mainMath.py`, `mainQA.py` |
| DoT-Bank | `--strategy dot_bank` | `main.py`, `mainMath.py`, `mainQA.py` |
| ParamAgent | `--strategy dot` plus generated reflection file | `main_param.py`, `mainMath_param.py`, `mainQA_parametric.py` |

`run_code.sh`, `run_math.sh`, and `run_qa.sh` run all baseline agents first.
They run ParamAgent only when the Step 3 reflection file exists; otherwise they
print which inference script to run first.

## Repository Layout

```text
code/          Code-generation workflow and agents
math/          Math workflow and agents
qa/            Multi-hop QA workflow and agents
benchmarks/    Included train/test data
generators/    Shared model and prompt-generation utilities
executors/     Shared execution/evaluation utilities
HowToUse.md    Detailed step-by-step usage guide
```
