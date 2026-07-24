# How To Use Organized Project Code

这个目录是一个整理后的可运行代码版本。建议从 `organized_project_code/`
目录运行所有命令：

```bash
cd organized_project_code
```

如果还没有装依赖，可以先运行：

```bash
bash setup.sh
```

API key 需要自己设置到环境变量里：

```bash
export OPENAI_API_KEY=<your_openai_key>
export TOGETHER_API_KEY=<your_together_key>
```

LoRA 训练和本地 Llama inference 需要 GPU、可访问的 Llama3-8B base model，
以及 Hugging Face 权限。所有 `.sh` 都写成了简单形式：需要改 GPU 数量、模型路径、
输入数据或输出目录时，直接打开对应 `.sh`，改顶部几行变量即可。

## Overall Workflow

每个 domain 都按同样四步跑：

| Step | 目的 | 产物 |
| --- | --- | --- |
| 1. Generate reflection data | 用 API 生成训练 LoRA 的 reflection / pitfall / decomposition 数据 | `benchmarks/...` 里的训练 JSON |
| 2. Finetune LoRA | 基于生成的数据 SFT/LoRA finetune Llama3-8B | `lora-llama3-8b-*` adapter |
| 3. Infer reflections | 用 finetuned LLM 给 test sample 生成每条样本的 reflection | `generated_reflections/.../*.jsonl` |
| 4. Run agents | 跑 Base、Reflexion、DoT、DoT-Bank、ParamAgent | `results/...` |

`generated_reflections/` 和 `results/` 是运行时产物，不包含在整理好的 benchmark 数据里。

Agent 名称和代码对应关系：

| Agent | `--strategy` | Python entrypoint |
| --- | --- | --- |
| Base / Simple | `simple` | `main*.py` |
| Reflexion | `reflexion` | `main*.py` |
| DoT | `dot` | `main*.py` |
| DoT-Bank | `dot_bank` | `main*.py` |
| ParamAgent | usually `dot` plus generated reflections | `main*_param*.py` |

## Code Domain

Code domain 目录：

```text
code/
```

默认 benchmark：

| 用途 | 数据 |
| --- | --- |
| LoRA training data 1 | `benchmarks/APP_code_datasets.json` |
| LoRA training data 2 | `benchmarks/augmented_coding_datasets_llama3_8b.json` |
| Default test data | `benchmarks/humaneval_full.jsonl` |
| Other test data | `benchmarks/mbpp-py.jsonl`, `benchmarks/leetcode_humaneval.jsonl` |

### 1. Generate Code Reflection Data

Run:

```bash
bash code/generate_reflection_data.sh
```

This calls:

```text
code/gen_code_from_llama3_8b.py
```

Input data:

```text
benchmarks/APP_code_datasets.json
benchmarks/augmented_coding_datasets.json
```

Output data:

```text
benchmarks/APP_code_datasets_llama3_8b.json
benchmarks/augmented_coding_datasets_llama3_8b.json
```

This step uses the Together API and needs `TOGETHER_API_KEY`.

### 2. Finetune Code LoRA

Run:

```bash
bash code/finetune_lora_llama3_8b.sh
```

This calls:

```text
code/LoRA_Llama3_Code_multigpu.py
```

Default base model:

```text
meta-llama/Meta-Llama-3.1-8B-Instruct
```

Training data:

```text
benchmarks/APP_code_datasets.json
benchmarks/augmented_coding_datasets_llama3_8b.json
```

Output adapter:

```text
lora-llama3-8b-code/
```

If you have fewer GPUs, edit this line in `code/finetune_lora_llama3_8b.sh`:

```bash
NUM_GPUS=8
```

### 3. Generate Code Per-Sample Reflections

Run:

```bash
bash code/infer_reflections_llama3_8b.sh
```

This calls:

```text
code/LoRA_Llama3_Code_multigpu_inference.py
```

Input test data:

```text
benchmarks/humaneval_full.jsonl
```

LoRA adapter:

```text
lora-llama3-8b-code/
```

Output reflection file:

```text
generated_reflections/code/humaneval_pitfalls_llama3_8b.jsonl
```

The output key is:

```text
high_temp_pitfall
```

`main_param.py` consumes this file through `--mistake_json_path`.

### 4. Run Code Agents

Run:

```bash
bash code/run_code.sh
```

This script runs these agents:

| Agent | Python file | Important args |
| --- | --- | --- |
| Base / Simple | `code/main.py` | `--strategy simple` |
| Reflexion | `code/main.py` | `--strategy reflexion` |
| DoT | `code/main.py` | `--strategy dot` |
| DoT-Bank | `code/main.py` | `--strategy dot_bank` |
| ParamAgent | `code/main_param.py` | `--strategy dot --use_mistakes --mistake_json_path generated_reflections/code/humaneval_pitfalls_llama3_8b.jsonl` |

Default model:

```text
llama3_1_8b
```

Default dataset:

```text
benchmarks/humaneval_full.jsonl
```

Default result folder:

```text
results/code/
```

To run on MBPP instead, edit `DATASET_PATH` in `code/run_code.sh`:

```bash
DATASET_PATH="benchmarks/mbpp-py.jsonl"
```

## Math Domain

Math domain 目录：

```text
math/
```

Default benchmark:

| 用途 | 数据 |
| --- | --- |
| LoRA training data | `benchmarks/math_finetune_dataset/math_finetune_dataset.json` |
| Default test data | `benchmarks/math/testset.json` |
| Reflection inference test data | `benchmarks/math/testset.jsonl` |
| Other math data | `benchmarks/game24.csv`, cached GSM8K/Hendrycks MATH data |

### 1. Generate Math Reflection Data

Run:

```bash
bash math/generate_reflection_data.sh
```

This calls:

```text
math/make_math_finetune_dataset.py
```

Output data:

```text
benchmarks/math_finetune_dataset/math_finetune_dataset.json
```

This step uses the OpenAI API and needs `OPENAI_API_KEY`.

### 2. Finetune Math LoRA

Run:

```bash
bash math/finetune_lora_llama3_8b.sh
```

This calls:

```text
math/LoRA_Llama3_Math_multigpu.py
```

Training data:

```text
benchmarks/math_finetune_dataset/math_finetune_dataset.json
```

Output adapter:

```text
lora-llama3-8b-math/
```

Default base model:

```text
meta-llama/Meta-Llama-3.1-8B-Instruct
```

If you have fewer GPUs, edit:

```bash
NUM_GPUS=8
```

### 3. Generate Math Per-Sample Reflections

Run:

```bash
bash math/infer_reflections_llama3_8b.sh
```

This calls:

```text
math/LoRA_Llama3_Math_multigpu_inference.py
```

Input test data:

```text
benchmarks/math/testset.jsonl
```

LoRA adapter:

```text
lora-llama3-8b-math/
```

Output reflection file:

```text
generated_reflections/math/math_pitfalls_llama3_8b.jsonl
```

The output key is:

```text
pitfalls_high_temp
```

`mainMath_param.py` consumes this file through `--insight_json_path`.

### 4. Run Math Agents

Run:

```bash
bash math/run_math.sh
```

This script runs these agents:

| Agent | Python file | Important args |
| --- | --- | --- |
| Base / Simple | `math/mainMath.py` | `--strategy simple` |
| Reflexion | `math/mainMath.py` | `--strategy reflexion` |
| DoT | `math/mainMath.py` | `--strategy dot` |
| DoT-Bank | `math/mainMath.py` | `--strategy dot_bank` |
| ParamAgent | `math/mainMath_param.py` | `--strategy dot --use_expert_module --insight_json_path generated_reflections/math/math_pitfalls_llama3_8b.jsonl` |

Default model:

```text
llama3_1_8b
```

Default dataset:

```text
benchmarks/math/testset.json
```

Default result folder:

```text
results/math/
```

## QA Domain

QA domain 目录：

```text
qa/
```

Default benchmark:

| 用途 | 数据 |
| --- | --- |
| Original decomposition training data | `benchmarks/hotpotQA_decomposition_dataset.json` |
| Llama3-generated decomposition training data | `benchmarks/hotpotQA_decomposition_dataset_llama3.json` |
| Default test data | `benchmarks/light_hotpotqa.json` |
| Other test data | `benchmarks/light_2wikimultihopqa.json` |

### 1. Generate QA Reflection Data

Run:

```bash
bash qa/generate_reflection_data.sh
```

This calls:

```text
qa/gen_question_decomposition_data_llama3.py
```

Input data:

```text
benchmarks/hotpotQA_decomposition_dataset.json
```

Output data:

```text
benchmarks/hotpotQA_decomposition_dataset_llama3.json
```

This step uses the Together API and needs `TOGETHER_API_KEY`.

### 2. Finetune QA LoRA

Run:

```bash
bash qa/finetune_lora_llama3_8b.sh
```

This calls:

```text
qa/LoRA_Llama3_QA_multigpu.py
```

Training data:

```text
benchmarks/hotpotQA_decomposition_dataset_llama3.json
```

Output adapter:

```text
lora-llama3-8b-qa/
```

Default base model:

```text
meta-llama/Meta-Llama-3.1-8B-Instruct
```

If you have fewer GPUs, edit:

```bash
NUM_GPUS=8
```

### 3. Generate QA Per-Sample Reflections

Run:

```bash
bash qa/infer_reflections_llama3_8b.sh
```

This calls:

```text
qa/LoRA_Llama3_QA_multigpu_inference.py
```

Input test data:

```text
benchmarks/light_hotpotqa.json
```

LoRA adapter:

```text
lora-llama3-8b-qa/
```

Output reflection file:

```text
generated_reflections/qa/hotpotqa_insights_llama3_8b.jsonl
```

The output key is:

```text
high_temp_insight
```

`mainQA_parametric.py` consumes this file through `--insight_json_path`.

To generate reflections for 2WikiMultiHopQA instead, edit these lines in
`qa/infer_reflections_llama3_8b.sh`:

```bash
INPUT_JSON="benchmarks/light_2wikimultihopqa.json"
OUTPUT_JSON="$OUTPUT_DIR/2wikimultihopqa_insights_llama3_8b.jsonl"
```

### 4. Run QA Agents

Run:

```bash
bash qa/run_qa.sh
```

This script runs these agents:

| Agent | Python file | Important args |
| --- | --- | --- |
| Base / Simple | `qa/mainQA.py` | `--strategy simple` |
| Reflexion | `qa/mainQA.py` | `--strategy reflexion` |
| DoT | `qa/mainQA.py` | `--strategy dot` |
| DoT-Bank | `qa/mainQA.py` | `--strategy dot_bank` |
| ParamAgent | `qa/mainQA_parametric.py` | `--strategy dot --use_expert_module --insight_json_path generated_reflections/qa/hotpotqa_insights_llama3_8b.jsonl` |

Default model:

```text
llama3_1_8b
```

Default dataset:

```text
benchmarks/light_hotpotqa.json
```

Default result folder:

```text
results/qa/
```

To run 2WikiMultiHopQA, edit `qa/run_qa.sh`:

```bash
DATASET_PATH="benchmarks/light_2wikimultihopqa.json"
REFLECTION_JSONL="generated_reflections/qa/2wikimultihopqa_insights_llama3_8b.jsonl"
```

## Running Individual Agents Manually

You do not have to run the full `run_*.sh`. You can run one agent directly.

Code Base:

```bash
python code/main.py \
  --run_name simple_llama3_code \
  --root_dir results/code/simple/ \
  --dataset_path benchmarks/humaneval_full.jsonl \
  --strategy simple \
  --language py \
  --model llama3_1_8b \
  --pass_at_k 1 \
  --max_iters 1 \
  --verbose
```

Code ParamAgent:

```bash
python code/main_param.py \
  --run_name paramagent_llama3_code \
  --root_dir results/code/paramagent/ \
  --dataset_path benchmarks/humaneval_full.jsonl \
  --strategy dot \
  --language py \
  --model llama3_1_8b \
  --pass_at_k 1 \
  --max_iters 5 \
  --inner_iter 5 \
  --use_mistakes \
  --mistake_json_path generated_reflections/code/humaneval_pitfalls_llama3_8b.jsonl \
  --device cuda:0 \
  --verbose
```

Math ParamAgent:

```bash
python math/mainMath_param.py \
  --run_name paramagent_llama3_math \
  --root_dir results/math/paramagent/ \
  --dataset_path benchmarks/math/testset.json \
  --strategy dot \
  --language py \
  --model llama3_1_8b \
  --pass_at_k 1 \
  --max_iters 5 \
  --use_expert_module \
  --insight_json_path generated_reflections/math/math_pitfalls_llama3_8b.jsonl \
  --device cuda:0 \
  --verbose
```

QA ParamAgent:

```bash
python qa/mainQA_parametric.py \
  --run_name paramagent_llama3_qa \
  --root_dir results/qa/paramagent/ \
  --dataset_path benchmarks/light_hotpotqa.json \
  --strategy dot \
  --language py \
  --model llama3_1_8b \
  --pass_at_k 1 \
  --max_iters 8 \
  --device cuda:0 \
  --use_expert_module \
  --insight_json_path generated_reflections/qa/hotpotqa_insights_llama3_8b.jsonl \
  --verbose
```

## Common Changes

Use fewer GPUs:

```bash
NUM_GPUS=1
```

Use a local base model path:

```bash
BASE_MODEL="/path/to/Meta-Llama-3.1-8B-Instruct"
```

Use a checkpoint inside a LoRA output folder:

```bash
LORA_PATH="./lora-llama3-8b-code/checkpoint-500"
```

Run a quick inference test:

```bash
# Add this flag to the end of the inference python command:
--test
```

Change output folders:

```bash
OUTPUT_DIR="./my-lora-output"
ROOT_DIR="results/my_experiment"
```

## Important Notes

- Step 4 ParamAgent depends on Step 3 reflection files. If the reflection file
  does not exist, the `run_*.sh` script skips ParamAgent and prints which Step 3
  script to run.
- API generation can be expensive. For real runs, check the script and dataset
  size before launching.
- Full LoRA training requires GPU memory. Start with fewer samples or fewer
  epochs if debugging.
- `benchmarks/` contains train/test data. Runtime reflection outputs belong in
  `generated_reflections/`, and experiment outputs belong in `results/`.
