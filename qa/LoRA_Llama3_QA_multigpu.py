#!/usr/bin/env python
"""
Multi-GPU QLoRA fine-tuning of Llama-3.1-8B on QA decomposition.

Designed for torchrun with 8× GPUs (e.g., 8×48 GB):
- 4-bit QLoRA to keep per-GPU memory manageable.
- DDP via torchrun (one process per GPU).
- Gradient checkpointing + bf16 compute for activation savings.
- Optional DeepSpeed config for ZeRO if you want optimizer/state sharding.

Example launch (single node, 8 GPUs):
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --nproc_per_node 6 qa/LoRA_Llama3_QA_multigpu.py \
  --train_json benchmarks/hotpotQA_decomposition_dataset_llama3.json \
  --output_dir ./lora-llama3-8b-qa \
  --base_model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --num_epochs 3 \
  --per_device_batch_size 2 \
  --grad_accum_steps 10 \
  --lr 3e-5 \
  --max_seq_len 2048 \
  --seed 1
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from peft.utils import prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# ------------------------------- I/O utilities -------------------------------

def load_qa_json(path: str | Path) -> List[Dict[str, str]]:
    """Load a JSON list of {'question', 'decomposition'} dicts."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list of objects.")
    for entry in data:
        if not {"question", "decomposition"} <= entry.keys():
            raise KeyError("Each JSON object must contain 'question' and 'decomposition'.")
    return data


# --------------------------- Prompt‐format helpers ---------------------------

SYS_PROMPT = (
    "You will be given a question.  "
    "Use your knowledge to extract the key point, the underlying intent, "
    "and possible inference patterns needed to answer the question."
)

PRE_INSIGHT_FEWSHOT = """
<Example 1>
q: Anatoly Maltsev and Valentin Turchin were both from Russia, which of the two is known for his work as a mathematician?

Question Parsing and Intent Extraction  
Intent:  
--------------------------------------------------------------------------------  
🔍 Key Components  
1. Entity A:  
- **Anatoly Maltsev** — mathematician and logician known for contributions in mathematical logic and abstract algebra  
2. Entity B:  
- **Valentin Turchin** — computer scientist and philosopher known for work in cybernetics and philosophy of science  
3. Implied Relationship:  
- Comparative inquiry: which individual is more closely associated with the domain of mathematics  
4. Answer Type Expected:  
- Person name (e.g., "Anatoly Maltsev")  
5. Reasoning Type:  
- Comparative factual reasoning  
6. Required Background:  
- Biographical knowledge or retrieved professional profiles  
--------------------------------------------------------------------------------  
🧠 Inference Trace  
1. Retrieve factual data about Maltsev and Turchin’s academic domains.  
2. Classify Maltsev as a mathematician based on core contributions to mathematical logic.  
3. Classify Turchin as mainly working in cybernetics and philosophy.  
4. Eliminate Turchin as primary mathematician.  
5. Conclude Maltsev is the individual known for mathematics.  
--------------------------------------------------------------------------------  
📝 Disambiguation Note  
- Nationality (Russia) does not help differentiate them.

<Example 2>
The Last Girl on Earth was the third concert tour by Barbadian recording artist Rihanna, the tour visited Europe, Asia, North America and Australia to support her fourth studio album, which 2009 , and fourth studio album by Barbadian singer Rihanna, and released on November 20, 2009 by Def Jam Recordings and SRP Records?

### Question Parsing and Intent Extraction

**Intent:**
--------------------------------------------------------------------------------
🔍 **Key Components**
1. **Entity A**:  
   - *The Last Girl on Earth* — Rihanna's third concert tour, associated with promoting a studio album  

2. **Entity B**:  
   - *Fourth studio album* by Rihanna — referenced multiple times, released in 2009  

3. **Key Relationship / Constraint**:  
   - Identify the name of Rihanna’s **fourth studio album**, which was released on **November 20, 2009**, and **supported by** her third concert tour, *The Last Girl on Earth*  

4. **Answer Type Expected**:  
   - Album title (e.g., *Rated R*)  

5. **Reasoning Type**:  
   - Factual entity retrieval based on event-album association and release date  

6. **Required Background**:  
   - Rihanna’s discography: album release dates and which albums were promoted during which concert tours  

--------------------------------------------------------------------------------

🧠 **Inference Trace**
- Determine the name of Rihanna’s **fourth studio album**  
- Confirm that this album was released on **November 20, 2009**  
- Verify that this album was the basis for the **"The Last Girl on Earth"** tour  
- Conclude that the album is **Rated R**

--------------------------------------------------------------------------------

📝 **Disambiguation Note**
- Although the question includes fragmented/redundant phrasing, the focus is clear: determine the album that matches both the **release date** and **tour association**
"""


def format_chat_prompt(question: str, decomposition: str | None = None) -> str:
    """
    Create a Llama-3 chat prompt with optional supervised target.
    """
    user_part = (
        "[INST] <<SYS>> "
        f"{SYS_PROMPT}"
        " <</SYS>>\n\n"
        f"Here are some examples:{PRE_INSIGHT_FEWSHOT}\n\n [Question]: {question.strip()}"
        " [/INST]"
    )
    if decomposition is None:
        return f"<s>{user_part}"
    return f"<s>{user_part} {decomposition.strip()}</s>"


# ---------------------------- Argument interface ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Multi-GPU QLoRA fine-tuner for QA decomposition")
    p.add_argument("--train_json", required=True, help="Path to training JSON.")
    p.add_argument("--output_dir", required=True, help="Where to save adapters & checkpoints.")
    p.add_argument("--base_model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--per_device_batch_size", type=int, default=1,
                   help="Micro-batch size per GPU.")
    p.add_argument("--grad_accum_steps", type=int, default=16,
                   help="Gradient accumulation steps to reach desired global batch.")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42, help="Random seed for split/shuffle.")
    p.add_argument("--deepspeed_config", type=str, default=None,
                   help="Optional DeepSpeed JSON (e.g., deepspeed.json).")
    return p.parse_args()


# ------------------------------- Main routine --------------------------------

def main() -> None:
    args = parse_args()

    # DDP setup: one process per GPU
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if local_rank == 0:
        print(f"Launching QLoRA QA on {world_size} GPUs — base model: {args.base_model}")

    torch.manual_seed(args.seed)

    # 1) Load + split dataset
    raw_data = load_qa_json(args.train_json)
    full_ds = Dataset.from_list(
        [{"full_text": format_chat_prompt(ex["question"], ex["decomposition"])} for ex in raw_data]
    ).shuffle(seed=args.seed)
    split = full_ds.train_test_split(test_size=0.15, seed=args.seed)
    train_ds, valid_ds = split["train"], split["test"]

    # 2) Tokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tok.pad_token = tok.eos_token

    # 3) Base model in 4-bit (QLoRA)
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_cfg,
        device_map={"": local_rank},  # torchrun assigns one device per rank
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # 4) LoRA adapters
    lora_cfg = LoraConfig(
        r=64,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    if local_rank == 0:
        model.print_trainable_parameters()

    # 5) Tokenisation
    def tok_fn(batch):
        t = tok(
            batch["full_text"],
            max_length=args.max_seq_len,
            truncation=True,
            padding="max_length",
        )
        t["labels"] = t["input_ids"].copy()
        return t

    train_tok = train_ds.map(tok_fn, batched=True, remove_columns=["full_text"])
    valid_tok = valid_ds.map(tok_fn, batched=True, remove_columns=["full_text"])

    # 6) Collator
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    # 7) Training arguments
    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        tf32=True,
        logging_steps=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=4,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        report_to="none",
        ddp_find_unused_parameters=False,
        deepspeed=args.deepspeed_config,
        remove_unused_columns=False,
    )

    # 8) Trainer
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=valid_tok,
        tokenizer=tok,
        data_collator=collator,
    )

    # 9) Train & save (rank 0 handles artifacts)
    trainer.train()
    if local_rank == 0:
        trainer.save_model(args.output_dir)
        with open(Path(args.output_dir) / "loss_history.json", "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)
        print(f"✅ Fine-tuning complete — artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
