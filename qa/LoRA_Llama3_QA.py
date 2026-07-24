#!/usr/bin/env python
# finetune_llama3_lora.py
"""
LoRA fine-tuning of Meta-Llama-3-8B-Instruct on a QA-decomposition dataset,
with in‐training validation each epoch and top-3 checkpoint retention.

Usage on single A100 80GB
-------------------------
python LoRA_Llama3_QA.py \
       --train_json benchmarks/hotpotQA_decomposition_dataset_llama3.json \
       --output_dir ./lora-llama3-8b-qa \
       --num_epochs 3 \
       --per_device_batch_size 8 \
       --grad_accum_steps 2 \
       --lr 2e-5 \
       --seed 1 \
       --device_id 0

To use GPU 3:
       python LoRA_Llama3_QA.py ... --device_id 3

Dependencies
------------
pip install "transformers>=4.41.0" "datasets>=2.18.0" \
            accelerate peft bitsandbytes==0.43.2 flash-attn==2.5.4
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig

# ------------------------------- I/O utilities -------------------------------

def load_qa_json(path: str | Path) -> List[Dict[str, str]]:
    """
    Load a JSON file containing a *list* of {"question", "decomposition"} dicts.

    Args:
        path: Path to the JSON file.

    Returns:
        A list of dicts, each with keys "question" and "decomposition".
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list of objects.")
    for entry in data:
        if not {"question", "decomposition"} <= entry.keys():
            raise KeyError("Each JSON object must contain both 'question' and 'decomposition'.")
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
    Create an Llama-3 chat prompt.  
    If `decomposition` is None, this is the *input* only (no assistant reply).
    Otherwise it concatenates input + assistant reply for supervised training.

    Args:
        question: the user’s question.
        decomposition: the target decomposition (or None for inference prompts).

    Returns:
        A single string ready for tokenization.
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
    else:
        return f"<s>{user_part} {decomposition.strip()}</s>"

# ---------------------------- Argument interface ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA FT for Llama-3-8B-Chat with validation each epoch"
    )
    parser.add_argument("--train_json",  required=True, help="Path to training JSON.")
    parser.add_argument("--output_dir",  required=True, help="Where to save adapters & checkpoints.")
    parser.add_argument("--num_epochs",  type=int, default=5)
    parser.add_argument("--per_device_batch_size", type=int, default=8,
                        help="Micro-batch size per GPU.")
    parser.add_argument("--grad_accum_steps",        type=int, default=2,
                        help="Gradient accumulation steps.")
    parser.add_argument("--lr",           type=float, default=2e-5)
    parser.add_argument("--max_seq_len",  type=int,   default=1024)
    parser.add_argument("--seed",         type=int,   default=1,
                        help="Random seed for train/validation split.")
    parser.add_argument("--device_id",    type=int,   default=0,
                        help="CUDA device ID (e.g., 0, 1, 2, 3)")
    return parser.parse_args()

# ------------------------------- Main routine --------------------------------

def main() -> None:
    args = parse_args()

    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    print(f"Using CUDA device: {args.device_id}")

    torch.manual_seed(args.seed)

    # 1. Load full dataset and split into train/validation
    raw_data = load_qa_json(args.train_json)
    full_ds = Dataset.from_list([
        {"full_text": format_chat_prompt(ex["question"], ex["decomposition"]) }
        for ex in raw_data
    ])
    split = full_ds.train_test_split(test_size=0.15, seed=args.seed)
    train_ds = split["train"]
    valid_ds = split["test"]

    # 2. Tokenizer & base model (4-bit + FlashAttention2)
    model_name = "hf_models/Llama-3.1-8B-Instruct/Llama-3.1-8B-Instruct"
    tokenizer  = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map={"": 0},            # Use device 0 after CUDA_VISIBLE_DEVICES remapping
        torch_dtype=torch.bfloat16,    # bf16 on A100
        # REMOVE attn_implementation - causes conflicts with 4-bit
    )

    # Prepare model for k-bit training (critical for 4-bit + gradient checkpointing)
    from peft.utils import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)

    # 3. Attach LoRA adapters
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
    model.print_trainable_parameters()

    # 4. Tokenization function
    def tokenize_fn(batch):
        tokens = tokenizer(
            batch["full_text"],
            max_length=args.max_seq_len,
            truncation=True,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    # 5. Prepare tokenized datasets
    train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=["full_text"])
    valid_tok = valid_ds.map(tokenize_fn, batched=True, remove_columns=["full_text"])

    # 6. Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # 7. Training arguments: epoch-based eval & save
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,

        evaluation_strategy="epoch",
        save_strategy="epoch",

        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        gradient_checkpointing=True,
        report_to="none",
    )

    # 8. Trainer setup
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_tok,
        eval_dataset=valid_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 9. Train & save
    trainer.train()
    trainer.save_model(args.output_dir)  
    print(f"✅ Fine‐tuning complete – adapters & best checkpoints in {args.output_dir}")
    with open(Path(args.output_dir) / "loss_history.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(f"✅ Fine-tuning complete — artifacts saved to {args.output_dir}")

if __name__ == "__main__":
    main()

