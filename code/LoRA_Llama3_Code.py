#!/usr/bin/env python
# finetune_llama3_lora_code.py
"""
LoRA fine-tuning of Meta-Llama-3-8B-Instruct to generate coding pitfalls
and flawed implementations.

Launch on single A100 80GB:

    python LoRA_Llama3_Code.py \
           --dataset1_path benchmarks/APP_code_datasets_llama3_8b.json \
           --dataset2_path benchmarks/augmented_coding_datasets_llama3_8b.json \
           --output_dir ./lora-llama3-8b-code \
           --num_epochs 3 \
           --per_device_batch_size 8 \
           --grad_accum_steps 4 \
           --lr 2e-5 \
           --device_id 0

To use GPU 3:
    python LoRA_Llama3_Code.py ... --device_id 3
"""

from __future__ import annotations
import argparse
import json
import torch
from pathlib import Path
from typing import List, Dict, Any
import sys
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import os
from typing import List, Dict, Union


def load_code_json(path):
    """
    Load a JSON list where each element can be:
      1. A string in format "func_sign: def foo():\ndocstring: '...'" (parsed to extract func_sign)
      2. A dict with "func_sign" and either:
         - "raw_text": list[str] or str
         - "raw_texts": list[str]
         - both "pitfalls": str and "flawed_impl": str (combined with double newlines)
    Returns each entry as {"func_sign": str, "raw_text": str} where the
    list of strings is joined by double newlines.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Top-level JSON must be a list")
    processed: List[Dict[str, str]] = []
    for ex in data:
        # Handle string-format entries (e.g., from augmented_coding_datasets.json)
        if isinstance(ex, str):
            # Parse string format: "func_sign: def foo():\ndocstring: '...'"
            # Extract func_sign by finding the first line
            lines = ex.split('\n')
            func_sign = None
            for line in lines:
                if line.startswith('func_sign:'):
                    func_sign = line.split('func_sign:', 1)[1].strip()
                    break

            if not func_sign:
                raise KeyError(f"String entry missing 'func_sign:' prefix: {ex[:100]}")

            processed.append({
                "func_sign": func_sign,
                "raw_text": ex  # Use the entire string as raw_text
            })
            continue

        # Handle dictionary-format entries
        if "func_sign" not in ex:
            raise KeyError("Missing 'func_sign' field")

        # Normalize either field into a list of strings
        if "raw_text" in ex or "raw_texts" in ex:
            k = "raw_texts" if "raw_texts" in ex else "raw_text"
            processed.append({
                "func_sign": ex["func_sign"],
                "raw_text": ex[k]
            })
        elif "pitfalls" in ex and "flawed_impl" in ex:
            # Handle datasets with separate pitfalls and flawed_impl fields
            combined_text = f"{ex['pitfalls']}\n\n{ex['flawed_impl']}"
            processed.append({
                "func_sign": ex["func_sign"],
                "raw_text": combined_text
            })
        else:
            raise KeyError("Entry must have 'raw_text', 'raw_texts', or both 'pitfalls' and 'flawed_impl' fields")

    return processed


# ───────────────────────────── Prompt helper ─────────────────────────────────
SYSTEM_PROMPT = (
    "You are an AI assistant in coding. "
    "Given a Python function signature and docstring, list potential pitfalls and "
    "provide no more than 6 flawed implementations."
)

def format_prompt(func_sign: str, raw_text: str | None = None) -> str:
    user_block = (
        "[INST] <<SYS>> "
        f"{SYSTEM_PROMPT} <</SYS>>\n\n"
        f"FUNC_SIGNATURE:\n{func_sign.strip()}\n\n"
        "[/INST]"
    )
    return (
        f"<s>{user_block}" if raw_text is None
        else f"<s>{user_block}\n{raw_text.strip()}</s>"
    )

# ───────────────────────────── Argument parser ───────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("LoRA code-pitfall fine-tuner")
    p.add_argument("--dataset1_path", required=True,
                   help="First JSON file")
    p.add_argument("--dataset2_path", required=True,
                   help="Second JSON file")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--per_device_batch_size", type=int, default=8)
    p.add_argument("--grad_accum_steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--seed", type=int, default=1,
                   help="RNG seed for shuffle/split")
    p.add_argument("--device_id", type=int, default=0,
                   help="CUDA device ID (e.g., 0, 1, 2, 3)")
    return p.parse_args()

# ─────────────────────────────── Main entry ──────────────────────────────────
def main() -> None:
    args = parse_args()

    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    print(f"Using CUDA device: {args.device_id}")

    # 1. Dataset preparation ── concat → shuffle → 85 % train / 15 % valid
    all_examples = (
        load_code_json(args.dataset1_path) +
        load_code_json(args.dataset2_path)
    )
    
    full_ds = Dataset.from_list([
        {"full_text": format_prompt(ex["func_sign"], ex["raw_text"])}
        for ex in all_examples
    ]).shuffle(seed=args.seed)
    split = full_ds.train_test_split(test_size=0.15, seed=args.seed)
    train_ds, valid_ds = split["train"], split["test"]

    # 2. Tokeniser + base model (4-bit QLoRA)
    # mdl_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    mdl_name = "hf_models/Llama-3.1-8B-Instruct/Llama-3.1-8B-Instruct"
    # mdl_name ="model/models--Qwen--Qwen2-0.5B/snapshots/91d2aff3f957f99e4c74c962f2f408dcc88a18d8"
    tok = AutoTokenizer.from_pretrained(mdl_name, use_fast=True)
    tok.pad_token = tok.eos_token
    
    
    # model = AutoModelForCausalLM.from_pretrained(
    #     mdl_name,
    #     device_map="auto",
    #     torch_dtype="auto",            # bf16 on A100
    #     load_in_4bit=True,
    #     attn_implementation="sdpa",
    # )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # Match your bf16 training
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        mdl_name,
        quantization_config=quantization_config,
        device_map={"": 0},  # Use device 0 after CUDA_VISIBLE_DEVICES remapping
        torch_dtype=torch.bfloat16,
        # REMOVE attn_implementation="sdpa" - causes conflicts with 4-bit
    )

    # ADD THIS BEFORE TRAINING (critical for 4-bit + gradient checkpointing):
    from peft.utils import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)  # ← MUST ADD THIS

    # 3. Attach LoRA adapters
    lora_cfg = LoraConfig(
        r=64, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, 
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_cfg)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.print_trainable_parameters()

    # 4. Tokenise on-the-fly
    def tok_fn(batch):
        t = tok(batch["full_text"],
                max_length=args.max_seq_len,
                truncation=True,
                padding="max_length")
        t["labels"] = t["input_ids"].copy()
        return t

    train_tok = train_ds.map(tok_fn, batched=True, remove_columns=["full_text"])
    valid_tok = valid_ds.map(tok_fn, batched=True, remove_columns=["full_text"])

    # 5. Data collator
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    # 6. TrainingArguments – epoch-based eval / save
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
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=5,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        report_to="none",
    )

    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=valid_tok,
        tokenizer=tok,
        data_collator=collator,
    )

    # 8. Train & save
    trainer.train()
    trainer.save_model(args.output_dir)
    with open(Path(args.output_dir) / "loss_history.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(f"✅ Fine-tuning complete — artifacts saved to {args.output_dir}")

if __name__ == "__main__":
    main()

