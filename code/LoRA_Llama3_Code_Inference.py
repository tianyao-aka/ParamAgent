#!/usr/bin/env python
# lora_inference.py
"""
Inference script for LoRA-fine-tuned Meta-Llama-3.1-8B-Instruct model.
Defines an AI agent that, given a Python function signature and docstring,
returns a list of potential pitfalls and a flawed implementation.
"""

from __future__ import annotations
import torch
from peft import PeftModel
import jsonlines
import os
from typing import List, Dict
import re
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel
import jsonlines
from tqdm import tqdm


def read_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File `{path}` does not exist.")
    elif not path.endswith(".jsonl"):
        raise ValueError(f"File `{path}` is not a jsonl file.")
    items = []
    with jsonlines.open(path) as reader:
        for item in reader:
            items += [item]
    return items


class CodePitfallAgent:
    """
    AI agent that generates coding pitfalls and flawed implementations
    from a function signature and docstring.
    """

    def __init__(
        self,
        base_model: str,
        lora_path: str,
        device: str = "cuda:0",
    ) -> None:
        """
        Load the base model, apply a 4-bit LoRA adapter, and optimize for RTX 4090.

        Args:
            base_model (str): HF identifier of the base Llama model.
            lora_path (str): Path to the directory containing LoRA weights.
            device (str): Torch device (e.g., "cuda:0" or "cpu").
        """
        self.device = torch.device(device)

        # 1. Enable TF32 on Ampere (e.g. RTX 4090) for faster matmuls
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # 2. Configure 4-bit quantization to compute in BF16 Tensor Cores
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        # 3. Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 4. Load quantized base model (quantization_config handles device placement)
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_cfg,
            torch_dtype=torch.bfloat16,
            # low_cpu_mem_usage=True,
            device_map={"": 0}
        )

        # 5. Apply LoRA weights
        self.model = PeftModel.from_pretrained(
            base,
            lora_path,
            torch_dtype=torch.bfloat16,
        )
        # self.model.to(self.device)
        self.model.eval()

        # 6. JIT-compile for faster generation (30-50% speedup after warmup)
        # DISABLED: Skipping torch.compile to avoid long first-run compilation
        # if hasattr(torch, "compile"):
        #     print("Enabling torch.compile - first generation will be slow (compiling), then much faster...")
        #     self.model = torch.compile(self.model, mode="reduce-overhead", backend="inductor")

    def generate(self, func_sign: str, temperature: float = 0.7) -> str:
        """
        Generate coding pitfalls and flawed implementations for a function.

        Args:
            func_sign (str): The Python function signature and docstring.
            temperature (float): Sampling temperature (higher → more diversity).

        Returns:
            str: The model's generated text.
        """
        system = (
            "[INST] <<SYS>> You are an AI assistant in coding. "
            "Given a Python function signature and docstring, list potential pitfalls "
            "and provide no more than 6 flawed implementations. <</SYS>>\n\n"
        )
        user = f"FUNC_SIGNATURE:\n{func_sign.strip()} [/INST]"
        prompt = f"<s>{system}{user}"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(self.device)

        # Use inference_mode for maximal throughput
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=900,
                do_sample=True,
                temperature=temperature,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        reply = generated.split("[/INST]", 1)[-1].strip()

        # Truncate trailing incomplete code fences
        fence_pos = reply.rfind("```")
        if fence_pos != -1:
            reply = reply[: fence_pos + 3]
        else:
            reply = re.sub(r"\s*(?:</s>|\[\/INST\])\s*", "", reply)

        return reply.strip()

    def generate_batch(self, func_signs: List[str], temperature: float = 0.7, batch_size: int = 4) -> List[str]:
        """
        Generate coding pitfalls for multiple function signatures in batches.
        This is 2-4x faster than processing one at a time.

        Args:
            func_signs (List[str]): List of Python function signatures and docstrings.
            temperature (float): Sampling temperature (higher → more diversity).
            batch_size (int): Number of samples to process in parallel (default: 4).

        Returns:
            List[str]: List of model-generated texts.
        """
        results = []

        # Process in batches
        for i in range(0, len(func_signs), batch_size):
            batch = func_signs[i:i + batch_size]

            # Format all prompts in the batch
            prompts = []
            for func_sign in batch:
                system = (
                    "[INST] <<SYS>> You are an AI assistant in coding. "
                    "Given a Python function signature and docstring, list potential pitfalls "
                    "and provide no more than 6 flawed implementations. <</SYS>>\n\n"
                )
                user = f"FUNC_SIGNATURE:\n{func_sign.strip()} [/INST]"
                prompt = f"<s>{system}{user}"
                prompts.append(prompt)

            # Tokenize batch
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(self.device)

            # Generate for entire batch
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=600,  # Reduced from 900 for faster generation
                    do_sample=True,
                    temperature=temperature,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode each output in the batch
            for output in outputs:
                generated = self.tokenizer.decode(output, skip_special_tokens=False)
                reply = generated.split("[/INST]", 1)[-1].strip()

                # Truncate trailing incomplete code fences
                fence_pos = reply.rfind("```")
                if fence_pos != -1:
                    reply = reply[: fence_pos + 3]
                else:
                    reply = re.sub(r"\s*(?:</s>|\[\/INST\])\s*", "", reply)

                results.append(reply.strip())

        return results

# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("CodePitfallAgent Inference")
    parser.add_argument(
        "--base_model", default= "hf_models/Llama-3.1-8B-Instruct/Llama-3.1-8B-Instruct",
        help="Base model identifier (e.g., meta-llama/Meta-Llama-3.1-8B-Instruct)"
    )
    parser.add_argument(
        "--lora_path", default="lora-llama3-8b-code/checkpoint-621/",
        help="Path to LoRA adapter directory"
    )
    parser.add_argument(
        "--device", default="cuda:0",
        help="Torch device (cuda or cpu)"
    )
    args = parser.parse_args()

    agent = CodePitfallAgent(
        base_model=args.base_model,
        lora_path=args.lora_path,
        device=args.device,
    )

    f = "benchmarks/humaneval_full.jsonl"
    save_path = "benchmarks/code_pitfalls/humaneval_full_pitfalls_llama3_8b.jsonl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    val = read_jsonl(f)
    N = len(val)
    # (Re)initialize the output file
    with open(save_path, 'w') as _:
        pass

    for i in tqdm(range(N), desc="Processing prompts"):
        p = val[i]['prompt']

        # generate and attach pitfalls
        # val[i]['pitfall'] = agent.generate(p)  # Disabled: using batch processing instead

        # Generate 8 high-temperature versions using batch processing (batch_size=5)
        prompts_batch = [p] * 8
        val[i]['high_temp_pitfall'] = agent.generate_batch(
            prompts_batch,
            temperature=0.7,
            batch_size=4
        )
        print ('ok')

        # append this item to disk immediately
        with jsonlines.open(save_path, mode='a') as writer:
            writer.write(val[i])



    # f = "benchmarks/mbpp-py.jsonl"
    # save_path = "benchmarks/code_pitfalls/mbpp_pitfalls.jsonl"
    # val = read_jsonl(f)
    # N = len(val)
    # # (Re)initialize the output file
    # with open(save_path, 'w') as _:
    #     pass

    # for i in tqdm(range(N), desc="Processing prompts"):
    #     p = val[i]['prompt']

    #     # generate and attach pitfalls
    #     val[i]['pitfall'] = agent.generate(p)
    #     val[i]['high_temp_pitfall'] = [
    #         agent.generate(p, temperature=1.0)
    #         for _ in range(10)
    #     ]

    #     # append this item to disk immediately
    #     with jsonlines.open(save_path, mode='a') as writer:
    #         writer.write(val[i])

