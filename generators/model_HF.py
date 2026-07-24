from typing import List, Union, Literal
import dataclasses
import os
import copy
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
# from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
import sys
sys.path.append("../")
import gpt_usage
import warnings
warnings.filterwarnings("ignore")


MessageRole = Literal["system", "user", "assistant"]

@dataclasses.dataclass()
class Message:
    role: MessageRole
    content: str


# ---------- Numerics & backend helpers ----------

def _select_dtype():
    """
    Prefer BF16 if supported by the GPU stack; otherwise FP16.
    """
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _force_sdpa_backend():
    """
    Pin attention to PyTorch SDPA to avoid JIT/ABI surprises from flash-attn/xFormers.
    """
    try:
        torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
    except Exception:
        pass
    # Allow TF32 for speed on Ada/Ampere
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def _warmup_generate(model, tokenizer, device):
    """
    Pay one-time CUDA/context/init costs so the first real call is not "hung."
    """
    try:
        with torch.inference_mode():
            t = tokenizer("Hello", return_tensors="pt").to(device)
            # Remove attn_implementation from generate call - it's only for model initialization
            model.generate(**t, max_new_tokens=1, do_sample=False)
    except Exception as e:
        # If warmup fails, continue - it's not critical
        print(f"Warning: Warmup generation failed: {e}")


class NaNInfClampProcessor(LogitsProcessor):
    """
    Guard against NaN/Inf logits that would break sampling.

    Converts NaN->0, ±Inf->±clamp, then clamps to a finite range in float32.
    """
    def __init__(self, clamp_value: float = 1e4):
        super().__init__()
        self.clamp = float(clamp_value)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = scores.to(torch.float32)
        scores = torch.nan_to_num(scores, nan=0.0, posinf=self.clamp, neginf=-self.clamp)
        scores = torch.clamp(scores, -self.clamp, self.clamp)
        return scores


# ---------- Base class with robust generation ----------

class HFModelBase:
    """
    Base for Hugging Face chat models.
    """
    def __init__(self, model_name: str, model, tokenizer, eos_token_id=None):
        self.name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id if eos_token_id is not None else self.tokenizer.eos_token_id
        self.is_chat = True
        self.model.eval()

        # Stable padding for decoder-only LMs: left pad and pad == eos
        self.tokenizer.padding_side = "left"
        if self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id



    def generate_chat(
        self,
        messages: List[Message],
        max_tokens: int = 1024,
        temperature: float = 0.2,
        num_comps: int = 1
    ) -> Union[List[str], str, None]:
        """
        Generate a chat completion.

        Inputs:
            messages: sequence of (role, content)
            max_tokens: maximum number of new tokens to generate
            temperature: sampling temperature (>0)
            num_comps: number of completions to return

        Output: str if num_comps==1 else List[str], or None if OOM
        """
        try:
            # Transformers sampling is undefined at temperature=0 → lower bound
            if temperature < 1e-4:
                temperature = 1e-4

            prompt = self.prepare_prompt(messages)

            # Encode (left padding)
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs.input_ids.to(self.model.device)
            attention_mask = inputs.attention_mask.to(self.model.device)
            input_token_count = int(input_ids.shape[1])

            # Compute available room vs model context
            max_ctx = None
            cfg = getattr(self.model, "config", None)
            if cfg is not None:
                if hasattr(cfg, "max_position_embeddings") and cfg.max_position_embeddings:
                    max_ctx = int(cfg.max_position_embeddings)
                elif hasattr(cfg, "n_positions") and cfg.n_positions:
                    max_ctx = int(cfg.n_positions)
            if max_ctx is None:
                max_ctx = 2048

            room = max_ctx - input_token_count
            max_new_tokens = max(1, min(int(max_tokens), room - 1))  # keep 1 token headroom
            if room <= 1:
                # Fall back to a small decode if the prompt nearly fills context
                max_new_tokens = 1

            # Logits guard
            logits_processor = LogitsProcessorList([NaNInfClampProcessor(clamp_value=1e4)])

            # Build generation kwargs - don't include attn_implementation here
            gen_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=True,
                temperature=float(temperature),
                top_p=0.95,
                top_k=50,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                num_return_sequences=int(num_comps),
                logits_processor=logits_processor,
            )

            with torch.inference_mode():
                outputs = self.model.generate(**gen_kwargs)

            # Token usage accounting (prompt counted per completion in your scheme)
            output_token_count = int(outputs.shape[1] - input_token_count)
            gpt_usage.prompt_tokens += input_token_count * num_comps
            gpt_usage.completion_tokens += output_token_count * num_comps

            outs = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
            outs = [self.extract_output(o, prompt) for o in outs]
            return outs[0] if len(outs) == 1 else outs

        except torch.cuda.OutOfMemoryError:
            # Clear GPU cache to help with recovery
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"OOM error in {self.name} generation, returning None")
            return None
        except RuntimeError as e:
            # Catch other CUDA/memory related runtime errors
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"Memory/CUDA error in {self.name} generation: {e}, returning None")
                return None
            else:
                # Re-raise non-memory related runtime errors
                raise e



    def prepare_prompt(self, messages: List[Message]) -> str:
        raise NotImplementedError

    def extract_output(self, output: str, prompt: str) -> str:
        raise NotImplementedError



class Qwen_1dot5B(HFModelBase):
    """
    Qwen 1.5B model using HuggingFace
    Uses the DeepSeek-R1-Distill-Qwen-1.5B model
    """
    def __init__(self):
        model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            padding_side='left'
        )
        
        # Set pad token differently from eos token
        if tokenizer.pad_token is None:
            # Try to use a different special token or add a new one
            special_tokens = tokenizer.special_tokens_map
            if '<pad>' not in tokenizer.get_vocab():
                tokenizer.add_special_tokens({'pad_token': '<pad>'})
            else:
                tokenizer.pad_token = '<pad>'
        
        # Ensure pad_token_id is different from eos_token_id
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token else tokenizer.bos_token
            if tokenizer.pad_token_id is None:
                # Add a new pad token
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Resize embeddings if we added new tokens
        model.resize_token_embeddings(len(tokenizer))
        
        super().__init__("Qwen_1.5B", model, tokenizer)

    def prepare_prompt(self, messages: List[Message]) -> str:
        """
        Prepare prompt using Qwen's chat template
        """
        # Convert messages to dict format expected by tokenizer
        chat_messages = []
        for message in messages:
            chat_messages.append({
                "role": message.role,
                "content": message.content
            })
        
        # Use the tokenizer's chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    chat_messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            except:
                # Fallback if chat template fails
                prompt = self._manual_format(messages)
        else:
            prompt = self._manual_format(messages)
        
        return prompt
    
    def _manual_format(self, messages: List[Message]) -> str:
        """Manual formatting for Qwen"""
        prompt = ""
        for message in messages:
            if message.role == "system":
                prompt += f"System: {message.content}\n\n"
            elif message.role == "user":
                prompt += f"Human: {message.content}\n\n"
            elif message.role == "assistant":
                prompt += f"Assistant: {message.content}\n\n"
        prompt += "Assistant: "
        return prompt

    def extract_output(self, output: str, prompt: str) -> str:
        """
        Extract the assistant's response from the full output
        """
        # Remove the input prompt from output
        if output.startswith(prompt):
            output = output[len(prompt):]
        
        # Clean up any special tokens
        output = output.strip()
        
        # Remove EOS token if present
        if self.tokenizer.eos_token and output.endswith(self.tokenizer.eos_token):
            output = output[:-len(self.tokenizer.eos_token)]
            
        return output.strip()



class Llama3_1_8B(HFModelBase):
    """
    Llama 3.1 8B Instruct via Hugging Face.
    - Uses repo: meta-llama/Llama-3.1-8B-Instruct
    - Normalizes rope_scaling for older transformers (expects {'type', 'factor'})
    - Forces SDPA attention backend for stability across machines
    - Chooses BF16 if supported, else FP16
    - Left padding; pad_token_id = eos_token_id
    - 1-token warm-up to pre-initialize CUDA context/kernels
    """
    def __init__(self):
        # Prefer SDPA; fall back to a local setup if helper is absent
        try:
            _force_sdpa_backend()
        except NameError:
            try:
                torch.backends.cuda.sdp_kernel(
                    enable_flash=False, enable_math=True, enable_mem_efficient=False
                )
            except Exception:
                pass
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

        # ---- Token handling (support both new `token=` and old `use_auth_token=` kw) ----
        hf_token = os.getenv("HF_TOKEN", None)
        def _load_tok():
            kw = {"padding_side": "left"}
            if hf_token:
                kw["token"] = hf_token
            try:
                return AutoTokenizer.from_pretrained(model_id, **kw)
            except TypeError:
                # Older transformers: fall back to use_auth_token
                kw.pop("token", None)
                if hf_token:
                    kw["use_auth_token"] = hf_token
                else:
                    kw["use_auth_token"] = True
                return AutoTokenizer.from_pretrained(model_id, **kw)

        def _load_cfg():
            kw = {}
            if hf_token:
                kw["token"] = hf_token
            try:
                return AutoConfig.from_pretrained(model_id, **kw)
            except TypeError:
                kw.pop("token", None)
                if hf_token:
                    kw["use_auth_token"] = hf_token
                else:
                    kw["use_auth_token"] = True
                return AutoConfig.from_pretrained(model_id, **kw)

        def _load_mdl(cfg, dtype):
            kw = {"config": cfg, "torch_dtype": dtype, "device_map": "auto"}
            if hf_token:
                kw["token"] = hf_token
            # Prefer SDPA attention kw if supported
            try:
                return AutoModelForCausalLM.from_pretrained(
                    model_id, attn_implementation="sdpa", **kw
                )
            except TypeError:
                kw.pop("attn_implementation", None)
                try:
                    return AutoModelForCausalLM.from_pretrained(model_id, **kw)
                except TypeError:
                    # Older transformers: fall back to use_auth_token
                    kw.pop("token", None)
                    if hf_token:
                        kw["use_auth_token"] = hf_token
                    else:
                        kw["use_auth_token"] = True
                    return AutoModelForCausalLM.from_pretrained(model_id, **kw)

        # ---- Load tokenizer & config ----
        tokenizer = _load_tok()
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        cfg = _load_cfg()
        # Normalize rope_scaling if config lacks 'type'
        rs = getattr(cfg, "rope_scaling", None)
        if isinstance(rs, dict) and "type" not in rs:
            # Map extended Llama-3.1 fields to a legacy-accepted schema
            factor = float(rs.get("factor", 1.0))
            cfg.rope_scaling = {"type": "dynamic", "factor": factor}

        # ---- Load model ----
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = _load_mdl(cfg, dtype)

        super().__init__("Llama-3.1-8B", model, tokenizer)

        # ---- Warm-up (use helper if present; else inline) ----
        try:
            _warmup_generate(self.model, self.tokenizer, self.model.device)
        except NameError:
            try:
                with torch.inference_mode():
                    t = self.tokenizer("Hello", return_tensors="pt").to(self.model.device)
                    # Remove attn_implementation from generate call
                    self.model.generate(**t, max_new_tokens=1, do_sample=False)
            except Exception:
                pass

    def prepare_prompt(self, messages: List[Message]) -> str:
        chat_messages = [{"role": m.role, "content": m.content} for m in messages]
        
        # Always try the tokenizer's chat template first
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    chat_messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                # Debug: print the prompt to see what it looks like
                # print(f"Chat template prompt: {repr(prompt[:200])}...")
                return prompt
            except Exception as e:
                print(f"Chat template failed: {e}, falling back to manual format")
        
        return self._manual_format(messages)

    def _manual_format(self, messages: List[Message]) -> str:
        # Don't add <|begin_of_text|> if the tokenizer already adds it
        prompt = "<|begin_of_text|>"
        for m in messages:
            if m.role == "system":
                prompt += f"<|start_header_id|>system<|end_header_id|>\n\n{m.content}<|eot_id|>"
            elif m.role == "user":
                prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{m.content}<|eot_id|>"
            elif m.role == "assistant":
                prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{m.content}<|eot_id|>"
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        print(f"Manual format prompt: {repr(prompt[:200])}...")
        return prompt

    def extract_output(self, output: str, prompt: str) -> str:
        # print(f"Raw output: {repr(output[:300])}...")
        # print(f"Prompt length: {len(prompt)}")
        
        # Handle duplicate <|begin_of_text|> issue - remove the first one
        if output.startswith("<|begin_of_text|><|begin_of_text|>"):
            output = output[len("<|begin_of_text|>"):]
            # print("Removed duplicate <|begin_of_text|> token")
        
        # Now try to remove the prompt
        if output.startswith(prompt):
            output = output[len(prompt):]
            # print("Removed prompt from output")
        else:
            # If exact match fails, try to find the assistant header
            assistant_start = "<|start_header_id|>assistant<|end_header_id|>\n\n"
            if assistant_start in output:
                idx = output.find(assistant_start)
                if idx >= 0:
                    output = output[idx + len(assistant_start):]
                    print("Found and extracted from assistant header")
        
        output = output.strip()
        
        # Trim common terminators if they appear
        for tok in ("<|eot_id|>", "<|end_of_text|>", "</s>", "<|eom_id|>"):
            if tok in output:
                output = output.split(tok)[0]
                break
        
        result = output.strip()
        # print(f"Extracted output: {repr(result[:100])}...")
        return result

    def generate_chat(
        self,
        messages: List[Message],
        max_tokens: int = 1024,
        temperature: float = 0.2,
        num_comps: int = 1
    ) -> Union[List[str], str, None]:
        """
        Generate a chat completion.

        Inputs:
            messages: sequence of (role, content)
            max_tokens: maximum number of new tokens to generate
            temperature: sampling temperature (>0)
            num_comps: number of completions to return

        Output: str if num_comps==1 else List[str], or None if OOM
        """
        try:
            # Transformers sampling is undefined at temperature=0 → lower bound
            if temperature < 1e-4:
                temperature = 1e-4

            prompt = self.prepare_prompt(messages)

            # Encode (left padding)
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs.input_ids.to(self.model.device)
            attention_mask = inputs.attention_mask.to(self.model.device)
            input_token_count = int(input_ids.shape[1])

            # Compute available room vs model context
            max_ctx = None
            cfg = getattr(self.model, "config", None)
            if cfg is not None:
                if hasattr(cfg, "max_position_embeddings") and cfg.max_position_embeddings:
                    max_ctx = int(cfg.max_position_embeddings)
                elif hasattr(cfg, "n_positions") and cfg.n_positions:
                    max_ctx = int(cfg.n_positions)
            if max_ctx is None:
                max_ctx = 2048

            room = max_ctx - input_token_count
            max_new_tokens = max(1, min(int(max_tokens), room - 1))  # keep 1 token headroom
            if room <= 1:
                # Fall back to a small decode if the prompt nearly fills context
                max_new_tokens = 1

            # Logits guard
            logits_processor = LogitsProcessorList([NaNInfClampProcessor(clamp_value=1e4)])

            # Build generation kwargs - fix parameter conflicts
            gen_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=True,
                temperature=float(temperature),
                top_p=0.95,
                top_k=50,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                num_return_sequences=int(num_comps),
                logits_processor=logits_processor,
            )

            # print(f"Generation params: max_new_tokens={max_new_tokens}, temperature={temperature}")

            with torch.inference_mode():
                outputs = self.model.generate(**gen_kwargs)

            # Token usage accounting (prompt counted per completion in your scheme)
            output_token_count = int(outputs.shape[1] - input_token_count)
            gpt_usage.prompt_tokens += input_token_count * num_comps
            gpt_usage.completion_tokens += output_token_count * num_comps

            outs = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
            outs = [self.extract_output(o, prompt) for o in outs]
            return outs[0] if len(outs) == 1 else outs

        except torch.cuda.OutOfMemoryError:
            # Clear GPU cache to help with recovery
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"OOM error in {self.name} generation, returning None")
            return None
        except RuntimeError as e:
            # Catch other CUDA/memory related runtime errors
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"Memory/CUDA error in {self.name} generation: {e}, returning None")
                return None
            else:
                # Re-raise non-memory related runtime errors
                raise e



# ---------- Mistral-7B v0.2 hardened loader ----------

class Mistral_7B(HFModelBase):
    """
    Mistral 7B Instruct v0.2 via Hugging Face.
    - Uses SDPA, stable padding, BF16/FP16 selection, and logits guard.
    """
    def __init__(self):
        _force_sdpa_backend()

        model_id = "mistralai/Mistral-7B-Instruct-v0.2"

        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        dtype = _select_dtype()
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto",
                attn_implementation="sdpa",
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto",
            )

        super().__init__("Mistral-7B-Instruct-v0.2", model, tokenizer)

        _warmup_generate(self.model, self.tokenizer, self.model.device)

    def prepare_prompt(self, messages: List[Message]) -> str:
        system_content = ""
        chat_messages = []
        for m in messages:
            if m.role == "system":
                system_content = m.content
            else:
                chat_messages.append({"role": m.role, "content": m.content})

        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                if system_content and chat_messages and chat_messages[0]["role"] == "user":
                    ua = chat_messages.copy()
                    ua[0] = {"role": "user", "content": f"{system_content}\n\n{ua[0]['content']}"}
                    return self.tokenizer.apply_chat_template(ua, tokenize=False, add_generation_prompt=True)
                return self.tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        return self._manual_format(messages)

    def _manual_format(self, messages: List[Message]) -> str:
        prompt = ""
        system_content = ""
        for m in messages:
            if m.role == "system":
                system_content = m.content
                continue
            if m.role == "user":
                if system_content and prompt == "":
                    prompt += f"[INST] {system_content}\n\n{m.content} [/INST]"
                    system_content = ""
                else:
                    prompt += f"[INST] {m.content} [/INST]"
            elif m.role == "assistant":
                prompt += f" {m.content}</s>"
        if not messages[-1].role == "assistant":
            prompt += " "
        return prompt

    def extract_output(self, output: str, prompt: str) -> str:
        if output.startswith(prompt):
            output = output[len(prompt):]
        output = output.strip()
        if "</s>" in output:
            output = output.split("</s>")[0]
        if "[/INST]" in output:
            output = output.split("[/INST]")[-1]
        return output.strip()


# ---------- Example usage ----------

if __name__ == "__main__":
    # Ensure counters exist
    if not hasattr(gpt_usage, "prompt_tokens"):
        gpt_usage.prompt_tokens = 0
    if not hasattr(gpt_usage, "completion_tokens"):
        gpt_usage.completion_tokens = 0

    msgs = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="what is 1+1 and 3*4?"),
    ]

    print("Testing Llama-3.1-8B...")
    llama = Llama3_1_8B()
    out = llama.generate_chat(msgs, temperature=0.7, max_tokens=1024)
    print("Llama:", out)
    print(f"\nToken usage — Prompt: {gpt_usage.prompt_tokens}, Completion: {gpt_usage.completion_tokens}")

    # print("\nTesting Mistral-7B v0.2...")
    # mis = Mistral_7B()
    # out = mis.generate_chat(msgs, temperature=0.7, max_tokens=64)
    # print("Mistral:", out)

    # print(f"\nToken usage — Prompt: {gpt_usage.prompt_tokens}, Completion: {gpt_usage.completion_tokens}")