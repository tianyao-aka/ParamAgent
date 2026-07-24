from typing import List, Union, Optional, Literal
import dataclasses



import sys
sys.path.append('../')
import gpt_usage
from termcolor import colored

#claude specific imports
import boto3
from botocore.exceptions import ClientError

from tenacity import (
    retry,
    stop_after_attempt,  # type: ignore
    wait_random_exponential,  # type: ignore
)
from openai import OpenAI
from together import Together
# NEW: DashScope native SDK
import os
from http import HTTPStatus
import dashscope
from dashscope import Generation
import re


# Singapore/International endpoint; switch to the Beijing endpoint if needed.
# dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"
dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"  # <-- use this if you are in CN region

# Pick up the API key from environment.
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")


def remove_unicode_chars(text: str) -> str:
    return re.sub(r'[^\x00-\x7F]+', '', text)

import anthropic
anthropic_client = anthropic.Anthropic()

# Initialize OpenAI client lazily to avoid requiring API key when not needed
_openai_client = None

def get_openai_client():
    """Lazy initialization of OpenAI client"""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client

# Together client singleton to avoid repeated instantiation (prevents memory leak)
_together_client = None

def get_together_client():
    """Lazy initialization of Together client"""
    global _together_client
    if _together_client is None:
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise RuntimeError("Set TOGETHER_API_KEY before using TogetherAI models.")
        _together_client = Together(
            api_key=api_key,
            timeout=60.0  # 60 second timeout to prevent hanging
        )
    return _together_client

# br_client = boto3.client("bedrock-runtime", region_name="us-west-2")

# client for us-east-1
br_client = boto3.client("bedrock-runtime", region_name="us-west-2")

MessageRole = Literal["system", "user", "assistant"]

@dataclasses.dataclass()
class Message():
    role: MessageRole
    content: str


def message_to_str(message: Message) -> str:
    return f"{message.role}: {message.content}"


def messages_to_str(messages: List[Message]) -> str:
    return "\n".join([message_to_str(message) for message in messages])



@retry(wait=wait_random_exponential(min=1, max=180), stop=stop_after_attempt(6))
def aliyun_chat(
    model: str,
    messages: List[Message],
    max_tokens: int = 1024,
    temperature: float = 0.2,
    num_comps: int = 1,
) -> Union[List[str], str]:
    """
    Calls Alibaba DashScope Generation API in 'message' format.

    Args:
        model: DashScope model id (e.g., "qwen2.5-1.5b-instruct").
        messages: List[Message] with roles in {"system","user","assistant"}.
        max_tokens: Max output tokens (DashScope supports max_tokens).
        temperature: Sampling temperature.
        num_comps: Number of completions to return (looped client-side).

    Returns:
        str if num_comps==1 else List[str].
    """
    ds_messages = [{"role": m.role, "content": m.content} for m in messages]
    outs: List[str] = []
    for _ in range(num_comps):
        resp = Generation.call(
            model=model,
            messages=ds_messages,
            result_format="message",
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        if getattr(resp, "status_code", None) != HTTPStatus.OK:
            raise RuntimeError(
                f"DashScope error: status={getattr(resp,'status_code',None)} "
                f"code={getattr(resp,'code',None)} msg={getattr(resp,'message',None)}"
            )

        # token usage accounting (best-effort; keys may vary by SDK version)
        try:
            usage = getattr(resp, "usage", {}) or {}
            gpt_usage.completion_tokens += int(usage.get("output_tokens", 0))
            gpt_usage.prompt_tokens += int(usage.get("input_tokens", 0))
        except Exception:
            pass

        outs.append(resp.output["choices"][0]["message"]["content"])
    return outs[0] if num_comps == 1 else outs


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def gpt_completion(
        model: str,
        prompt: str,
        max_tokens: int = 1024,
        stop_strs: Optional[List[str]] = None,
        temperature: float = 0.0,
        num_comps=1,
) -> Union[List[str], str]:
    client = get_openai_client()
    response = client.completions.create(model=model,
    prompt=prompt,
    temperature=temperature,
    max_tokens=max_tokens,
    top_p=1,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=stop_strs,
    n=num_comps)
    
    # update token usage
    gpt_usage.completion_tokens += response.usage.completion_tokens
    gpt_usage.prompt_tokens += response.usage.prompt_tokens
    
    if num_comps == 1:
        return response.choices[0].text  # type: ignore

    return [choice.text for choice in response.choices]  # type: ignore


@retry(wait=wait_random_exponential(min=1, max=180), stop=stop_after_attempt(6))
def gpt_chat(
    model: str,
    messages: List[Message],
    max_tokens: int = 1024,
    temperature: float = 0.0,
    num_comps=1,
) -> Union[List[str], str]:
    client = get_openai_client()

    if model == "o1-mini" or model == "o1-preview":
        m = [dataclasses.asdict(message) for message in messages]

        for message in m:
            if message["role"] == "system":
                message["role"] = "assistant"

            message['content'] = remove_unicode_chars(message['content'])

        response = client.chat.completions.create(model=model,
        messages=m,
        n=num_comps)

    elif model == "gpt-5-mini":
        # gpt-5-mini requires max_completion_tokens instead of max_tokens
        # and only supports temperature=1 (default)
        response = client.chat.completions.create(model=model,
        messages=[dataclasses.asdict(message) for message in messages],
        max_completion_tokens=max_tokens,
        n=num_comps)

    else:
        response = client.chat.completions.create(model=model,
        messages=[dataclasses.asdict(message) for message in messages],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1.,
        frequency_penalty=0.,
        presence_penalty=0.,
        n=num_comps)

    print(response.choices[0].message.content)

    # update token usage
    gpt_usage.completion_tokens += response.usage.completion_tokens
    gpt_usage.prompt_tokens += response.usage.prompt_tokens
    
    
    if num_comps == 1:
        return response.choices[0].message.content  # type: ignore

    return [choice.message.content for choice in response.choices]  # type: ignore


@retry(wait=wait_random_exponential(min=1, max=180), stop=stop_after_attempt(6))
def claude_chat(
    model: str,
    messages: List[Message],
    max_tokens: int = 1024,
    temperature: float = 1.0, #0.0,
    num_comps=1,
) -> Union[List[str], str]: 
    
    #pre-process messages
    system_prompts = None
    user_messages = []
    for message in messages:
        if message.role == 'system':
            if system_prompts is None:
                system_prompts = [{"text": message.content}]
            else:
                system_prompts.append([{"text": message.content}])
        
        else:
            msg = {"role": message.role,
                    "content": [{"text": message.content}]}
            user_messages.append(msg)
    
    response = br_client.converse(
        modelId=model,
        messages=user_messages,
        system=system_prompts,
        inferenceConfig={
                        "maxTokens": max_tokens,
                        "temperature": temperature,
                        "topP": 0.9
                        },
    )

    # update token usage for Claude -- Pending
    gpt_usage.completion_tokens += response['usage']['outputTokens']
    gpt_usage.prompt_tokens += response['usage']['inputTokens']
    
    return response["output"]["message"]["content"][0]["text"] 


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def together_chat(
    model: str,
    messages: List['Message'],  # assuming 'Message' has fields: role, content
    max_tokens: int = 1024,
    temperature: float = 0.1,
    num_comps: int = 1,
) -> Union[List[str], str]:
    
    """
    Chat interface for Together AI models, with retry mechanism.

    Args:
        model (str): Together model identifier.
        messages (List[Message]): Chat history with `role` and `content`.
        max_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature.
        num_comps (int): Number of completions to return (1 = default).

    Returns:
        Union[List[str], str]: Model response(s) as string or list of strings.
    """
    
    # Use singleton client to avoid repeated instantiation (prevents memory leak)
    client = get_together_client()

    # Convert internal Message format to OpenAI-style schema
    converted_messages = [
        {"role": message.role, "content": message.content} for message in messages
    ]

    # Call Together chat API
    response = client.chat.completions.create(
        model=model,
        messages=converted_messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        n=num_comps,  # Request multiple completions if needed
    )

    # Update token usage if needed (example only, replace with your global tracking logic)
    if hasattr(response, "usage"):
        gpt_usage.prompt_tokens += response.usage.prompt_tokens
        gpt_usage.completion_tokens += response.usage.completion_tokens

    # Extract completions with None checking and diagnostics
    completions = []
    for idx, choice in enumerate(response.choices):
        content = choice.message.content
        if content is None:
            print(f"DEBUG: Together AI returned None content for choice {idx} with model {model}")
            print(f"DEBUG: Response object: {response}")
            completions.append(None)
        else:
            completions.append(content)

    # Debug: Log first 200 chars of first completion
    if completions and completions[0]:
        print(f"DEBUG: Together AI response preview (first 200 chars): {completions[0][:200]}")
    elif completions:
        print(f"DEBUG: Together AI returned empty or None completion")

    if num_comps == 1:
        return completions[0]
    return completions


@retry(wait=wait_random_exponential(min=1, max=180), stop=stop_after_attempt(6))
def anthropic_chat(
    model: str,
    messages: List[Message],
    max_tokens: int = 1024,
    temperature: float = 0.0,
    num_comps=1,
    cache_hits: int = 0,
) -> Union[List[str], str]:
    
    #pre-process messages
    system_prompts = None
    user_messages = []
    for message in messages:
        if message.role == 'system':
            if system_prompts is None:
                # if cache_hits <= 4:
                #     system_prompts = [{"type": "text", "text": message.content, "cache_control": {"type": "ephemeral"}}]
                # else:
                system_prompts = [{"type": "text", "text": message.content}]
            else:
                # if cache_hits <= 4:
                #     system_prompts.append({"type": "text", "text": message.content, "cache_control": {"type": "ephemeral"}})
                # else:
                system_prompts.append({"type": "text", "text": message.content})
        
        else:
            # if cache_hits <= 4:
            #     msg = {"role": message.role,
            #             "content": [{"type": "text", "text": message.content, "cache_control": {"type": "ephemeral"}}]}
            # else:
            #     msg = {"role": message.role,
            #             "content": [{"type": "text", "text": message.content}]}
            
            msg = {"role": message.role,
                    "content": [{"type": "text", "text": message.content}]}
                
            user_messages.append(msg)
            

    response = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        temperature=temperature,
        system=system_prompts,
        messages=user_messages,
    )

    # update token usage for Claude -- Pending
    gpt_usage.completion_tokens += response.usage.output_tokens
    gpt_usage.prompt_tokens += response.usage.input_tokens
    # gpt_usage.cache_creation_input_tokens += response.usage.cache_creation_input_tokens
    # gpt_usage.cache_read_input_tokens += response.usage.cache_read_input_tokens
    
    print(f"cache hits: {cache_hits}")
    print(colored(f"API usage: {response.usage}", 'green'))
    
    return response.content[0].text

class ModelBase():
    def __init__(self, name: str):
        self.name = name
        self.is_chat = False

    def __repr__(self) -> str:
        return f'{self.name}'

    def generate_chat(self, messages: List[Message], max_tokens: int = 1024, temperature: float = 0.2, num_comps: int = 1) -> Union[List[str], str]:
        raise NotImplementedError

    def generate(self, prompt: str, max_tokens: int = 1024, stop_strs: Optional[List[str]] = None, temperature: float = 0.0, num_comps=1) -> Union[List[str], str]:
        raise NotImplementedError


class GPTChat(ModelBase):
    def __init__(self, model_name: str):
        self.name = model_name
        self.is_chat = True

    def generate_chat(self, messages: List[Message], max_tokens: int = 1024, temperature: float = 0.2, num_comps: int = 1) -> Union[List[str], str]:
        return gpt_chat(self.name, messages, max_tokens, temperature, num_comps)



class TogetherAIChat(ModelBase):
    """
    A ModelBase subclass for interacting with Together AI chat models.

    Args:
        model_name (str): Identifier of the Together AI model (e.g., "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo").
    """
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.is_chat = True
        self.cache_hit_ctr = 0  # total hits for prompt caching

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        stop_strs: Optional[List[str]] = None,
        temperature: float = 0.0,
        num_comps: int = 1
    ) -> Union[List[str], str]:
        """
        Single-turn completion API (non-chat). Not implemented for pure chat models.
        
        Args:
            prompt (str): The input prompt string.
            max_tokens (int): Maximum number of tokens to generate.
            stop_strs (List[str], optional): List of stop sequences.
            temperature (float): Sampling temperature.
            num_comps (int): Number of completions to return.
        
        Returns:
            Union[List[str], str]: The generated completion(s).
        """
        raise NotImplementedError("TogetherAIChat does not support single-turn completions; use generate_chat() instead.")

    def generate_chat(
        self,
        messages: List[Message],
        max_tokens: int = 1024,
        temperature: float = 0.2,
        num_comps: int = 1
    ) -> Union[List[str], str]:
        """
        Multi-turn chat interface using Together AI.

        Args:
            messages (List[Message]): Sequence of Message(role, content) dicts or objects.
            max_tokens (int): Maximum number of tokens to generate in the response.
            temperature (float): Sampling temperature.
            num_comps (int): Number of chat completions to return.

        Returns:
            Union[List[str], str]: The chat response(s) as string or list of strings.
        """
        # call the together_chat wrapper you defined earlier
        return together_chat(
            model=self.name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            num_comps=num_comps
        )


class ClaudeChat(ModelBase):
    """_summary_

    Args:
        ModelBase (_type_): _description_
    """
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.is_chat = True
        self.cache_hit_ctr = 0 #log total hits for prompt caching

    def generate(self, prompt: str, max_tokens: int = 1024, stop_strs: Optional[List[str]] = None, temperature: float = 0.0, num_comps=1) -> Union[List[str], str]:
        print("invoked")
        raise NotImplementedError

    def generate_chat(self, messages: List[Message], max_tokens: int = 1024, temperature: float = 0.2, num_comps: int = 1) -> Union[List[str], str]:
        return claude_chat(self.name, messages, max_tokens, temperature, num_comps)
        
        # # for anthropic client
        # self.cache_hit_ctr += 0
        # return anthropic_chat(self.name, messages, max_tokens, temperature, num_comps, cache_hits=self.cache_hit_ctr)

class Sonnet3(ClaudeChat):
    def __init__(self):
        model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
        super().__init__(model_id)


class Sonnet35(ClaudeChat):
    def __init__(self):
        model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
        super().__init__(model_id)


class Llama3_1_405B(TogetherAIChat):
    def __init__(self):
        model_id = "meta-llama/Llama-3.1-405B-Instruct-Turbo"
        super().__init__(model_id)


class Llama3_1_70B(TogetherAIChat):
    def __init__(self):
        model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
        super().__init__(model_id)
        

class Llama3_1_8B(TogetherAIChat):
    def __init__(self):
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        super().__init__(model_id)


class Mistral_7B(TogetherAIChat):
    def __init__(self):
        model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        super().__init__(model_id)


class Qwen_7B(TogetherAIChat):
    def __init__(self):
        model_id = "Qwen/Qwen2.5-7B-Instruct-Turbo"
        super().__init__(model_id)


class Qwen3_70B(TogetherAIChat):
    def __init__(self):
        model_id = "Qwen/Qwen3-Next-80B-A3B-Instruct"
        super().__init__(model_id)


class Qwen_1dot5B(TogetherAIChat):
    def __init__(self):
        # client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        # endpoint = client.endpoints.create(
        #     model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        #     hardware="1x_nvidia_h100_80gb_sxm",
        #     min_replicas=1,
        #     max_replicas=1,
        #     display_name="tianjun-deepseek-r1-qwen-1_5b"
        # )
        # model_id = endpoint.name
        # print ('qwen endpoint name:',model_id)
        # model_id = "arize-ai/qwen-2-1.5b-instruct"
        model_id = "Qwen/Qwen2.5-72B-Instruct-Turbo"
        super().__init__(model_id)


class TianjunLlama3_8B_Lora_Direct_Sol(TogetherAIChat):
    def __init__(self):
        model_id = "tianjun/llama3-8b-lora_direct_sol"
        super().__init__(model_id)


class Qwen2_1dot5B(TogetherAIChat):
    def __init__(self):
        model_id = "arize-ai/qwen-2-1.5b-instruct"
        super().__init__(model_id)


# class Qwen_1dot5B(ModelBase):
#     def __init__(self, model_name: str = "deepseek-r1-distill-qwen-1.5b"):
#         super().__init__(model_name)
#         self.is_chat = True
#         print (f"using qwen version: deepseek-r1-distill-qwen-1.5b")
#         # If you actually want DeepSeek-R1-Distill-Qwen via Model Studio instead:
#         # self.name = "deepseek-r1-distill-qwen-1.5b"

#     def generate_chat(
#         self,
#         messages: List[Message],
#         max_tokens: int = 1024,
#         temperature: float = 0.2,
#         num_comps: int = 1,
#     ) -> Union[List[str], str]:
#         return aliyun_chat(self.name, messages, max_tokens, temperature, num_comps)


class Llama2_7B(TogetherAIChat):
    def __init__(self):
        print ('current model name is incorrect for Llama2_7B')
        model_id = "togethercomputer/llama-2-7b-chat"
        super().__init__(model_id)


class GPT4(GPTChat):
    def __init__(self):
        super().__init__("gpt-4")


class GPT4o(GPTChat):
    """
    Added GPT4o
    """
    def __init__(self):
        super().__init__("gpt-4o")


class GPT4oMini(GPTChat):
    """
    Added GPT4o-mini
    """
    def __init__(self):
        super().__init__("gpt-4o-mini")


class GPT5Mini(GPTChat):
    """
    GPT-5-mini model wrapper
    """
    def __init__(self):
        super().__init__("gpt-5-mini")


class o1(GPTChat):
    """
    GPT o1
    """
    def __init__(self):
        super().__init__("o1-preview")


class o1mini(GPTChat):
    """
    GPT o1
    """
    def __init__(self):
        super().__init__("o1-mini")


class GPT4turbo(GPTChat):
    """
    Added GPT4-turbo
    """
    def __init__(self):
        super().__init__("gpt-4-turbo")


class GPT35(GPTChat):
    def __init__(self):
        super().__init__("gpt-3.5-turbo")


class GPTDavinci(ModelBase):
    def __init__(self, model_name: str):
        self.name = model_name

    def generate(self, prompt: str, max_tokens: int = 1024, stop_strs: Optional[List[str]] = None, temperature: float = 0, num_comps=1) -> Union[List[str], str]:
        return gpt_completion(self.name, prompt, max_tokens, stop_strs, temperature, num_comps)


class HFModelBase(ModelBase):
    """
    Base for huggingface chat models
    """

    def __init__(self, model_name: str, model, tokenizer, eos_token_id=None):
        self.name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id if eos_token_id is not None else self.tokenizer.eos_token_id
        self.is_chat = True

    def generate_chat(self, messages: List[Message], max_tokens: int = 1024, temperature: float = 0.2, num_comps: int = 1) -> Union[List[str], str]:
        # NOTE: HF does not like temp of 0.0.
        if temperature < 0.0001:
            temperature = 0.0001

        prompt = self.prepare_prompt(messages)

        outputs = self.model.generate(
            prompt,
            max_new_tokens=min(
                max_tokens, self.model.config.max_position_embeddings),
            use_cache=True,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            eos_token_id=self.eos_token_id,
            num_return_sequences=num_comps,
        )

        outs = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        assert isinstance(outs, list)
        for i, out in enumerate(outs):
            assert isinstance(out, str)
            outs[i] = self.extract_output(out)

        if len(outs) == 1:
            return outs[0]  # type: ignore
        else:
            return outs  # type: ignore

    def prepare_prompt(self, messages: List[Message]):
        raise NotImplementedError

    def extract_output(self, output: str) -> str:
        raise NotImplementedError


class StarChat(HFModelBase):
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceH4/starchat-beta",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceH4/starchat-beta",
        )
        super().__init__("starchat", model, tokenizer, eos_token_id=49155)

    def prepare_prompt(self, messages: List[Message]):
        prompt = ""
        for i, message in enumerate(messages):
            prompt += f"<|{message.role}|>\n{message.content}\n<|end|>\n"
            if i == len(messages) - 1:
                prompt += "<|assistant|>\n"

        return self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

    def extract_output(self, output: str) -> str:
        out = output.split("<|assistant|>")[1]
        if out.endswith("<|end|>"):
            out = out[:-len("<|end|>")]

        return out


class CodeLlama(HFModelBase):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    def __init__(self, version: Literal["34b", "13b", "7b"] = "34b"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            f"codellama/CodeLlama-{version}-Instruct-hf",
            add_eos_token=True,
            add_bos_token=True,
            padding_side='left'
        )
        model = AutoModelForCausalLM.from_pretrained(
            f"codellama/CodeLlama-{version}-Instruct-hf",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        super().__init__("codellama", model, tokenizer)

    def prepare_prompt(self, messages: List[Message]):
        if messages[0].role != "system":
            messages = [
                Message(role="system", content=self.DEFAULT_SYSTEM_PROMPT)
            ] + messages
        messages = [
            Message(role=messages[1].role, content=self.B_SYS +
                    messages[0].content + self.E_SYS + messages[1].content)
        ] + messages[2:]
        assert all([msg.role == "user" for msg in messages[::2]]) and all(
            [msg.role == "assistant" for msg in messages[1::2]]
        ), (
            "model only supports 'system', 'user' and 'assistant' roles, "
            "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
        )
        messages_tokens: List[int] = sum(
            [
                self.tokenizer.encode(
                    f"{self.B_INST} {(prompt.content).strip()} {self.E_INST} {(answer.content).strip()} ",
                )
                for prompt, answer in zip(
                    messages[::2],
                    messages[1::2],
                )
            ],
            [],
        )
        assert messages[-1].role == "user", f"Last message must be from user, got {messages[-1].role}"
        messages_tokens += self.tokenizer.encode(
            f"{self.B_INST} {(messages[-1].content).strip()} {self.E_INST}",
        )
        # remove eos token from last message
        messages_tokens = messages_tokens[:-1]
        import torch
        return torch.tensor([messages_tokens]).to(self.model.device)

    def extract_output(self, output: str) -> str:
        out = output.split("[/INST]")[-1].split("</s>")[0].strip()
        return out


class GPT_OSS_20B(TogetherAIChat):
    def __init__(self):
        model_id = "openai/gpt-oss-20b"
        super().__init__(model_id)


if __name__ == "__main__":
    model = Qwen_1dot5B()
    # Create a simple test message
    test_messages = [
        Message(role="user", content="hello there")
    ]
    
    print(f"Testing model: {model.name}")
    print(f"Input message: {test_messages[0].content}")
    print("-" * 50)

    # Generate response using the chat interface
    response = model.generate_chat(
        messages=test_messages,
        max_tokens=100,
        temperature=0.2,
        num_comps=1
    )
    
    
    print(f"Model response: {response}")
    
