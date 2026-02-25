from typing import List, Union, Optional, Literal
import dataclasses



import sys
sys.path.append('../')
import gpt_usage

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



def remove_unicode_chars(text: str) -> str:
    return re.sub(r'[^\x00-\x7F]+', '', text)

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
        _together_client = Together(
            api_key=os.environ.get("TOGETHER_API_KEY", "xxxxx"),
            timeout=60.0  # 60 second timeout to prevent hanging
        )
    return _together_client

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
        model_id = "Qwen/Qwen2.5-72B-Instruct-Turbo"
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
    

