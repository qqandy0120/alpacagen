import asyncio
from typing import Optional, Union, List, Dict
from functools import partial
import transformers
import torch
from openai import AsyncOpenAI
from abc import ABC, abstractmethod

class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    async def get_response(self, prompt: str, max_tokens: int = 1024) -> str:
        pass

class OpenAIClient(BaseLLMClient):
    """OpenAI-compatible LLM client."""
    
    def __init__(self, api_key: str = None, llm_model: str = 'gpt-4o'):
        self.client = AsyncOpenAI(api_key=api_key)
        self.llm_model = llm_model
    
    async def get_response(self, prompt: str, max_tokens: int = 1024) -> str:
        response = await self.client.completions.create(
            model=self.llm_model,
            prompt=prompt,
            max_tokens=max_tokens
        )
        return response.choices[0].text

class AzureClient(BaseLLMClient):
    """Azure OpenAI-compatible LLM client."""
    
    def __init__(self, api_key: str = None, base_url: str = None, llm_model: str = 'azure-gpt-4o'):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.llm_model = llm_model
    
    async def get_response(self, prompt: str, max_tokens: int = 1024) -> str:
        response = await self.client.completions.create(
            model=self.llm_model,
            prompt=prompt,
            max_tokens=max_tokens
        )
        return response.choices[0].text

class HuggingFaceClient(BaseLLMClient):
    """HuggingFace Transformers LLM client."""
    
    def __init__(self, llm_model: str):
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=llm_model,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
    
    async def get_response(self, prompt: str, max_tokens: int = 1024) -> str:
        # Run pipeline in a thread pool since it's synchronous
        loop = asyncio.get_event_loop()
        message = [{"role": "user", "content": prompt}]
        
        outputs = await loop.run_in_executor(
            None,
            partial(self.pipeline, message, max_tokens=max_tokens)
        )
        
        return outputs[0]["generated_text"][-1]