"""
Generator Module
Handles generating responses using LLMs
"""

import os
from typing import List, Optional
from abc import ABC, abstractmethod


class BaseGenerator(ABC):
    """Abstract base class for LLM generators."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response for the given prompt."""
        pass


class OpenAIGenerator(BaseGenerator):
    """OpenAI-based text generator."""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        """
        Initialize OpenAI generator.

        Args:
            model: OpenAI model name
            api_key: OpenAI API key
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai is required. Install with: pip install openai"
            )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable."
            )

        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response using OpenAI."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return response.choices[0].message.content


class HuggingFaceGenerator(BaseGenerator):
    """HuggingFace Transformers-based generator (local, free)."""

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        max_length: int = 512,
        temperature: float = 0.7,
    ):
        """
        Initialize HuggingFace generator.

        Args:
            model_name: HuggingFace model identifier
            max_length: Maximum generation length
            temperature: Sampling temperature
        """
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
        except ImportError:
            raise ImportError(
                "transformers is required. "
                "Install with: pip install transformers torch"
            )

        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.pipe = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=max_length,
        )

    def generate(self, prompt: str) -> str:
        """Generate a response using HuggingFace model."""
        result = self.pipe(
            prompt,
            max_length=self.max_length,
            temperature=self.temperature,
            do_sample=True,
        )
        return result[0]["generated_text"]


class Generator:
    """
    Manager class for text generation.
    Provides a unified interface for different LLM providers.
    """

    DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Always base your answers on the given context. If the context doesn't contain enough information to answer the question, say so.
Be concise but thorough in your responses."""

    RAG_PROMPT_TEMPLATE = """Answer the question based on the following context:

Context:
{context}

Question: {question}

Instructions:
- Use only the information from the context above
- If the context doesn't contain the answer, say "I don't have enough information to answer this question based on the provided context."
- Cite the source when relevant
- Be concise but comprehensive

Answer:"""

    def __init__(
        self,
        provider: str = "openai",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        """
        Initialize the generator.

        Args:
            provider: LLM provider ('openai' or 'huggingface')
            model_name: Model name/identifier
            api_key: API key for paid services
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.provider = provider
        self.system_prompt = self.DEFAULT_SYSTEM_PROMPT

        if provider == "openai":
            model = model_name or "gpt-3.5-turbo"
            self.generator = OpenAIGenerator(
                model=model,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        elif provider == "huggingface":
            model = model_name or "google/flan-t5-base"
            self.generator = HuggingFaceGenerator(
                model_name=model,
                max_length=max_tokens,
                temperature=temperature,
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response."""
        sys_prompt = system_prompt or self.system_prompt

        if isinstance(self.generator, OpenAIGenerator):
            return self.generator.generate(prompt, sys_prompt)
        else:
            # For other generators, prepend system prompt
            full_prompt = f"{sys_prompt}\n\n{prompt}"
            return self.generator.generate(full_prompt)

    def generate_rag_response(
        self,
        question: str,
        context: str,
        custom_template: Optional[str] = None,
    ) -> str:
        """
        Generate a RAG response with context.

        Args:
            question: User's question
            context: Retrieved context
            custom_template: Optional custom prompt template

        Returns:
            Generated response
        """
        template = custom_template or self.RAG_PROMPT_TEMPLATE

        prompt = template.format(context=context, question=question)
        return self.generate(prompt)
