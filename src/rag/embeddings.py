"""
Embeddings Module
Handles converting text to vector embeddings
"""

import os
from typing import List, Optional
from abc import ABC, abstractmethod


class BaseEmbedding(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text."""
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI embedding model."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ):
        """
        Initialize OpenAI embeddings.

        Args:
            model: OpenAI embedding model name
            api_key: OpenAI API key (uses env var if not provided)
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai is required. Install with: pip install openai"
            )

        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable."
            )

        self.client = OpenAI(api_key=self.api_key)

        # Model dimensions
        self._dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text using OpenAI."""
        response = self.client.embeddings.create(
            input=text,
            model=self.model,
        )
        return response.data[0].embedding

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts using OpenAI."""
        response = self.client.embeddings.create(
            input=texts,
            model=self.model,
        )
        return [item.embedding for item in response.data]

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimensions.get(self.model, 1536)


class SentenceTransformerEmbedding(BaseEmbedding):
    """Sentence Transformers embedding model (local, free)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize Sentence Transformers embeddings.

        Args:
            model_name: Name of the sentence transformer model
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text."""
        embedding = self.model.encode(text)
        return embedding.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension


class EmbeddingManager:
    """
    Manager class for handling embeddings.
    Provides a unified interface for different embedding providers.
    """

    def __init__(
        self,
        provider: str = "sentence_transformer",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the embedding manager.

        Args:
            provider: Embedding provider ('openai' or 'sentence_transformer')
            model_name: Model name/identifier
            api_key: API key for paid services
        """
        self.provider = provider

        if provider == "openai":
            model = model_name or "text-embedding-3-small"
            self.embedder = OpenAIEmbedding(model=model, api_key=api_key)
        elif provider == "sentence_transformer":
            model = model_name or "all-MiniLM-L6-v2"
            self.embedder = SentenceTransformerEmbedding(model_name=model)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text."""
        return self.embedder.embed_text(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        return self.embedder.embed_texts(texts)

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self.embedder.dimension
