"""
Embedding Manager
=================

This module handles the conversion of text to vector embeddings.
Supports both FREE (HuggingFace) and PAID (OpenAI) embeddings.

FREE OPTION: HuggingFace sentence-transformers
- Runs locally, no API costs
- Good quality embeddings
- Model: all-MiniLM-L6-v2 (384 dimensions, fast)

PAID OPTION: OpenAI
- Cloud-based
- Higher quality for some tasks
- Requires API key and costs money
"""

import logging
from typing import Optional

from langchain_core.embeddings import Embeddings

from app.config import get_settings

logger = logging.getLogger(__name__)


def create_embeddings(provider: Optional[str] = None) -> Embeddings:
    """
    Create embeddings instance based on provider.

    Args:
        provider: Override provider from settings ("huggingface" or "openai")

    Returns:
        LangChain Embeddings instance
    """
    settings = get_settings()
    provider = provider or settings.embedding_provider

    logger.info(f"Creating embeddings with provider: {provider}")

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=settings.huggingface_embedding_model,
            model_kwargs={"device": "cpu"},  # Use "cuda" if GPU available
            encode_kwargs={"normalize_embeddings": True},
        )

    elif provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key,
        )

    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")


class EmbeddingManager:
    """
    Manages text embedding generation.

    By default uses FREE HuggingFace embeddings that run locally.
    Can be configured to use OpenAI for higher quality (paid).

    Usage:
        manager = EmbeddingManager()
        embeddings = manager.get_embeddings()
        vector = embeddings.embed_query("Hello world")
    """

    _instance: Optional["EmbeddingManager"] = None
    _embeddings: Optional[Embeddings] = None

    def __new__(cls) -> "EmbeddingManager":
        """Singleton pattern ensures we only create one embedding client."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the embedding manager with settings."""
        if self._embeddings is None:
            self._initialize_embeddings()

    def _initialize_embeddings(self) -> None:
        """Create the embeddings client."""
        settings = get_settings()

        try:
            self._embeddings = create_embeddings(settings.embedding_provider)
            logger.info(
                f"Initialized embeddings with provider: {settings.embedding_provider}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise

    def get_embeddings(self) -> Embeddings:
        """
        Get the embeddings instance for use with vector stores.

        Returns:
            Embeddings instance

        Raises:
            RuntimeError: If embeddings not initialized
        """
        if self._embeddings is None:
            raise RuntimeError("Embeddings not initialized")
        return self._embeddings

    def embed_text(self, text: str) -> list[float]:
        """
        Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        embeddings = self.get_embeddings()
        return embeddings.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple documents in batch.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings = self.get_embeddings()
        return embeddings.embed_documents(texts)

    @property
    def dimension(self) -> int:
        """
        Get the embedding dimension for the current model.

        Returns:
            int: Number of dimensions in embedding vector
        """
        settings = get_settings()

        # Dimensions for common models
        dimensions = {
            # HuggingFace models
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "sentence-transformers/all-mpnet-base-v2": 768,
            "BAAI/bge-small-en-v1.5": 384,
            "BAAI/bge-base-en-v1.5": 768,
            # OpenAI models
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

        model = (
            settings.huggingface_embedding_model
            if settings.embedding_provider == "huggingface"
            else settings.openai_embedding_model
        )

        return dimensions.get(model, 384)
