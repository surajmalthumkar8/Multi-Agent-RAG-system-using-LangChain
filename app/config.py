"""
Configuration Management
========================

This module centralizes all configuration using pydantic-settings.
It loads environment variables and provides type-safe access to config values.

SUPPORTED LLM PROVIDERS (all have free tiers):
1. ollama     - Local LLMs (Llama, Mistral) - completely free
2. huggingface - HuggingFace Inference API - free tier available
3. groq       - Groq Cloud - free tier with fast inference
4. google     - Google Gemini - free tier available
5. openai     - OpenAI - paid (for reference)
"""

import os
from pathlib import Path
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    All settings have sensible defaults for development.
    In production, override via environment variables or .env file.
    """

    # =========================================================================
    # LLM Provider Selection
    # =========================================================================
    llm_provider: Literal["ollama", "huggingface", "groq", "google", "openai"] = Field(
        default="ollama",
        description="Which LLM provider to use (ollama is free and local)"
    )

    # =========================================================================
    # API Keys (only needed for cloud providers)
    # =========================================================================
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key (only if using openai provider)"
    )

    huggingface_api_key: str = Field(
        default="",
        description="HuggingFace API key (free at huggingface.co)"
    )

    groq_api_key: str = Field(
        default="",
        description="Groq API key (free at console.groq.com)"
    )

    google_api_key: str = Field(
        default="",
        description="Google API key (free at makersuite.google.com)"
    )

    # =========================================================================
    # Model Configuration
    # =========================================================================
    # Models for each provider
    ollama_model: str = Field(
        default="llama3.2",
        description="Ollama model (llama3.2, mistral, phi3, etc.)"
    )

    huggingface_model: str = Field(
        default="mistralai/Mistral-7B-Instruct-v0.2",
        description="HuggingFace model ID"
    )

    groq_model: str = Field(
        default="llama-3.1-8b-instant",
        description="Groq model (llama-3.1-8b-instant, mixtral-8x7b-32768)"
    )

    google_model: str = Field(
        default="gemini-1.5-flash",
        description="Google Gemini model"
    )

    openai_model: str = Field(
        default="gpt-3.5-turbo",
        description="OpenAI model"
    )

    # Temperature controls randomness: 0 = deterministic, 1 = creative
    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="LLM temperature (lower = more focused)"
    )

    # =========================================================================
    # Embedding Configuration
    # =========================================================================
    embedding_provider: Literal["huggingface", "openai"] = Field(
        default="huggingface",
        description="Embedding provider (huggingface is free and local)"
    )

    huggingface_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Free local embedding model"
    )

    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model (paid)"
    )

    # =========================================================================
    # Ollama Configuration
    # =========================================================================
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL"
    )

    # =========================================================================
    # Vector Store Configuration
    # =========================================================================
    faiss_index_path: Path = Field(
        default=Path("./data/faiss_index"),
        description="Directory to store FAISS index"
    )

    documents_path: Path = Field(
        default=Path("./data/documents"),
        description="Directory containing source documents"
    )

    chunk_size: int = Field(
        default=1000,
        description="Size of document chunks for embedding"
    )

    chunk_overlap: int = Field(
        default=200,
        description="Overlap between consecutive chunks"
    )

    retrieval_top_k: int = Field(
        default=5,
        description="Number of documents to retrieve"
    )

    # =========================================================================
    # API Configuration
    # =========================================================================
    api_host: str = Field(
        default="0.0.0.0",
        description="Host to bind the API server"
    )

    api_port: int = Field(
        default=8000,
        description="Port for the API server"
    )

    debug_mode: bool = Field(
        default=True,
        description="Enable debug mode for development"
    )

    # =========================================================================
    # Logging Configuration
    # =========================================================================
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level"
    )

    class Config:
        """Pydantic configuration for settings."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.faiss_index_path.mkdir(parents=True, exist_ok=True)
        self.documents_path.mkdir(parents=True, exist_ok=True)

    def get_model_name(self) -> str:
        """Get the model name for the selected provider."""
        model_map = {
            "ollama": self.ollama_model,
            "huggingface": self.huggingface_model,
            "groq": self.groq_model,
            "google": self.google_model,
            "openai": self.openai_model,
        }
        return model_map.get(self.llm_provider, self.ollama_model)


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings: Application configuration instance
    """
    return Settings()
