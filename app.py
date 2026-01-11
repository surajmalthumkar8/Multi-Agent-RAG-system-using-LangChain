"""
Hugging Face Spaces Entry Point
================================

This file is the entry point for Hugging Face Spaces deployment.
It imports and runs the FastAPI application.
"""

import os

# Set environment variables for HuggingFace Spaces
# These can be overridden by Space secrets
os.environ.setdefault("LLM_PROVIDER", "huggingface")
os.environ.setdefault("EMBEDDING_PROVIDER", "huggingface")
os.environ.setdefault("HUGGINGFACE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
os.environ.setdefault("HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Import the FastAPI app
from app.main import app

# For Hugging Face Spaces, we need to expose the app
# Spaces will automatically run this with uvicorn
