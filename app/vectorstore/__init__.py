"""
Vector Store Module
===================

Handles document embeddings and FAISS vector storage for semantic search.
"""

from app.vectorstore.embeddings import EmbeddingManager
from app.vectorstore.faiss_store import FAISSVectorStore

__all__ = ["EmbeddingManager", "FAISSVectorStore"]
