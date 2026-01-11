"""
FAISS Vector Store
==================

This module manages the FAISS vector database for semantic document search.

WHAT IS FAISS?
- Facebook AI Similarity Search
- Efficient library for similarity search in high-dimensional vectors
- Stores document embeddings and enables fast nearest-neighbor search

WHY FAISS?
- Fast: Optimized C++ with Python bindings
- Scalable: Handles millions of vectors
- Free: No external service needed (unlike Pinecone)
- Persistent: Can save/load index to disk

HOW IT WORKS:
1. Documents are split into chunks
2. Each chunk is embedded into a vector
3. Vectors are indexed in FAISS
4. Query is embedded and compared to all vectors
5. Most similar vectors (and their chunks) are returned
"""

import logging
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.vectorstore.embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    Manages FAISS vector store for document retrieval.

    This class provides:
    - Document indexing with automatic chunking
    - Semantic similarity search
    - Persistent storage (save/load from disk)
    - Singleton pattern for memory efficiency

    Usage:
        store = FAISSVectorStore()
        store.add_documents([Document(page_content="...", metadata={...})])
        results = store.similarity_search("query", k=5)
    """

    _instance: Optional["FAISSVectorStore"] = None
    _store: Optional[FAISS] = None
    _initialized: bool = False

    def __new__(cls) -> "FAISSVectorStore":
        """Singleton pattern - one vector store instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the vector store."""
        # Only initialize once
        if not self._initialized:
            self._settings = get_settings()
            self._embedding_manager = EmbeddingManager()
            self._text_splitter = self._create_text_splitter()
            self._try_load_existing_index()
            FAISSVectorStore._initialized = True

    def _create_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """
        Create text splitter for chunking documents.

        WHY RecursiveCharacterTextSplitter?
        - Tries to split on natural boundaries (paragraphs, sentences)
        - Falls back to characters if needed
        - Maintains context within chunks

        Chunk size of 1000 chars (~250 tokens) is a good balance:
        - Small enough to be specific
        - Large enough to maintain context
        """
        return RecursiveCharacterTextSplitter(
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
            length_function=len,
            # Split hierarchy: paragraphs -> sentences -> words -> chars
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _try_load_existing_index(self) -> None:
        """
        Try to load an existing FAISS index from disk.

        If no index exists, the store remains uninitialized
        until documents are added.
        """
        index_path = self._settings.faiss_index_path
        if (index_path / "index.faiss").exists():
            try:
                self._store = FAISS.load_local(
                    str(index_path),
                    self._embedding_manager.get_embeddings(),
                    allow_dangerous_deserialization=True,
                )
                logger.info(f"Loaded existing FAISS index from {index_path}")
            except Exception as e:
                logger.warning(f"Could not load existing index: {e}")
                self._store = None
        else:
            logger.info("No existing FAISS index found. Ready for indexing.")

    def add_documents(
        self,
        documents: list[Document],
        chunk: bool = True
    ) -> int:
        """
        Add documents to the vector store.

        Args:
            documents: List of LangChain Document objects
            chunk: Whether to split documents into chunks (default: True)

        Returns:
            Number of chunks created and indexed

        Example:
            docs = [Document(page_content="Long text...", metadata={"source": "file.pdf"})]
            chunks_created = store.add_documents(docs)
        """
        if not documents:
            logger.warning("No documents provided to index")
            return 0

        # Split documents into chunks if requested
        if chunk:
            chunks = self._text_splitter.split_documents(documents)
            logger.info(
                f"Split {len(documents)} documents into {len(chunks)} chunks"
            )
        else:
            chunks = documents

        # Create or update the FAISS index
        embeddings = self._embedding_manager.get_embeddings()

        if self._store is None:
            # Create new index
            self._store = FAISS.from_documents(chunks, embeddings)
            logger.info(f"Created new FAISS index with {len(chunks)} chunks")
        else:
            # Add to existing index
            self._store.add_documents(chunks)
            logger.info(f"Added {len(chunks)} chunks to existing index")

        # Persist to disk
        self._save_index()

        return len(chunks)

    def _save_index(self) -> None:
        """Save the FAISS index to disk for persistence."""
        if self._store is None:
            return

        index_path = self._settings.faiss_index_path
        index_path.mkdir(parents=True, exist_ok=True)

        self._store.save_local(str(index_path))
        logger.info(f"Saved FAISS index to {index_path}")

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> list[tuple[Document, float]]:
        """
        Search for documents similar to the query.

        This is the core retrieval function used by the Retriever Agent.

        Args:
            query: The search query text
            k: Number of results to return (default from settings)

        Returns:
            List of (Document, score) tuples, sorted by relevance
            Score is between 0 and 1, where 1 is most similar

        Raises:
            RuntimeError: If no documents have been indexed

        Example:
            results = store.similarity_search("password reset", k=3)
            for doc, score in results:
                print(f"Score: {score:.2f}, Content: {doc.page_content[:100]}")
        """
        if self._store is None:
            raise RuntimeError(
                "No documents indexed. Please add documents first."
            )

        k = k or self._settings.retrieval_top_k

        # FAISS returns (Document, score) tuples
        # Score is L2 distance; we convert to similarity (0-1 range)
        results = self._store.similarity_search_with_score(query, k=k)

        # Convert L2 distance to similarity score
        # L2 distance: 0 = identical, higher = less similar
        # We normalize using: similarity = 1 / (1 + distance)
        normalized_results = []
        for doc, distance in results:
            # Convert distance to similarity (0-1 range)
            similarity = 1 / (1 + distance)
            normalized_results.append((doc, similarity))

        return normalized_results

    def similarity_search_simple(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> list[Document]:
        """
        Simple search that returns just documents (no scores).

        Convenience method for when you just need the documents.

        Args:
            query: The search query
            k: Number of results

        Returns:
            List of Document objects
        """
        results = self.similarity_search(query, k)
        return [doc for doc, _ in results]

    def delete_all(self) -> None:
        """
        Delete all documents from the vector store.

        WARNING: This is destructive and cannot be undone.
        """
        self._store = None
        # Remove saved index files
        index_path = self._settings.faiss_index_path
        if index_path.exists():
            import shutil
            shutil.rmtree(index_path)
            index_path.mkdir(parents=True, exist_ok=True)
        logger.info("Deleted all documents from vector store")

    @property
    def is_ready(self) -> bool:
        """Check if the vector store has documents indexed."""
        return self._store is not None

    @property
    def document_count(self) -> int:
        """Get the number of document chunks in the store."""
        if self._store is None:
            return 0
        # FAISS doesn't expose this directly, so we use the index
        return self._store.index.ntotal

    def as_retriever(self, **kwargs):
        """
        Get a LangChain Retriever interface.

        This allows the vector store to be used directly
        in LangChain chains and agents.

        Args:
            **kwargs: Passed to FAISS.as_retriever()

        Returns:
            LangChain Retriever instance
        """
        if self._store is None:
            raise RuntimeError("No documents indexed")

        return self._store.as_retriever(
            search_kwargs={"k": self._settings.retrieval_top_k},
            **kwargs
        )
