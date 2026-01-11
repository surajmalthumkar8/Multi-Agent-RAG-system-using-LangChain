"""
Document Service
================

Service for ingesting and managing documents in the knowledge base.

This service handles:
- Loading documents from various file formats
- Splitting documents into chunks
- Indexing in the vector store
- Tracking document metadata

WHY A SEPARATE SERVICE?
- Separates ingestion from query processing
- Can be run as a batch job
- Easy to add new document types
- Provides clear API for document management
"""

import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    DirectoryLoader,
)

from app.vectorstore.faiss_store import FAISSVectorStore
from app.schemas.models import (
    IngestionRequest,
    IngestionResponse,
    DocumentInfo,
)
from app.config import get_settings

logger = logging.getLogger(__name__)


class DocumentService:
    """
    Service for document ingestion and management.

    Handles loading documents from files, chunking them,
    and storing in the vector database.

    Usage:
        service = DocumentService()
        result = await service.ingest_directory("/path/to/docs")
    """

    # Supported file extensions and their loaders
    SUPPORTED_EXTENSIONS = {
        ".txt": TextLoader,
        ".md": TextLoader,
        ".pdf": PyPDFLoader,
        ".docx": UnstructuredWordDocumentLoader,
    }

    def __init__(self, vector_store: Optional[FAISSVectorStore] = None):
        """
        Initialize the document service.

        Args:
            vector_store: FAISS store instance (uses singleton if not provided)
        """
        self._vector_store = vector_store or FAISSVectorStore()
        self._settings = get_settings()

    async def ingest_documents(
        self,
        request: IngestionRequest
    ) -> IngestionResponse:
        """
        Ingest documents based on request.

        Args:
            request: IngestionRequest specifying what to ingest

        Returns:
            IngestionResponse with results
        """
        if request.file_paths:
            # Ingest specific files
            return await self._ingest_files(
                request.file_paths,
                request.force_reindex
            )
        else:
            # Ingest all documents in the configured directory
            return await self.ingest_directory(
                str(self._settings.documents_path),
                request.force_reindex
            )

    async def ingest_directory(
        self,
        directory_path: str,
        force_reindex: bool = False
    ) -> IngestionResponse:
        """
        Ingest all supported documents from a directory.

        Recursively finds and indexes all supported file types.

        Args:
            directory_path: Path to directory
            force_reindex: If True, clear existing index first

        Returns:
            IngestionResponse with details
        """
        path = Path(directory_path)

        if not path.exists():
            return IngestionResponse(
                documents_processed=0,
                chunks_created=0,
                errors=[f"Directory not found: {directory_path}"],
            )

        if not path.is_dir():
            return IngestionResponse(
                documents_processed=0,
                chunks_created=0,
                errors=[f"Not a directory: {directory_path}"],
            )

        # Clear existing index if requested
        if force_reindex:
            logger.info("Force reindex requested - clearing existing index")
            self._vector_store.delete_all()

        # Find all supported files
        all_files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            all_files.extend(path.glob(f"**/*{ext}"))

        if not all_files:
            return IngestionResponse(
                documents_processed=0,
                chunks_created=0,
                errors=[f"No supported files found in {directory_path}"],
            )

        # Ingest each file
        return await self._ingest_files(
            [str(f) for f in all_files],
            force_reindex=False  # Already handled above
        )

    async def _ingest_files(
        self,
        file_paths: list[str],
        force_reindex: bool = False
    ) -> IngestionResponse:
        """
        Ingest a list of specific files.

        Args:
            file_paths: List of file paths to ingest
            force_reindex: Clear existing index first

        Returns:
            IngestionResponse with details
        """
        if force_reindex:
            self._vector_store.delete_all()

        documents_info = []
        errors = []
        total_chunks = 0

        for file_path in file_paths:
            try:
                result = await self._ingest_single_file(file_path)
                documents_info.append(result)
                total_chunks += result.chunk_count
            except Exception as e:
                error_msg = f"Failed to process {file_path}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

        return IngestionResponse(
            documents_processed=len(documents_info),
            chunks_created=total_chunks,
            documents=documents_info,
            errors=errors,
        )

    async def _ingest_single_file(self, file_path: str) -> DocumentInfo:
        """
        Ingest a single file into the vector store.

        Args:
            file_path: Path to the file

        Returns:
            DocumentInfo about the processed file

        Raises:
            ValueError: If file type not supported
            FileNotFoundError: If file doesn't exist
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = path.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported: {list(self.SUPPORTED_EXTENSIONS.keys())}"
            )

        # Load the document
        loader_class = self.SUPPORTED_EXTENSIONS[extension]
        loader = loader_class(str(path))
        documents = loader.load()

        # Add metadata
        for i, doc in enumerate(documents):
            doc.metadata.update({
                "source": str(path),
                "file_name": path.name,
                "file_type": extension,
                "chunk_index": i,
                "ingested_at": datetime.utcnow().isoformat(),
            })

        # Add to vector store (handles chunking)
        chunks_created = self._vector_store.add_documents(documents)

        logger.info(f"Ingested {path.name}: {chunks_created} chunks")

        return DocumentInfo(
            filename=path.name,
            file_type=extension,
            chunk_count=chunks_created,
        )

    async def ingest_text(
        self,
        text: str,
        source_name: str = "direct_input",
        metadata: Optional[dict] = None
    ) -> int:
        """
        Ingest raw text directly.

        Useful for adding content without creating files.

        Args:
            text: Text content to ingest
            source_name: Name to use as source
            metadata: Additional metadata

        Returns:
            Number of chunks created
        """
        doc = Document(
            page_content=text,
            metadata={
                "source": source_name,
                "file_type": "text",
                "ingested_at": datetime.utcnow().isoformat(),
                **(metadata or {}),
            }
        )

        return self._vector_store.add_documents([doc])

    def get_document_count(self) -> int:
        """Get the total number of document chunks in the store."""
        return self._vector_store.document_count

    def is_ready(self) -> bool:
        """Check if the document store has any documents."""
        return self._vector_store.is_ready

    def clear_all_documents(self) -> None:
        """
        Clear all documents from the vector store.

        WARNING: This is destructive and cannot be undone.
        """
        self._vector_store.delete_all()
        logger.info("All documents cleared from vector store")
