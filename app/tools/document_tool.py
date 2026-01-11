"""
Document Loader Tool
====================

Tool for loading and processing documents into the vector store.

This tool handles:
- Loading documents from files
- Splitting into chunks
- Adding to the vector store

SUPPORTED FORMATS:
- .txt: Plain text files
- .pdf: PDF documents
- .md: Markdown files
- .docx: Word documents
"""

import logging
from pathlib import Path
from typing import Optional

from langchain.tools import Tool
from langchain_core.tools import StructuredTool
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
)
from pydantic import BaseModel, Field

from app.vectorstore.faiss_store import FAISSVectorStore
from app.config import get_settings

logger = logging.getLogger(__name__)


class DocumentLoadInput(BaseModel):
    """Input for document loading tool."""
    file_path: str = Field(description="Path to the document file to load")


def load_document(file_path: str) -> list[Document]:
    """
    Load a document from file path.

    Automatically selects the appropriate loader based on file extension.

    Args:
        file_path: Path to the document

    Returns:
        List of Document objects (may be multiple for PDFs)

    Raises:
        ValueError: If file type not supported
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = path.suffix.lower()

    # Select loader based on extension
    loaders = {
        ".txt": TextLoader,
        ".md": TextLoader,
        ".pdf": PyPDFLoader,
        ".docx": UnstructuredWordDocumentLoader,
    }

    loader_class = loaders.get(extension)
    if loader_class is None:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            f"Supported: {list(loaders.keys())}"
        )

    # Load the document
    loader = loader_class(str(path))
    documents = loader.load()

    # Add source metadata
    for doc in documents:
        doc.metadata["source"] = str(path)
        doc.metadata["file_type"] = extension

    logger.info(f"Loaded {len(documents)} documents from {file_path}")

    return documents


def create_document_loader_tool(
    vector_store: FAISSVectorStore = None
) -> StructuredTool:
    """
    Create a tool for loading documents into the vector store.

    This tool is useful for agents that need to ingest new documents
    into the knowledge base.

    Args:
        vector_store: FAISS store instance

    Returns:
        StructuredTool for document loading
    """
    store = vector_store or FAISSVectorStore()

    def load_and_index(file_path: str) -> str:
        """Load a document and add it to the vector store."""
        try:
            documents = load_document(file_path)
            chunks_created = store.add_documents(documents)
            return (
                f"Successfully loaded and indexed document: {file_path}\n"
                f"Created {chunks_created} searchable chunks."
            )
        except FileNotFoundError as e:
            return f"Error: {str(e)}"
        except ValueError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Failed to load document: {e}")
            return f"Failed to load document: {str(e)}"

    return StructuredTool.from_function(
        func=load_and_index,
        name="load_document",
        description=(
            "Load a document file and add it to the knowledge base. "
            "Supports .txt, .md, .pdf, and .docx files. "
            "The document will be automatically chunked and indexed for search."
        ),
        args_schema=DocumentLoadInput,
    )


def load_directory(
    directory_path: str,
    vector_store: FAISSVectorStore = None,
    extensions: list[str] = None
) -> dict:
    """
    Load all documents from a directory.

    Args:
        directory_path: Path to directory containing documents
        vector_store: FAISS store instance
        extensions: List of extensions to include (default: all supported)

    Returns:
        Dictionary with loading results
    """
    store = vector_store or FAISSVectorStore()
    path = Path(directory_path)

    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    if not path.is_dir():
        raise ValueError(f"Not a directory: {directory_path}")

    # Default extensions
    supported = extensions or [".txt", ".md", ".pdf", ".docx"]

    results = {
        "files_processed": 0,
        "chunks_created": 0,
        "errors": [],
        "files": [],
    }

    # Find all matching files
    for ext in supported:
        for file_path in path.glob(f"**/*{ext}"):
            try:
                documents = load_document(str(file_path))
                chunks = store.add_documents(documents)
                results["files_processed"] += 1
                results["chunks_created"] += chunks
                results["files"].append({
                    "path": str(file_path),
                    "chunks": chunks,
                })
            except Exception as e:
                results["errors"].append({
                    "path": str(file_path),
                    "error": str(e),
                })
                logger.error(f"Failed to load {file_path}: {e}")

    logger.info(
        f"Loaded {results['files_processed']} files, "
        f"created {results['chunks_created']} chunks"
    )

    return results
