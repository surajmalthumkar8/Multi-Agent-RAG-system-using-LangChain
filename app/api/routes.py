"""
API Routes
==========

FastAPI endpoints for the Multi-Agent RAG system.

ENDPOINTS:
- POST /query: Submit a query to the RAG system
- POST /ingest: Ingest documents into the knowledge base
- GET /health: Health check endpoint
- DELETE /documents: Clear all documents

WHY FastAPI?
- Automatic OpenAPI documentation
- Type validation via Pydantic
- Async support for scalability
- Easy to test
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from app.schemas.models import (
    QueryRequest,
    QueryResponse,
    IngestionRequest,
    IngestionResponse,
    HealthResponse,
)
from app.services.orchestrator import MultiAgentOrchestrator
from app.services.document_service import DocumentService
from app import __version__

logger = logging.getLogger(__name__)

# Create router with prefix and tags for OpenAPI docs
router = APIRouter(prefix="/api/v1", tags=["rag"])

# Lazy initialization of services
# These are created on first request to avoid startup delays
_orchestrator: Optional[MultiAgentOrchestrator] = None
_document_service: Optional[DocumentService] = None


def get_orchestrator() -> MultiAgentOrchestrator:
    """Get or create the orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MultiAgentOrchestrator()
    return _orchestrator


def get_document_service() -> DocumentService:
    """Get or create the document service instance."""
    global _document_service
    if _document_service is None:
        _document_service = DocumentService()
    return _document_service


# =============================================================================
# Health Check
# =============================================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check if the API is running and the vector store is ready",
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns the API status and whether the knowledge base is ready.
    """
    orchestrator = get_orchestrator()

    return HealthResponse(
        status="healthy",
        version=__version__,
        vector_store_ready=orchestrator.is_ready,
        document_count=orchestrator.document_count,
    )


# =============================================================================
# Query Endpoint
# =============================================================================

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Submit Query",
    description="Submit a question to the Multi-Agent RAG system",
)
async def submit_query(request: QueryRequest) -> QueryResponse:
    """
    Process a user query through the multi-agent RAG pipeline.

    The query flows through:
    1. Router Agent: Classifies intent and routes
    2. Retriever Agent: Searches knowledge base
    3. Reasoning Agent: Generates grounded response
    4. Action Agent: Executes actions if needed

    Args:
        request: QueryRequest with the user's question

    Returns:
        QueryResponse with answer and sources

    Raises:
        HTTPException: If processing fails
    """
    try:
        orchestrator = get_orchestrator()

        # Check if we have documents
        if not orchestrator.is_ready:
            logger.warning("Query received but no documents in knowledge base")
            # We still process - reasoning agent will handle this gracefully

        response = await orchestrator.process_query(request)

        logger.info(
            f"Query processed: {request.query[:50]}... -> "
            f"{len(response.answer)} chars, {len(response.sources)} sources"
        )

        return response

    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )


# =============================================================================
# Document Ingestion
# =============================================================================

@router.post(
    "/ingest",
    response_model=IngestionResponse,
    summary="Ingest Documents",
    description="Ingest documents into the knowledge base",
)
async def ingest_documents(request: IngestionRequest) -> IngestionResponse:
    """
    Ingest documents into the vector store.

    If file_paths are specified, only those files are ingested.
    Otherwise, all documents in the configured directory are ingested.

    Args:
        request: IngestionRequest specifying what to ingest

    Returns:
        IngestionResponse with processing results
    """
    try:
        service = get_document_service()
        response = await service.ingest_documents(request)

        if response.errors:
            logger.warning(f"Ingestion completed with errors: {response.errors}")
        else:
            logger.info(
                f"Ingestion complete: {response.documents_processed} documents, "
                f"{response.chunks_created} chunks"
            )

        return response

    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest documents: {str(e)}"
        )


@router.post(
    "/ingest/text",
    summary="Ingest Text",
    description="Ingest raw text directly into the knowledge base",
)
async def ingest_text(
    text: str,
    source_name: str = "direct_input"
) -> dict:
    """
    Ingest raw text directly without creating a file.

    Args:
        text: The text content to ingest
        source_name: Name to use as the source

    Returns:
        Dictionary with chunks created count
    """
    try:
        service = get_document_service()
        chunks = await service.ingest_text(text, source_name)

        return {
            "success": True,
            "chunks_created": chunks,
            "source": source_name,
        }

    except Exception as e:
        logger.error(f"Text ingestion failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest text: {str(e)}"
        )


# =============================================================================
# Document Management
# =============================================================================

@router.delete(
    "/documents",
    summary="Clear Documents",
    description="Delete all documents from the knowledge base",
)
async def clear_documents() -> dict:
    """
    Clear all documents from the vector store.

    WARNING: This is destructive and cannot be undone.

    Returns:
        Confirmation message
    """
    try:
        service = get_document_service()
        service.clear_all_documents()

        return {
            "success": True,
            "message": "All documents have been cleared from the knowledge base",
        }

    except Exception as e:
        logger.error(f"Failed to clear documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear documents: {str(e)}"
        )


@router.get(
    "/documents/count",
    summary="Document Count",
    description="Get the number of document chunks in the knowledge base",
)
async def get_document_count() -> dict:
    """
    Get the current document chunk count.

    Returns:
        Dictionary with the count
    """
    service = get_document_service()

    return {
        "count": service.get_document_count(),
        "ready": service.is_ready(),
    }


# =============================================================================
# Conversation Management
# =============================================================================

@router.delete(
    "/conversations/{conversation_id}",
    summary="Clear Conversation",
    description="Clear memory for a specific conversation",
)
async def clear_conversation(conversation_id: str) -> dict:
    """
    Clear memory for a specific conversation.

    Args:
        conversation_id: The conversation to clear

    Returns:
        Confirmation message
    """
    orchestrator = get_orchestrator()
    orchestrator._memory_manager.clear_conversation(conversation_id)

    return {
        "success": True,
        "message": f"Cleared memory for conversation: {conversation_id}",
    }
