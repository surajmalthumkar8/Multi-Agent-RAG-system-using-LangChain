"""
Data Models
===========

This module defines all Pydantic models used throughout the application.
Models provide:
- Request/response validation for API endpoints
- Type safety for internal data flow
- Automatic API documentation via OpenAPI

WHY Pydantic?
- Runtime type validation (catches errors early)
- Automatic JSON serialization
- Integration with FastAPI for auto-docs
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================

class AgentType(str, Enum):
    """
    Types of agents in the multi-agent system.

    Each agent has a single responsibility:
    - ROUTER: Classifies query intent and routes to appropriate agent
    - RETRIEVER: Searches vector store for relevant documents
    - REASONING: Generates grounded responses from context
    - ACTION: Executes specific actions (e.g., create ticket, send email)
    """
    ROUTER = "router"
    RETRIEVER = "retriever"
    REASONING = "reasoning"
    ACTION = "action"


class ActionType(str, Enum):
    """
    Supported actions the Action Agent can execute.

    These represent real business operations in a support system.
    """
    CREATE_TICKET = "create_ticket"
    ESCALATE = "escalate"
    SEND_EMAIL = "send_email"
    SEARCH_KB = "search_knowledge_base"
    NONE = "none"  # No action needed, just informational response


# =============================================================================
# Document Models
# =============================================================================

class DocumentInfo(BaseModel):
    """Metadata about a source document."""
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File extension (pdf, txt, etc.)")
    chunk_count: int = Field(..., description="Number of chunks created")
    ingested_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When document was processed"
    )


class RetrievedDocument(BaseModel):
    """
    A document chunk retrieved from the vector store.

    Contains both the content and metadata for traceability.
    This allows users to verify the source of information.
    """
    content: str = Field(..., description="The text content of the chunk")
    source: str = Field(..., description="Source file path")
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Similarity score (1.0 = perfect match)"
    )
    chunk_index: Optional[int] = Field(
        default=None,
        description="Position of chunk in original document"
    )


# =============================================================================
# Conversation Models
# =============================================================================

class ConversationMessage(BaseModel):
    """
    A single message in the conversation history.

    Used by memory module to maintain context across turns.
    """
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When message was sent"
    )


# =============================================================================
# Agent Response Models
# =============================================================================

class AgentResponse(BaseModel):
    """
    Response from an individual agent.

    Each agent returns this structure for consistent handling.
    """
    agent_type: AgentType = Field(..., description="Which agent responded")
    output: str = Field(..., description="Agent's output text")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Agent's confidence in response"
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Additional agent-specific data"
    )


# =============================================================================
# API Request/Response Models
# =============================================================================

class QueryRequest(BaseModel):
    """
    User query request to the RAG system.

    Example:
        {
            "query": "How do I reset my password?",
            "conversation_id": "abc123",
            "include_sources": true
        }
    """
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's question or request"
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="ID for conversation continuity"
    )
    include_sources: bool = Field(
        default=True,
        description="Whether to return source documents"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "How do I reset my password?",
                "conversation_id": "user-session-123",
                "include_sources": True
            }
        }


class QueryResponse(BaseModel):
    """
    Response from the RAG system.

    Contains the answer and optionally the source documents
    that were used to generate it.
    """
    answer: str = Field(..., description="The generated answer")
    sources: list[RetrievedDocument] = Field(
        default_factory=list,
        description="Documents used to generate answer"
    )
    conversation_id: str = Field(..., description="Conversation identifier")
    agent_trace: list[str] = Field(
        default_factory=list,
        description="Sequence of agents that processed the query"
    )
    action_taken: Optional[ActionType] = Field(
        default=None,
        description="Action executed, if any"
    )
    processing_time_ms: float = Field(
        ...,
        description="Total processing time in milliseconds"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "To reset your password, go to Settings > Security > Reset Password...",
                "sources": [
                    {
                        "content": "Password reset instructions...",
                        "source": "docs/security.pdf",
                        "relevance_score": 0.92
                    }
                ],
                "conversation_id": "user-session-123",
                "agent_trace": ["router", "retriever", "reasoning"],
                "action_taken": None,
                "processing_time_ms": 1250.5
            }
        }


# =============================================================================
# Document Ingestion Models
# =============================================================================

class IngestionRequest(BaseModel):
    """Request to ingest documents into the vector store."""
    file_paths: list[str] = Field(
        default_factory=list,
        description="Specific files to ingest (empty = all in documents_path)"
    )
    force_reindex: bool = Field(
        default=False,
        description="Re-index even if already processed"
    )


class IngestionResponse(BaseModel):
    """Response after document ingestion."""
    documents_processed: int = Field(..., description="Number of files processed")
    chunks_created: int = Field(..., description="Total chunks created")
    documents: list[DocumentInfo] = Field(
        default_factory=list,
        description="Details of each processed document"
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Any errors encountered"
    )


# =============================================================================
# Health Check Model
# =============================================================================

class HealthResponse(BaseModel):
    """API health check response."""
    status: str = Field(..., description="'healthy' or 'unhealthy'")
    version: str = Field(..., description="API version")
    vector_store_ready: bool = Field(
        ...,
        description="Whether vector store is initialized"
    )
    document_count: int = Field(
        ...,
        description="Number of documents in vector store"
    )
