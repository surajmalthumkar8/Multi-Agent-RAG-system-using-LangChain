"""
Multi-Agent Orchestrator
========================

The orchestrator is the "conductor" of the multi-agent system.
It coordinates the flow between agents and manages the overall pipeline.

PIPELINE FLOW:

    ┌─────────┐
    │  Query  │
    └────┬────┘
         │
    ┌────▼────┐
    │ Router  │ ──► Classifies query intent
    │  Agent  │
    └────┬────┘
         │
    ┌────▼──────────────┐
    │  Needs Retrieval? │
    └────┬──────────────┘
         │Yes
    ┌────▼────┐
    │Retriever│ ──► Searches vector store
    │  Agent  │
    └────┬────┘
         │
    ┌────▼────┐
    │Reasoning│ ──► Generates grounded response
    │  Agent  │
    └────┬────┘
         │
    ┌────▼──────────┐
    │ Needs Action? │
    └────┬──────────┘
         │Yes
    ┌────▼────┐
    │ Action  │ ──► Executes action (ticket, escalate)
    │  Agent  │
    └────┬────┘
         │
    ┌────▼────┐
    │Response │
    └─────────┘

WHY AN ORCHESTRATOR?
- Decouples agents from each other
- Easy to modify pipeline without changing agents
- Provides central logging and monitoring
- Handles error recovery at pipeline level
"""

import logging
import time
from typing import Optional
import uuid

from app.agents.router_agent import RouterAgent
from app.agents.retriever_agent import RetrieverAgent
from app.agents.reasoning_agent import ReasoningAgent
from app.agents.action_agent import ActionAgent
from app.schemas.models import (
    QueryRequest,
    QueryResponse,
    RetrievedDocument,
    ActionType,
)
from app.memory.conversation_memory import ConversationMemoryManager
from app.vectorstore.faiss_store import FAISSVectorStore

logger = logging.getLogger(__name__)


class MultiAgentOrchestrator:
    """
    Orchestrates the multi-agent RAG pipeline.

    This class:
    1. Receives user queries
    2. Routes them through appropriate agents
    3. Manages state between agents
    4. Returns consolidated responses

    Usage:
        orchestrator = MultiAgentOrchestrator()
        response = await orchestrator.process_query(
            QueryRequest(query="How do I reset my password?")
        )
    """

    def __init__(
        self,
        vector_store: Optional[FAISSVectorStore] = None,
        memory_manager: Optional[ConversationMemoryManager] = None,
    ):
        """
        Initialize the orchestrator with all agents.

        Args:
            vector_store: FAISS store (uses singleton if not provided)
            memory_manager: Conversation memory (creates new if not provided)
        """
        # Shared dependencies
        self._vector_store = vector_store or FAISSVectorStore()
        self._memory_manager = memory_manager or ConversationMemoryManager()

        # Initialize all agents
        self._router = RouterAgent()
        self._retriever = RetrieverAgent(vector_store=self._vector_store)
        self._reasoning = ReasoningAgent(memory_manager=self._memory_manager)
        self._action = ActionAgent()

        logger.info("Multi-agent orchestrator initialized")

    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """
        Process a user query through the multi-agent pipeline.

        This is the main entry point for the RAG system.

        Args:
            request: QueryRequest with user's question

        Returns:
            QueryResponse with answer, sources, and metadata
        """
        start_time = time.time()

        # Generate or use existing conversation ID
        conversation_id = request.conversation_id or self._memory_manager.generate_conversation_id()

        # Track which agents process this query
        agent_trace = []

        logger.info(f"Processing query: {request.query[:100]}...")

        try:
            # Step 1: Route the query
            routing_response = await self._router.safe_execute({
                "query": request.query,
                "context": self._memory_manager.get_context_string(conversation_id),
            })
            agent_trace.append("router")

            routing_meta = routing_response.metadata
            needs_retrieval = routing_meta.get("needs_retrieval", True)
            needs_action = routing_meta.get("needs_action", False)
            suggested_action = routing_meta.get("suggested_action", "none")

            logger.info(
                f"Routing: retrieval={needs_retrieval}, action={needs_action}"
            )

            # Step 2: Retrieve documents if needed
            context = ""
            retrieved_docs = []

            if needs_retrieval:
                retrieval_response = await self._retriever.safe_execute({
                    "query": request.query,
                })
                agent_trace.append("retriever")

                context = retrieval_response.output
                retrieved_docs = retrieval_response.metadata.get("documents", [])

            # Step 3: Generate reasoning response
            reasoning_response = await self._reasoning.safe_execute({
                "query": request.query,
                "context": context or "No context available.",
                "conversation_id": conversation_id,
            })
            agent_trace.append("reasoning")

            answer = reasoning_response.output

            # Step 4: Execute action if needed
            action_taken = None

            if needs_action:
                action_response = await self._action.safe_execute({
                    "query": request.query,
                    "context": context,
                    "action_type": suggested_action,
                })
                agent_trace.append("action")

                # Append action result to answer
                if action_response.metadata.get("action_taken"):
                    answer += f"\n\n---\n**Action Taken:**\n{action_response.output}"
                    action_taken = ActionType(
                        action_response.metadata.get("action_type", "none")
                    )

            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000

            # Build response
            sources = []
            if request.include_sources and retrieved_docs:
                for doc_dict in retrieved_docs:
                    sources.append(RetrievedDocument(**doc_dict))

            logger.info(
                f"Query processed in {processing_time_ms:.2f}ms, "
                f"agents: {' -> '.join(agent_trace)}"
            )

            return QueryResponse(
                answer=answer,
                sources=sources,
                conversation_id=conversation_id,
                agent_trace=agent_trace,
                action_taken=action_taken,
                processing_time_ms=processing_time_ms,
            )

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)

            processing_time_ms = (time.time() - start_time) * 1000

            return QueryResponse(
                answer=(
                    "I apologize, but I encountered an error processing your request. "
                    "Please try again or contact support if the issue persists."
                ),
                sources=[],
                conversation_id=conversation_id,
                agent_trace=agent_trace,
                action_taken=None,
                processing_time_ms=processing_time_ms,
            )

    async def process_simple_query(self, query: str) -> str:
        """
        Simple interface for quick queries.

        Skips the full pipeline and just does retrieval + reasoning.

        Args:
            query: User's question

        Returns:
            Answer string
        """
        # Retrieve context
        if self._vector_store.is_ready:
            retrieval_response = await self._retriever.safe_execute({
                "query": query,
            })
            context = retrieval_response.output
        else:
            context = "No documents in knowledge base."

        # Generate response
        reasoning_response = await self._reasoning.safe_execute({
            "query": query,
            "context": context,
        })

        return reasoning_response.output

    @property
    def is_ready(self) -> bool:
        """Check if the orchestrator is ready to process queries."""
        return self._vector_store.is_ready

    @property
    def document_count(self) -> int:
        """Get number of documents in the knowledge base."""
        return self._vector_store.document_count
