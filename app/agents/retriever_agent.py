"""
Retriever Agent
===============

The Retriever Agent is responsible for finding relevant documents
from the vector store based on the user's query.

RESPONSIBILITIES:
1. Take the user query
2. Search the FAISS vector store
3. Return relevant document chunks with scores
4. Optionally re-rank results for better accuracy

WHY A SEPARATE RETRIEVER AGENT?
- Single responsibility: Only handles retrieval
- Can be enhanced independently (re-ranking, hybrid search)
- Makes testing easier
- Allows for retrieval-specific optimizations

RETRIEVAL STRATEGY:
1. Embed the query using same model as documents
2. Find top-k nearest neighbors in vector space
3. Return documents with relevance scores
4. (Optional) Re-rank using cross-encoder for better accuracy
"""

import logging
from typing import Any

from app.agents.base_agent import BaseAgent
from app.schemas.models import AgentResponse, AgentType, RetrievedDocument
from app.vectorstore.faiss_store import FAISSVectorStore
from app.config import get_settings

logger = logging.getLogger(__name__)


class RetrieverAgent(BaseAgent):
    """
    Retrieves relevant documents from the vector store.

    This agent encapsulates the retrieval logic, making it easy to:
    - Swap vector stores (FAISS -> Pinecone)
    - Add re-ranking
    - Implement hybrid search (semantic + keyword)
    """

    def __init__(self, vector_store: FAISSVectorStore = None, **kwargs):
        """
        Initialize the Retriever Agent.

        Args:
            vector_store: FAISS store instance (uses singleton if not provided)
            **kwargs: Passed to BaseAgent
        """
        super().__init__(**kwargs)
        self._vector_store = vector_store or FAISSVectorStore()
        self._settings = get_settings()

    @property
    def agent_type(self) -> AgentType:
        """Return the agent type."""
        return AgentType.RETRIEVER

    async def execute(
        self,
        input_data: dict[str, Any],
        **kwargs
    ) -> AgentResponse:
        """
        Retrieve relevant documents for the query.

        Args:
            input_data: Must contain 'query' key
            **kwargs: Optional 'top_k' to override default

        Returns:
            AgentResponse with retrieved documents in metadata
        """
        query = input_data.get("query", "")
        top_k = kwargs.get("top_k", self._settings.retrieval_top_k)

        if not query:
            return AgentResponse(
                agent_type=self.agent_type,
                output="No query provided for retrieval",
                confidence=0.0,
                metadata={"documents": [], "error": "empty_query"}
            )

        # Check if vector store is ready
        if not self._vector_store.is_ready:
            return AgentResponse(
                agent_type=self.agent_type,
                output="No documents in knowledge base. Please ingest documents first.",
                confidence=0.0,
                metadata={"documents": [], "error": "no_documents"}
            )

        # Perform similarity search
        try:
            results = self._vector_store.similarity_search(query, k=top_k)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return AgentResponse(
                agent_type=self.agent_type,
                output=f"Retrieval error: {str(e)}",
                confidence=0.0,
                metadata={"documents": [], "error": str(e)}
            )

        # Convert results to RetrievedDocument objects
        documents = []
        for doc, score in results:
            retrieved_doc = RetrievedDocument(
                content=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                relevance_score=score,
                chunk_index=doc.metadata.get("chunk_index")
            )
            documents.append(retrieved_doc)

        # Calculate overall confidence based on best match
        best_score = documents[0].relevance_score if documents else 0.0

        # Format documents for context
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(
                f"[Document {i}] (Source: {doc.source}, Relevance: {doc.relevance_score:.2f})\n"
                f"{doc.content}"
            )

        context_string = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant documents found."

        logger.info(
            f"Retrieved {len(documents)} documents, best score: {best_score:.2f}"
        )

        return AgentResponse(
            agent_type=self.agent_type,
            output=context_string,
            confidence=best_score,
            metadata={
                "documents": [doc.model_dump() for doc in documents],
                "document_count": len(documents),
                "best_relevance_score": best_score,
            }
        )

    def get_retriever(self):
        """
        Get LangChain-compatible retriever interface.

        This allows the agent to be used directly in LangChain chains.

        Returns:
            LangChain Retriever instance
        """
        return self._vector_store.as_retriever()
