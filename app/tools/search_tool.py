"""
Search Tool
===========

LangChain tool for semantic search in the vector store.

This tool wraps the FAISS vector store to provide a clean interface
for agents to search documents.

WHY A TOOL?
- LangChain agents work with tools as their action primitives
- Tools have clear input/output schemas
- Makes the search capability composable
"""

from langchain.tools import Tool
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.vectorstore.faiss_store import FAISSVectorStore


class SearchInput(BaseModel):
    """Input schema for the search tool."""
    query: str = Field(description="The search query to find relevant documents")
    num_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to return"
    )


def create_search_tool(vector_store: FAISSVectorStore = None) -> StructuredTool:
    """
    Create a search tool for semantic document search.

    The tool searches the FAISS vector store and returns
    relevant document chunks with their sources.

    Args:
        vector_store: FAISS store instance (uses singleton if not provided)

    Returns:
        StructuredTool that can be used by LangChain agents

    Example:
        tool = create_search_tool()
        result = tool.invoke({"query": "password reset", "num_results": 3})
    """
    store = vector_store or FAISSVectorStore()

    def search_documents(query: str, num_results: int = 5) -> str:
        """
        Search for documents matching the query.

        Returns formatted string of results for agent consumption.
        """
        if not store.is_ready:
            return "Error: No documents in knowledge base. Please ingest documents first."

        try:
            results = store.similarity_search(query, k=num_results)
        except Exception as e:
            return f"Search error: {str(e)}"

        if not results:
            return "No relevant documents found for the query."

        # Format results for agent consumption
        output_parts = [f"Found {len(results)} relevant documents:\n"]

        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("source", "unknown")
            content = doc.page_content[:500]  # Limit length
            if len(doc.page_content) > 500:
                content += "..."

            output_parts.append(
                f"\n[Result {i}] (Relevance: {score:.2f}, Source: {source})\n"
                f"{content}"
            )

        return "\n".join(output_parts)

    return StructuredTool.from_function(
        func=search_documents,
        name="search_knowledge_base",
        description=(
            "Search the knowledge base for information relevant to the query. "
            "Use this to find documentation, policies, procedures, and FAQs. "
            "Returns relevant document excerpts with their sources."
        ),
        args_schema=SearchInput,
    )


def create_simple_search_tool(vector_store: FAISSVectorStore = None) -> Tool:
    """
    Create a simple search tool with just a query string.

    This is an alternative for agents that work better with
    simple string inputs rather than structured inputs.

    Args:
        vector_store: FAISS store instance

    Returns:
        Simple Tool with string input
    """
    store = vector_store or FAISSVectorStore()

    def search(query: str) -> str:
        """Search documents with a query string."""
        if not store.is_ready:
            return "No documents in knowledge base."

        try:
            results = store.similarity_search(query, k=5)
            if not results:
                return "No relevant documents found."

            output = []
            for doc, score in results:
                output.append(f"[{score:.2f}] {doc.page_content[:300]}...")

            return "\n\n".join(output)
        except Exception as e:
            return f"Search error: {str(e)}"

    return Tool(
        name="search",
        func=search,
        description="Search the knowledge base for relevant information",
    )
