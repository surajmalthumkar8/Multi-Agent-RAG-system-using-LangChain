"""
Reasoning Agent
===============

The Reasoning Agent generates grounded responses based on retrieved context.

RESPONSIBILITIES:
1. Take the user query and retrieved context
2. Generate an accurate, helpful response
3. Ground all claims in the provided context
4. Admit when information is not available

KEY PRINCIPLE: GROUNDING
- Every statement must be traceable to source documents
- Never hallucinate or make up information
- If context doesn't contain the answer, say so
- This builds user trust and prevents misinformation

WHY SEPARATE FROM RETRIEVER?
- Clear separation of concerns
- Retrieval can be optimized independently
- Reasoning prompts can be tuned without affecting retrieval
- Makes the system more testable
"""

import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.base_agent import BaseAgent
from app.schemas.models import AgentResponse, AgentType
from app.memory.conversation_memory import ConversationMemoryManager

logger = logging.getLogger(__name__)


# The reasoning prompt is critical for grounding
# Key elements:
# 1. Clear role definition
# 2. Explicit grounding instructions
# 3. How to handle missing information
# 4. Format guidelines
REASONING_SYSTEM_PROMPT = """You are a helpful customer support assistant.

YOUR ROLE:
- Answer user questions accurately based on the provided context
- Be helpful, professional, and concise
- Guide users through solutions step-by-step when appropriate

CRITICAL GROUNDING RULES:
1. ONLY use information from the provided context documents
2. If the context doesn't contain the answer, say: "I don't have specific information about that in my knowledge base. Let me connect you with a human agent who can help."
3. Do NOT make up information, policies, or procedures
4. When referencing information, you may cite the source document
5. If context is partially relevant, use what's applicable and note limitations

RESPONSE FORMAT:
- Be concise but complete
- Use bullet points for lists or steps
- If providing steps, number them
- End with a helpful follow-up question if appropriate

CONVERSATION HISTORY:
{chat_history}

RETRIEVED CONTEXT:
{context}

Remember: It's better to admit you don't know than to provide incorrect information."""

USER_PROMPT = """User Question: {query}

Please provide a helpful, accurate response based on the context above."""


class ReasoningAgent(BaseAgent):
    """
    Generates grounded responses from retrieved context.

    This agent is the "brain" that synthesizes information
    from the retriever into coherent, accurate responses.
    """

    def __init__(
        self,
        memory_manager: ConversationMemoryManager = None,
        **kwargs
    ):
        """
        Initialize the Reasoning Agent.

        Args:
            memory_manager: For conversation context (optional)
            **kwargs: Passed to BaseAgent
        """
        super().__init__(**kwargs)
        self._memory_manager = memory_manager or ConversationMemoryManager()

    @property
    def agent_type(self) -> AgentType:
        """Return the agent type."""
        return AgentType.REASONING

    async def execute(
        self,
        input_data: dict[str, Any],
        **kwargs
    ) -> AgentResponse:
        """
        Generate a grounded response.

        Args:
            input_data:
                - query: User's question
                - context: Retrieved documents (from Retriever Agent)
                - conversation_id: For memory (optional)
            **kwargs: Additional options

        Returns:
            AgentResponse with the generated answer
        """
        query = input_data.get("query", "")
        context = input_data.get("context", "No context provided.")
        conversation_id = input_data.get("conversation_id", "")

        if not query:
            return AgentResponse(
                agent_type=self.agent_type,
                output="No query provided",
                confidence=0.0,
                metadata={"error": "empty_query"}
            )

        # Get conversation history if available
        chat_history = ""
        if conversation_id:
            chat_history = self._memory_manager.get_context_string(conversation_id)

        # Format the prompts
        system_prompt = REASONING_SYSTEM_PROMPT.format(
            chat_history=chat_history or "No previous conversation.",
            context=context
        )

        user_prompt = USER_PROMPT.format(query=query)

        # Create messages for the LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        # Generate response
        try:
            response = await self._llm.ainvoke(messages)
            answer = response.content
        except Exception as e:
            logger.error(f"Reasoning failed: {e}")
            return AgentResponse(
                agent_type=self.agent_type,
                output=f"I encountered an error generating a response: {str(e)}",
                confidence=0.0,
                metadata={"error": str(e)}
            )

        # Estimate confidence based on context relevance
        # This is a simple heuristic - could be improved with more sophisticated methods
        confidence = self._estimate_confidence(context, answer)

        # Update conversation memory
        if conversation_id:
            self._memory_manager.add_user_message(conversation_id, query)
            self._memory_manager.add_ai_message(conversation_id, answer)

        logger.info(f"Generated response with confidence: {confidence:.2f}")

        return AgentResponse(
            agent_type=self.agent_type,
            output=answer,
            confidence=confidence,
            metadata={
                "has_context": bool(context and context != "No context provided."),
                "conversation_id": conversation_id,
            }
        )

    def _estimate_confidence(self, context: str, answer: str) -> float:
        """
        Estimate confidence in the generated response.

        This is a simple heuristic based on:
        1. Whether context was provided
        2. Whether the answer admits uncertainty

        A production system might use:
        - NLI (Natural Language Inference) to check grounding
        - Semantic similarity between answer and context
        - LLM self-evaluation

        Args:
            context: The retrieved context
            answer: The generated answer

        Returns:
            Confidence score between 0 and 1
        """
        # Start with base confidence
        confidence = 0.7

        # No context = lower confidence
        if not context or context == "No context provided.":
            confidence = 0.3

        # Phrases indicating uncertainty
        uncertainty_phrases = [
            "i don't have",
            "not sure",
            "can't find",
            "no information",
            "don't know",
            "couldn't find",
            "not in my knowledge",
        ]

        answer_lower = answer.lower()
        for phrase in uncertainty_phrases:
            if phrase in answer_lower:
                confidence = min(confidence, 0.4)
                break

        # Very short answers might indicate issues
        if len(answer) < 50:
            confidence = min(confidence, 0.5)

        return confidence
