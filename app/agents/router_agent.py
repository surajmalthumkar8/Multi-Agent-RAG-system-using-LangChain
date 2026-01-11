"""
Router Agent
============

The Router Agent is the "traffic controller" of the multi-agent system.
It analyzes incoming queries and decides which agents should handle them.

RESPONSIBILITIES:
1. Classify query intent (question, action request, or both)
2. Determine if retrieval is needed
3. Decide if actions should be executed
4. Return routing instructions for the orchestrator

WHY A ROUTER?
- Not all queries need all agents
- Saves compute by skipping unnecessary agents
- Enables conditional logic in the pipeline
- Makes the system more efficient

ROUTING LOGIC:
┌─────────────────────────────────────────────────────────────────┐
│ Query Type          │ Retrieval │ Reasoning │ Action           │
├─────────────────────┼───────────┼───────────┼──────────────────┤
│ Factual question    │ YES       │ YES       │ NO               │
│ How-to question     │ YES       │ YES       │ NO               │
│ Action request      │ MAYBE     │ YES       │ YES              │
│ Small talk          │ NO        │ YES       │ NO               │
└─────────────────────────────────────────────────────────────────┘
"""

import json
import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from app.agents.base_agent import BaseAgent
from app.schemas.models import AgentResponse, AgentType, ActionType

logger = logging.getLogger(__name__)


class RoutingDecision(BaseModel):
    """
    Structured output from the Router Agent.

    This Pydantic model ensures the LLM returns valid routing decisions.
    """
    needs_retrieval: bool = Field(
        description="Whether to search the knowledge base for context"
    )
    needs_reasoning: bool = Field(
        default=True,
        description="Whether to generate a reasoned response"
    )
    needs_action: bool = Field(
        description="Whether to execute an action (create ticket, etc.)"
    )
    suggested_action: str = Field(
        default="none",
        description="Which action to execute if needs_action is True"
    )
    query_type: str = Field(
        description="Classification: 'factual', 'how_to', 'action', 'general'"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the routing decision"
    )
    reasoning: str = Field(
        description="Brief explanation of the routing decision"
    )


# The router prompt is carefully designed to:
# 1. Give clear context about the system
# 2. Provide examples of different query types
# 3. Request structured JSON output
# 4. Ask for reasoning (helps catch errors)
ROUTER_PROMPT = """You are a query router for a customer support system.
Your job is to analyze the user's query and decide how to handle it.

AVAILABLE AGENTS:
1. Retriever: Searches the knowledge base for relevant information
2. Reasoning: Generates responses based on context
3. Action: Executes actions like creating tickets or escalating

ROUTING RULES:
- Factual questions (what, who, when, where) -> Retriever + Reasoning
- How-to questions (how do I, steps to) -> Retriever + Reasoning
- Action requests (create ticket, escalate, reset password) -> Retriever + Reasoning + Action
- General conversation (hello, thanks, goodbye) -> Reasoning only
- Complaints or urgent issues -> Retriever + Reasoning + Action (escalate)

AVAILABLE ACTIONS:
- create_ticket: Create a support ticket
- escalate: Escalate to human agent
- send_email: Send email notification
- search_knowledge_base: Deep search in KB
- none: No action needed

USER QUERY: {query}

CONVERSATION CONTEXT: {context}

Analyze the query and provide your routing decision.

{format_instructions}"""


class RouterAgent(BaseAgent):
    """
    Routes queries to appropriate agents based on intent classification.

    The Router Agent uses an LLM to understand the query intent and
    determine which agents should process it. This enables:

    1. Efficiency: Skip unnecessary agents
    2. Accuracy: Match queries to appropriate handlers
    3. Flexibility: Easy to add new routing logic
    """

    def __init__(self, **kwargs):
        """Initialize the Router Agent."""
        super().__init__(**kwargs)
        # Parser ensures LLM output matches RoutingDecision schema
        self._parser = PydanticOutputParser(pydantic_object=RoutingDecision)
        self._prompt = ChatPromptTemplate.from_template(ROUTER_PROMPT)

    @property
    def agent_type(self) -> AgentType:
        """Return the agent type."""
        return AgentType.ROUTER

    async def execute(
        self,
        input_data: dict[str, Any],
        **kwargs
    ) -> AgentResponse:
        """
        Analyze query and determine routing.

        Args:
            input_data: Must contain 'query' key, optionally 'context'

        Returns:
            AgentResponse with routing decision in metadata
        """
        query = input_data.get("query", "")
        context = input_data.get("context", "No previous context")

        if not query:
            return AgentResponse(
                agent_type=self.agent_type,
                output="No query provided",
                confidence=0.0,
                metadata={"error": "empty_query"}
            )

        # Format the prompt with query and context
        formatted_prompt = self._prompt.format_messages(
            query=query,
            context=context,
            format_instructions=self._parser.get_format_instructions()
        )

        # Get routing decision from LLM
        response = await self._llm.ainvoke(formatted_prompt)

        # Parse the structured output
        try:
            decision = self._parser.parse(response.content)
        except Exception as e:
            logger.warning(f"Failed to parse routing decision: {e}")
            # Default to full pipeline on parse error
            decision = RoutingDecision(
                needs_retrieval=True,
                needs_reasoning=True,
                needs_action=False,
                suggested_action="none",
                query_type="unknown",
                confidence=0.5,
                reasoning="Failed to parse, using default routing"
            )

        # Convert suggested_action string to ActionType enum
        try:
            action_type = ActionType(decision.suggested_action)
        except ValueError:
            action_type = ActionType.NONE

        logger.info(
            f"Routed query: retrieval={decision.needs_retrieval}, "
            f"action={decision.needs_action} ({action_type.value})"
        )

        return AgentResponse(
            agent_type=self.agent_type,
            output=decision.reasoning,
            confidence=decision.confidence,
            metadata={
                "needs_retrieval": decision.needs_retrieval,
                "needs_reasoning": decision.needs_reasoning,
                "needs_action": decision.needs_action,
                "suggested_action": action_type.value,
                "query_type": decision.query_type,
            }
        )
