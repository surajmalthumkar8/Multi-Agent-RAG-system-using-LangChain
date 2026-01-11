"""
Action Agent
============

The Action Agent executes specific actions based on user requests.

RESPONSIBILITIES:
1. Parse action requests from the query
2. Execute the appropriate action
3. Return confirmation and next steps
4. Handle action failures gracefully

SUPPORTED ACTIONS:
- create_ticket: Create a support ticket
- escalate: Escalate to human agent
- send_email: Send email notification
- search_kb: Deep search in knowledge base

WHY AN ACTION AGENT?
- Separates "thinking" from "doing"
- Actions can be audited and logged
- Easy to add new actions
- Can integrate with external systems (ticketing, email, etc.)

ARCHITECTURE:
    ┌─────────────────────────────────────────┐
    │           Action Agent                   │
    │  ┌──────────────────────────────────┐   │
    │  │    Action Router (LLM-based)     │   │
    │  └─────────────┬────────────────────┘   │
    │                │                         │
    │   ┌────────────┼────────────┐           │
    │   ▼            ▼            ▼           │
    │ ┌─────┐   ┌─────────┐  ┌─────────┐     │
    │ │Ticket│   │Escalate│  │ Email   │     │
    │ │Tool  │   │ Tool   │  │  Tool   │     │
    │ └─────┘   └─────────┘  └─────────┘     │
    └─────────────────────────────────────────┘
"""

import logging
from typing import Any, Optional
from datetime import datetime
import uuid

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.agents.base_agent import BaseAgent
from app.schemas.models import AgentResponse, AgentType, ActionType

logger = logging.getLogger(__name__)


# In a production system, these would integrate with real services
# For now, we simulate the actions and return structured results

class TicketData(BaseModel):
    """Data for a created support ticket."""
    ticket_id: str
    title: str
    description: str
    priority: str
    created_at: str
    status: str = "open"


class EscalationData(BaseModel):
    """Data for an escalation."""
    escalation_id: str
    reason: str
    priority: str
    queue: str
    estimated_wait: str


class ActionResult(BaseModel):
    """Result of an action execution."""
    action_type: ActionType
    success: bool
    message: str
    data: Optional[dict] = None


ACTION_PROMPT = """You are an action executor for a customer support system.

Based on the conversation, you need to execute the requested action.

AVAILABLE ACTIONS:
1. create_ticket - Create a support ticket for the issue
2. escalate - Escalate to a human agent
3. send_email - Send an email notification
4. search_knowledge_base - Perform a deeper search
5. none - No action needed

USER QUERY: {query}

CONTEXT: {context}

REQUESTED ACTION: {action_type}

Generate appropriate details for this action:
- For tickets: Generate a clear title and description
- For escalation: Determine priority and reason
- For email: Determine recipient type and content summary

Respond in this JSON format:
{{
    "title": "Brief title of the issue",
    "description": "Detailed description",
    "priority": "low|medium|high|urgent",
    "reason": "Why this action is being taken"
}}"""


class ActionAgent(BaseAgent):
    """
    Executes actions based on user requests and reasoning output.

    This agent handles the "doing" part of the system - creating tickets,
    escalating issues, and other concrete actions.

    In production, this would integrate with:
    - Ticketing systems (Zendesk, Jira, ServiceNow)
    - Email services (SendGrid, SES)
    - Communication platforms (Slack, Teams)
    """

    def __init__(self, **kwargs):
        """Initialize the Action Agent."""
        super().__init__(**kwargs)
        self._prompt = ChatPromptTemplate.from_template(ACTION_PROMPT)

    @property
    def agent_type(self) -> AgentType:
        """Return the agent type."""
        return AgentType.ACTION

    async def execute(
        self,
        input_data: dict[str, Any],
        **kwargs
    ) -> AgentResponse:
        """
        Execute the requested action.

        Args:
            input_data:
                - query: User's original query
                - context: Retrieved context
                - action_type: Which action to execute
            **kwargs: Additional options

        Returns:
            AgentResponse with action result
        """
        query = input_data.get("query", "")
        context = input_data.get("context", "")
        action_type_str = input_data.get("action_type", "none")

        # Parse action type
        try:
            action_type = ActionType(action_type_str)
        except ValueError:
            action_type = ActionType.NONE

        # If no action needed, return early
        if action_type == ActionType.NONE:
            return AgentResponse(
                agent_type=self.agent_type,
                output="No action required.",
                confidence=1.0,
                metadata={"action_type": "none", "action_taken": False}
            )

        # Execute the appropriate action
        result = await self._execute_action(
            action_type=action_type,
            query=query,
            context=context
        )

        return AgentResponse(
            agent_type=self.agent_type,
            output=result.message,
            confidence=1.0 if result.success else 0.5,
            metadata={
                "action_type": result.action_type.value,
                "action_taken": result.success,
                "action_data": result.data,
            }
        )

    async def _execute_action(
        self,
        action_type: ActionType,
        query: str,
        context: str
    ) -> ActionResult:
        """
        Execute a specific action.

        This method routes to the appropriate action handler.

        Args:
            action_type: Which action to execute
            query: User's query
            context: Retrieved context

        Returns:
            ActionResult with success/failure and details
        """
        # Map action types to handlers
        action_handlers = {
            ActionType.CREATE_TICKET: self._create_ticket,
            ActionType.ESCALATE: self._escalate,
            ActionType.SEND_EMAIL: self._send_email,
            ActionType.SEARCH_KB: self._search_kb,
        }

        handler = action_handlers.get(action_type)
        if handler is None:
            return ActionResult(
                action_type=action_type,
                success=False,
                message=f"Unknown action type: {action_type.value}",
            )

        try:
            return await handler(query, context)
        except Exception as e:
            logger.error(f"Action {action_type.value} failed: {e}")
            return ActionResult(
                action_type=action_type,
                success=False,
                message=f"Action failed: {str(e)}",
            )

    async def _create_ticket(self, query: str, context: str) -> ActionResult:
        """
        Create a support ticket.

        In production, this would call a ticketing API.
        Here we simulate the ticket creation.
        """
        # Generate ticket details using LLM
        details = await self._generate_action_details(
            ActionType.CREATE_TICKET, query, context
        )

        # Simulate ticket creation
        ticket = TicketData(
            ticket_id=f"TKT-{uuid.uuid4().hex[:8].upper()}",
            title=details.get("title", "Support Request"),
            description=details.get("description", query),
            priority=details.get("priority", "medium"),
            created_at=datetime.utcnow().isoformat(),
        )

        logger.info(f"Created ticket: {ticket.ticket_id}")

        return ActionResult(
            action_type=ActionType.CREATE_TICKET,
            success=True,
            message=f"I've created support ticket {ticket.ticket_id} for your issue. "
                    f"Our team will review it shortly. Priority: {ticket.priority}.",
            data=ticket.model_dump(),
        )

    async def _escalate(self, query: str, context: str) -> ActionResult:
        """
        Escalate to a human agent.

        In production, this would add to a queue or notify agents.
        """
        details = await self._generate_action_details(
            ActionType.ESCALATE, query, context
        )

        escalation = EscalationData(
            escalation_id=f"ESC-{uuid.uuid4().hex[:8].upper()}",
            reason=details.get("reason", "Customer requested human assistance"),
            priority=details.get("priority", "medium"),
            queue="general_support",
            estimated_wait="5-10 minutes",
        )

        logger.info(f"Created escalation: {escalation.escalation_id}")

        return ActionResult(
            action_type=ActionType.ESCALATE,
            success=True,
            message=f"I've escalated your request to a human agent. "
                    f"Reference: {escalation.escalation_id}. "
                    f"Estimated wait time: {escalation.estimated_wait}. "
                    f"A support representative will assist you shortly.",
            data=escalation.model_dump(),
        )

    async def _send_email(self, query: str, context: str) -> ActionResult:
        """
        Send an email notification.

        In production, this would call an email service.
        """
        details = await self._generate_action_details(
            ActionType.SEND_EMAIL, query, context
        )

        # Simulate email sending
        email_id = f"EMAIL-{uuid.uuid4().hex[:8].upper()}"

        logger.info(f"Sent email: {email_id}")

        return ActionResult(
            action_type=ActionType.SEND_EMAIL,
            success=True,
            message="I've sent a confirmation email to your registered email address. "
                    "Please check your inbox (and spam folder) shortly.",
            data={
                "email_id": email_id,
                "subject": details.get("title", "Support Update"),
            },
        )

    async def _search_kb(self, query: str, context: str) -> ActionResult:
        """
        Perform a deeper knowledge base search.

        This could trigger a more thorough search with different parameters.
        """
        # In production, this might search with different strategies
        return ActionResult(
            action_type=ActionType.SEARCH_KB,
            success=True,
            message="I've initiated a deeper search of our knowledge base. "
                    "This may take a moment for complex queries.",
            data={"search_query": query},
        )

    async def _generate_action_details(
        self,
        action_type: ActionType,
        query: str,
        context: str
    ) -> dict:
        """
        Use LLM to generate appropriate details for an action.

        Args:
            action_type: Type of action
            query: User's query
            context: Retrieved context

        Returns:
            Dictionary with action-specific details
        """
        try:
            formatted = self._prompt.format_messages(
                query=query,
                context=context[:2000],  # Limit context length
                action_type=action_type.value,
            )
            response = await self._llm.ainvoke(formatted)

            # Parse JSON from response
            import json
            content = response.content

            # Try to extract JSON from the response
            if "{" in content:
                json_start = content.index("{")
                json_end = content.rindex("}") + 1
                json_str = content[json_start:json_end]
                return json.loads(json_str)

        except Exception as e:
            logger.warning(f"Failed to generate action details: {e}")

        # Return defaults
        return {
            "title": "Support Request",
            "description": query,
            "priority": "medium",
            "reason": "User request",
        }
