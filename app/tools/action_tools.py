"""
Action Tools
============

LangChain tools for executing actions like creating tickets,
escalating issues, and sending notifications.

These tools wrap the action logic to make it available to agents.
In production, they would integrate with real external systems.
"""

import logging
from datetime import datetime
from typing import Optional
import uuid

from langchain.tools import Tool
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Input Schemas
# =============================================================================

class TicketInput(BaseModel):
    """Input schema for ticket creation."""
    title: str = Field(description="Brief title for the support ticket")
    description: str = Field(description="Detailed description of the issue")
    priority: str = Field(
        default="medium",
        description="Priority level: low, medium, high, urgent"
    )
    customer_id: Optional[str] = Field(
        default=None,
        description="Customer ID if known"
    )


class EscalationInput(BaseModel):
    """Input schema for escalation."""
    reason: str = Field(description="Reason for escalation to human agent")
    priority: str = Field(
        default="medium",
        description="Priority level: low, medium, high, urgent"
    )
    context: Optional[str] = Field(
        default=None,
        description="Relevant context for the human agent"
    )


class EmailInput(BaseModel):
    """Input schema for email notifications."""
    subject: str = Field(description="Email subject line")
    body_summary: str = Field(description="Summary of email content")
    recipient_type: str = Field(
        default="customer",
        description="Recipient type: customer, support_team, manager"
    )


# =============================================================================
# Tool Functions
# =============================================================================

def create_support_ticket(
    title: str,
    description: str,
    priority: str = "medium",
    customer_id: Optional[str] = None,
) -> str:
    """
    Create a support ticket in the system.

    In production, this would call a ticketing API (Zendesk, Jira, etc.).
    For now, we simulate the creation.

    Args:
        title: Ticket title
        description: Detailed description
        priority: low, medium, high, urgent
        customer_id: Customer identifier

    Returns:
        Confirmation message with ticket ID
    """
    # Generate ticket ID
    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"

    # Validate priority
    valid_priorities = ["low", "medium", "high", "urgent"]
    if priority.lower() not in valid_priorities:
        priority = "medium"

    # In production: API call to ticketing system
    # ticketing_client.create(title=title, description=description, ...)

    logger.info(f"Created ticket {ticket_id}: {title} (Priority: {priority})")

    return (
        f"Support ticket created successfully.\n"
        f"Ticket ID: {ticket_id}\n"
        f"Title: {title}\n"
        f"Priority: {priority.capitalize()}\n"
        f"Status: Open\n"
        f"Our team will review and respond within the SLA for {priority} priority tickets."
    )


def escalate_to_human(
    reason: str,
    priority: str = "medium",
    context: Optional[str] = None,
) -> str:
    """
    Escalate the conversation to a human agent.

    In production, this would:
    - Add to a queue in the support platform
    - Notify available agents
    - Transfer the chat session

    Args:
        reason: Why escalation is needed
        priority: Urgency level
        context: Context to pass to human agent

    Returns:
        Confirmation with escalation details
    """
    escalation_id = f"ESC-{uuid.uuid4().hex[:8].upper()}"

    # Estimated wait times by priority (in production, query actual queue)
    wait_times = {
        "low": "15-30 minutes",
        "medium": "5-15 minutes",
        "high": "2-5 minutes",
        "urgent": "Under 2 minutes",
    }

    wait_time = wait_times.get(priority.lower(), "5-15 minutes")

    logger.info(f"Created escalation {escalation_id}: {reason}")

    return (
        f"Escalation initiated successfully.\n"
        f"Reference ID: {escalation_id}\n"
        f"Reason: {reason}\n"
        f"Priority: {priority.capitalize()}\n"
        f"Estimated wait time: {wait_time}\n\n"
        f"A human support agent will join this conversation shortly. "
        f"Please stay on this chat."
    )


def send_email_notification(
    subject: str,
    body_summary: str,
    recipient_type: str = "customer",
) -> str:
    """
    Send an email notification.

    In production, this would integrate with email services
    like SendGrid, AWS SES, or similar.

    Args:
        subject: Email subject
        body_summary: Summary of what the email contains
        recipient_type: Who receives the email

    Returns:
        Confirmation message
    """
    email_id = f"EMAIL-{uuid.uuid4().hex[:8].upper()}"

    # Map recipient types to descriptions
    recipient_descriptions = {
        "customer": "your registered email address",
        "support_team": "the support team",
        "manager": "the appropriate manager",
    }

    recipient_desc = recipient_descriptions.get(
        recipient_type.lower(),
        "the specified recipient"
    )

    logger.info(f"Sent email {email_id}: {subject} to {recipient_type}")

    return (
        f"Email notification sent.\n"
        f"Email ID: {email_id}\n"
        f"Subject: {subject}\n"
        f"Sent to: {recipient_desc}\n"
        f"Please check the inbox (and spam folder) within the next few minutes."
    )


# =============================================================================
# Tool Creation Functions
# =============================================================================

def create_ticket_tool() -> StructuredTool:
    """
    Create the ticket creation tool.

    Returns:
        StructuredTool for creating support tickets
    """
    return StructuredTool.from_function(
        func=create_support_ticket,
        name="create_support_ticket",
        description=(
            "Create a support ticket for issues that need follow-up. "
            "Use when the issue can't be resolved immediately, "
            "requires investigation, or needs to be tracked. "
            "Provide a clear title and detailed description."
        ),
        args_schema=TicketInput,
    )


def create_escalation_tool() -> StructuredTool:
    """
    Create the escalation tool.

    Returns:
        StructuredTool for escalating to human agents
    """
    return StructuredTool.from_function(
        func=escalate_to_human,
        name="escalate_to_human",
        description=(
            "Escalate the conversation to a human support agent. "
            "Use when: the customer explicitly requests a human, "
            "the issue is too complex for automated handling, "
            "the customer is frustrated or upset, "
            "or there's a sensitive matter requiring human judgment."
        ),
        args_schema=EscalationInput,
    )


def create_email_tool() -> StructuredTool:
    """
    Create the email notification tool.

    Returns:
        StructuredTool for sending emails
    """
    return StructuredTool.from_function(
        func=send_email_notification,
        name="send_email",
        description=(
            "Send an email notification about the support interaction. "
            "Use for: sending confirmation of actions taken, "
            "providing written documentation of solutions, "
            "or notifying relevant parties about issues."
        ),
        args_schema=EmailInput,
    )


def get_all_action_tools() -> list[StructuredTool]:
    """
    Get all action tools for agent use.

    Returns:
        List of all action tools
    """
    return [
        create_ticket_tool(),
        create_escalation_tool(),
        create_email_tool(),
    ]
