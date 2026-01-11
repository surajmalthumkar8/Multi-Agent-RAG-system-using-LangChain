"""
Conversation Memory Manager
===========================

This module manages conversation history for multi-turn interactions.

WHY MEMORY?
- Users expect context: "What about the second option?" requires memory
- Follow-up questions need previous context
- Maintains coherent, contextual conversations

We implement a simple window-based memory that keeps the last N messages.
This avoids deprecated LangChain memory modules and gives us full control.
"""

import logging
from typing import Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from app.schemas.models import ConversationMessage

logger = logging.getLogger(__name__)

# Maximum messages to keep in memory (per conversation)
# 10 exchanges = 20 messages (user + assistant)
DEFAULT_WINDOW_SIZE = 10


@dataclass
class ConversationHistory:
    """Holds the message history for a single conversation."""
    messages: deque = field(default_factory=lambda: deque(maxlen=DEFAULT_WINDOW_SIZE * 2))
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)


class ConversationMemoryManager:
    """
    Manages conversation memory across multiple user sessions.

    Each conversation_id gets its own memory instance.
    This allows multiple concurrent users without mixing context.

    Usage:
        manager = ConversationMemoryManager()

        # Add messages
        manager.add_user_message("session-123", "Hello")
        manager.add_ai_message("session-123", "Hi there!")

        # Get context for prompts
        context = manager.get_context_string("session-123")
    """

    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE):
        """
        Initialize the memory manager.

        Args:
            window_size: Number of conversation turns to remember
        """
        self._window_size = window_size
        # Dictionary mapping conversation_id -> ConversationHistory
        self._conversations: dict[str, ConversationHistory] = {}

    def _get_or_create_history(
        self,
        conversation_id: str
    ) -> ConversationHistory:
        """
        Get existing history or create new one for conversation.

        Args:
            conversation_id: Unique identifier for the conversation

        Returns:
            ConversationHistory for this conversation
        """
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = ConversationHistory(
                messages=deque(maxlen=self._window_size * 2)
            )
            logger.debug(f"Created new memory for conversation: {conversation_id}")

        return self._conversations[conversation_id]

    def add_user_message(self, conversation_id: str, content: str) -> None:
        """
        Add a user message to conversation history.

        Args:
            conversation_id: Conversation identifier
            content: The user's message text
        """
        history = self._get_or_create_history(conversation_id)

        message = ConversationMessage(role="user", content=content)
        history.messages.append(message)
        history.last_updated = datetime.utcnow()

        logger.debug(f"Added user message to {conversation_id}: {content[:50]}...")

    def add_ai_message(self, conversation_id: str, content: str) -> None:
        """
        Add an AI response to conversation history.

        Args:
            conversation_id: Conversation identifier
            content: The AI's response text
        """
        history = self._get_or_create_history(conversation_id)

        message = ConversationMessage(role="assistant", content=content)
        history.messages.append(message)
        history.last_updated = datetime.utcnow()

        logger.debug(f"Added AI message to {conversation_id}: {content[:50]}...")

    def get_messages(self, conversation_id: str) -> list[ConversationMessage]:
        """
        Get all messages in a conversation.

        Useful for API responses that want to show conversation history.

        Args:
            conversation_id: Conversation identifier

        Returns:
            List of messages in chronological order
        """
        if conversation_id not in self._conversations:
            return []
        return list(self._conversations[conversation_id].messages)

    def get_context_string(self, conversation_id: str) -> str:
        """
        Get conversation history as a formatted string.

        Useful for including in prompts.

        Args:
            conversation_id: Conversation identifier

        Returns:
            Formatted string of conversation history
        """
        messages = self.get_messages(conversation_id)

        if not messages:
            return "No previous conversation."

        lines = []
        for msg in messages:
            role = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{role}: {msg.content}")

        return "\n".join(lines)

    def get_messages_for_llm(self, conversation_id: str) -> list[dict]:
        """
        Get messages formatted for LLM consumption.

        Returns messages in the format expected by chat models:
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

        Args:
            conversation_id: Conversation identifier

        Returns:
            List of message dictionaries
        """
        messages = self.get_messages(conversation_id)
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    def clear_conversation(self, conversation_id: str) -> None:
        """
        Clear all memory for a conversation.

        Args:
            conversation_id: Conversation to clear
        """
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            logger.info(f"Cleared memory for conversation: {conversation_id}")

    def generate_conversation_id(self) -> str:
        """
        Generate a new unique conversation ID.

        Returns:
            UUID string for new conversation
        """
        return str(uuid.uuid4())

    @property
    def active_conversations(self) -> int:
        """Get count of active conversations in memory."""
        return len(self._conversations)
