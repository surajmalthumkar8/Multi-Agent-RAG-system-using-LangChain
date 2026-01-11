"""
Base Agent
==========

Abstract base class for all agents in the multi-agent system.
Supports multiple LLM providers including free options.

SUPPORTED PROVIDERS:
1. ollama     - Local LLMs (FREE, requires Ollama installed)
2. huggingface - HuggingFace API (FREE tier available)
3. groq       - Groq Cloud (FREE tier, very fast)
4. google     - Google Gemini (FREE tier)
5. openai     - OpenAI (PAID)
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import get_settings
from app.schemas.models import AgentResponse, AgentType

logger = logging.getLogger(__name__)


def create_llm(
    provider: Optional[str] = None,
    temperature: Optional[float] = None,
) -> BaseChatModel:
    """
    Create an LLM instance based on the configured provider.

    Args:
        provider: Override the default provider from settings
        temperature: Override the default temperature

    Returns:
        A LangChain chat model instance

    Raises:
        ValueError: If provider is not supported
    """
    settings = get_settings()
    provider = provider or settings.llm_provider
    temp = temperature if temperature is not None else settings.llm_temperature

    logger.info(f"Creating LLM with provider: {provider}")

    if provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=temp,
        )

    elif provider == "huggingface":
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
        # Use HuggingFace Inference API
        llm = HuggingFaceEndpoint(
            repo_id=settings.huggingface_model,
            huggingfacehub_api_token=settings.huggingface_api_key,
            temperature=temp,
            max_new_tokens=1024,
        )
        return ChatHuggingFace(llm=llm)

    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=temp,
        )

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=settings.google_model,
            google_api_key=settings.google_api_key,
            temperature=temp,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.openai_model,
            openai_api_key=settings.openai_api_key,
            temperature=temp,
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    Each agent must implement:
    - agent_type: What kind of agent this is
    - execute(): Main logic for the agent

    Provides:
    - Multi-provider LLM initialization
    - Consistent error handling
    - Logging infrastructure
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        provider: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        """
        Initialize the base agent.

        Args:
            llm: Pre-configured LLM (optional, for testing/customization)
            provider: Override LLM provider from settings
            temperature: Override default temperature
        """
        self._settings = get_settings()

        # Use provided LLM or create based on provider
        if llm is not None:
            self._llm = llm
        else:
            self._llm = create_llm(provider, temperature)

        logger.info(f"Initialized {self.agent_type.value} agent")

    @property
    @abstractmethod
    def agent_type(self) -> AgentType:
        """Return the type of this agent."""
        pass

    @abstractmethod
    async def execute(
        self,
        input_data: dict[str, Any],
        **kwargs
    ) -> AgentResponse:
        """Execute the agent's main logic."""
        pass

    async def safe_execute(
        self,
        input_data: dict[str, Any],
        **kwargs
    ) -> AgentResponse:
        """
        Execute with error handling wrapper.

        This ensures agents always return a valid response,
        even if errors occur.
        """
        try:
            logger.debug(f"{self.agent_type.value} starting execution")
            response = await self.execute(input_data, **kwargs)
            logger.debug(f"{self.agent_type.value} completed successfully")
            return response

        except Exception as e:
            logger.error(
                f"{self.agent_type.value} failed: {type(e).__name__}: {e}",
                exc_info=True
            )
            return AgentResponse(
                agent_type=self.agent_type,
                output=f"Agent error: {str(e)}",
                confidence=0.0,
                metadata={"error": str(e), "error_type": type(e).__name__}
            )

    def _format_prompt(self, template: str, **kwargs) -> str:
        """Format a prompt template with variables."""
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Missing prompt variable: {e}")
            return template

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"{self.__class__.__name__}(type={self.agent_type.value})"
