"""
Agents Module
=============

Multi-agent system with specialized agents for different tasks.

ARCHITECTURE:
                    ┌──────────────┐
                    │ User Query   │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ Router Agent │ ◄─── Classifies query intent
                    └──────┬───────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
    ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
    │  Retriever  │ │  Reasoning  │ │   Action    │
    │    Agent    │ │    Agent    │ │    Agent    │
    └─────────────┘ └─────────────┘ └─────────────┘
           │               ▲               │
           └───────────────┘               │
         (context flows to reasoning)      │
                    ┌──────────────────────┘
                    ▼
             ┌──────────────┐
             │   Response   │
             └──────────────┘
"""

from app.agents.base_agent import BaseAgent
from app.agents.router_agent import RouterAgent
from app.agents.retriever_agent import RetrieverAgent
from app.agents.reasoning_agent import ReasoningAgent
from app.agents.action_agent import ActionAgent

__all__ = [
    "BaseAgent",
    "RouterAgent",
    "RetrieverAgent",
    "ReasoningAgent",
    "ActionAgent",
]
