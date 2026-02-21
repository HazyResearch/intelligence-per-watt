"""Agent harnesses for benchmarking LLM-powered agents.

Provides BaseAgent ABC and agent implementations (ReAct, OpenHands, Terminus)
with integrated MCP tool support and energy telemetry.
"""

from ipw.agents.base import BaseAgent
from ipw.core.registry import AgentRegistry

__all__ = [
    "AgentRegistry",
    "BaseAgent",
]
