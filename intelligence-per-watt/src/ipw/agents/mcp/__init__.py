"""MCP (Model Context Protocol) server implementations.

This module provides unified interfaces for:
- Local models (via Ollama, vLLM)
- Cloud APIs (OpenAI, Anthropic, Gemini, OpenRouter)
- Tools (calculator, web search, code interpreter)
- Retrieval (BM25, dense, grep, hybrid)

All servers automatically capture telemetry (energy, power, cost, latency).
"""

# Retrieval module - import lazily to avoid mandatory dependencies
from ipw.agents.mcp import retrieval
from ipw.agents.mcp.anthropic_server import AnthropicMCPServer
from ipw.agents.mcp.base import BaseMCPServer, MCPToolResult
from ipw.agents.mcp.openai_server import OpenAIMCPServer
from ipw.agents.mcp.openrouter_server import OpenRouterMCPServer
from ipw.agents.mcp.tool_registry import ADPDomainServer, ToolCategory, ToolRegistry, ToolSpec, get_registry
from ipw.agents.mcp.tool_server import (
    CalculatorServer,
    CodeInterpreterServer,
    FileReadServer,
    FileWriteServer,
    ThinkServer,
    WebSearchServer,
)
from ipw.agents.mcp.vllm_server import VLLMMCPServer


def __getattr__(name):
    if name == "OllamaMCPServer":
        from ipw.agents.mcp.ollama_server import OllamaMCPServer
        return OllamaMCPServer
    if name == "GeminiMCPServer":
        from ipw.agents.mcp.gemini_server import GeminiMCPServer
        return GeminiMCPServer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Base classes
    "BaseMCPServer",
    "MCPToolResult",
    # Model servers
    "OllamaMCPServer",
    "OpenAIMCPServer",
    "AnthropicMCPServer",
    "GeminiMCPServer",
    "OpenRouterMCPServer",
    "VLLMMCPServer",
    # Tool servers
    "CalculatorServer",
    "WebSearchServer",
    "CodeInterpreterServer",
    "ThinkServer",
    "FileReadServer",
    "FileWriteServer",
    # Tool registry
    "ToolRegistry",
    "ToolSpec",
    "ToolCategory",
    "ADPDomainServer",
    "get_registry",
    # Retrieval module
    "retrieval",
]
