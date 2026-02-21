# Agents Overview

IPW supports profiling multi-turn agent workloads through pluggable agent harnesses. Each agent wraps an existing framework (Agno, OpenHands SDK, or terminal-bench) and adds energy telemetry instrumentation.

## Architecture

All agents inherit from `BaseAgent` (`ipw/agents/base.py`) and are registered with `AgentRegistry`:

```python
class BaseAgent:
    def __init__(
        self,
        mcp_tools: dict[str, BaseMCPServer] | None = None,
        event_recorder: EventRecorder | None = None,
    ) -> None:
        self.mcp_tools = mcp_tools or {}
        self.event_recorder = event_recorder

    def run(self, input: str, **kwargs) -> Any:
        raise NotImplementedError
```

### Key Concepts

**EventRecorder** (`ipw/telemetry/events.py`): A thread-safe recorder that captures timestamped events during agent execution. Events are correlated with energy telemetry readings to compute per-action energy costs.

Supported event types:

| Event Type | Description |
|------------|-------------|
| `lm_inference_start` / `lm_inference_end` | LLM call boundaries |
| `tool_call_start` / `tool_call_end` | Tool invocation boundaries |
| `prefill_start` / `prefill_end` | Prefill phase (if detectable) |
| `decode_start` / `decode_end` | Decode phase (if detectable) |
| `submodel_call_start` / `submodel_call_end` | Sub-model calls from MCP tools |

**AgentRunResult** (`ipw/core/types.py`): The return type from `agent.run()`, capturing:

```python
@dataclass(slots=True)
class AgentRunResult:
    content: str                          # Final response text
    tool_calls_attempted: int = 0         # Number of tool calls made
    tool_calls_succeeded: int = 0         # Number that succeeded
    tool_names_used: list[str] = field(default_factory=list)
    num_turns: int = 0                    # Total agent turns
    input_tokens: int = 0                # Total input tokens
    output_tokens: int = 0               # Total output tokens
    cost_usd: float = 0.0               # API cost
    trace: QueryTrace | None = None      # Full trace (if available)
```

**MCP Tools** (`ipw/agents/mcp/`): Model Context Protocol servers that provide tool capabilities to agents. Each MCP server wraps an external service (inference API, retrieval system) with a standard interface.

## Available Agents

| Agent ID | Framework | Install Extra | Use Case |
|----------|-----------|---------------|----------|
| `react` | [Agno](https://github.com/agno-agi/agno) | `ipw[react]` | General tool-augmented reasoning |
| `openhands` | [OpenHands SDK](https://github.com/All-Hands-AI/OpenHands) | `ipw[openhands]` | Autonomous task execution, coding |
| `terminus` | [terminal-bench](https://github.com/terminal-bench/terminal-bench) | `ipw[terminus]` | Terminal/CLI task benchmarking |

## Registration

Agents self-register using the `@AgentRegistry.register("id")` decorator:

```python
from ipw.agents.base import BaseAgent
from ipw.core.registry import AgentRegistry
from ipw.core.types import AgentRunResult


@AgentRegistry.register("my-agent")
class MyAgent(BaseAgent):
    def run(self, input: str, **kwargs) -> AgentRunResult:
        self._record_event("lm_inference_start", model="my-model")
        try:
            # Agent logic here
            result = self._do_reasoning(input)
            return AgentRunResult(content=result)
        finally:
            self._record_event("lm_inference_end", model="my-model")
```

## Telemetry Instrumentation

Agents instrument their execution by calling `self._record_event()` at key boundaries. The `BaseAgent` class provides this method, which delegates to the attached `EventRecorder`:

```python
def _record_event(self, event_type: str, **metadata) -> None:
    if self.event_recorder is not None:
        self.event_recorder.record(event_type, **metadata)
```

This design means telemetry is optional -- agents work identically with or without an `EventRecorder` attached.

## MCP Tool Servers

The `ipw/agents/mcp/` directory contains tool servers that agents can use:

### Inference Servers

Wrap LLM APIs so agents can make sub-queries:

- `OpenAIServer` -- calls the OpenAI API
- `AnthropicServer` -- calls the Anthropic API
- `GeminiServer` -- calls the Google Gemini API
- `OllamaServer` -- calls a local Ollama instance
- `VLLMServer` -- calls a local vLLM instance
- `OpenRouterServer` -- calls the OpenRouter API

### Retrieval Servers

Provide document retrieval for RAG-style agents:

- `BM25Server` -- sparse BM25 retrieval
- `DenseServer` -- dense vector retrieval
- `GrepServer` -- grep-based text search
- `HybridServer` -- combined BM25 + dense retrieval

All servers implement `BaseMCPServer` from `ipw/agents/mcp/base.py` and are registered in the `ToolRegistry` at `ipw/agents/mcp/tool_registry.py`.
