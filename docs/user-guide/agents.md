# Agents

## Overview

IPW profiles multi-turn agent workloads through pluggable agent harnesses. Each agent wraps an existing framework and adds energy telemetry instrumentation. All agents inherit from `BaseAgent` and are registered with `AgentRegistry`.

### Available Agents

| Agent ID | Framework | Install Extra | Use Case |
|----------|-----------|---------------|----------|
| `react` | [Agno](https://github.com/agno-agi/agno) | `ipw[react]` | General tool-augmented reasoning |
| `openhands` | [OpenHands SDK](https://github.com/All-Hands-AI/OpenHands) | `ipw[openhands]` | Autonomous task execution, coding |
| `terminus` | [terminal-bench](https://github.com/terminal-bench/terminal-bench) | `ipw[terminus]` | Terminal/CLI task benchmarking |
| `terminus-tb` | [terminal-bench](https://github.com/terminal-bench/terminal-bench) | `ipw[terminus]` | TerminalBench native (Docker managed by runner) |

### Event Types

The `EventRecorder` captures timestamped events during agent execution, correlated with energy telemetry to compute per-action energy costs.

| Event Type | Description |
|------------|-------------|
| `lm_inference_start` / `lm_inference_end` | LLM call boundaries |
| `tool_call_start` / `tool_call_end` | Tool invocation boundaries |
| `prefill_start` / `prefill_end` | Prefill phase (if detectable) |
| `decode_start` / `decode_end` | Decode phase (if detectable) |
| `submodel_call_start` / `submodel_call_end` | Sub-model calls from MCP tools |

---

## ReAct (Agno)

```bash
uv pip install -e 'intelligence-per-watt[react]'
```

The ReAct agent uses [Agno](https://github.com/agno-agi/agno) to implement Reasoning + Acting (ReAct) style tool-augmented reasoning. It wraps Agno's `Agent` class and instruments tool calls for energy tracking.

```bash
ipw run \
  --agent react \
  --model gpt-4o \
  --dataset gaia \
  --max-queries 10
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | Any | required | Agno Model instance (e.g., `OpenAIChat`) |
| `tools` | list[Callable] | required | List of callable tool functions |
| `instructions` | str | built-in | Custom system instructions |
| `max_turns` | int | -- | Maximum reasoning iterations |

---

## OpenHands

```bash
uv pip install -e 'intelligence-per-watt[openhands]'
```

The OpenHands agent uses the [OpenHands SDK](https://github.com/All-Hands-AI/OpenHands) for autonomous task execution with per-tool energy tracking. It is designed for complex, multi-step tasks such as software engineering, research, and document analysis.

```bash
ipw run \
  --agent openhands \
  --model gpt-4o \
  --dataset swebench \
  --max-turns 30
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | Any | required | LLM model instance |
| `tools` | list | None | OpenHands Tool specs |
| `mcp_tools` | dict | None | MCP server instances for sub-queries |
| `max_turns` | int | 20 | Maximum iterations per run |

---

## Terminus

```bash
uv pip install -e 'intelligence-per-watt[terminus]'
```

The Terminus agent uses [terminal-bench](https://github.com/terminal-bench/terminal-bench) to run tasks inside Docker containers with tmux, enabling benchmarking of terminal/CLI task execution.

**Prerequisites:** Docker Engine installed and running; current user in the `docker` group (or sudo access).

```bash
ipw run \
  --agent terminus \
  --model gpt-4o \
  --dataset terminalbench \
  --max-queries 10
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | required | Model name (e.g., `"gpt-4o"`) |
| `docker_image` | str | `"ubuntu:22.04"` | Docker image for the container |
| `container_name` | str | `"terminus-container"` | Name for the Docker container |
| `max_turns` | int | -- | Maximum agent turns |

---

## MCP Tools

Model Context Protocol servers provide tool capabilities to agents. Each MCP server wraps an external service with a standard interface.

**Inference servers** -- wrap LLM APIs so agents can make sub-queries:

| Server | Backend |
|--------|---------|
| `OpenAIServer` | OpenAI API |
| `AnthropicServer` | Anthropic API |
| `GeminiServer` | Google Gemini API |
| `OllamaServer` | Local Ollama instance |
| `VLLMServer` | Local vLLM instance |
| `OpenRouterServer` | OpenRouter API |

**Retrieval servers** -- provide document retrieval for RAG-style agents:

| Server | Method |
|--------|--------|
| `BM25Server` | Sparse BM25 retrieval |
| `DenseServer` | Dense vector retrieval |
| `GrepServer` | Grep-based text search |
| `HybridServer` | Combined BM25 + dense retrieval |

---

## Writing a Custom Agent

See [Extending IPW](../extending/index.md) for how to implement and register a custom agent.
