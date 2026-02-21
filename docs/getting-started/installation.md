# Installation

## Prerequisites

Before installing Intelligence Per Watt, ensure you have:

- **Python >= 3.13** -- managed with [uv](https://docs.astral.sh/uv/)
- **Rust compiler** -- for building the energy monitor ([install](https://www.rust-lang.org/tools/install))
- **Protocol Buffer compiler** (`protoc`) -- for gRPC proto compilation ([install](https://protobuf.dev/installation/))
- **An inference runtime** -- [Ollama](https://ollama.ai/) or [vLLM](https://docs.vllm.ai/) for local models, or an OpenAI-compatible API

See [Prerequisites](prerequisites.md) for platform-specific details.

## Base Install

Clone the repository and set up the Python environment:

```bash
git clone https://github.com/HazyResearch/intelligence-per-watt.git
cd intelligence-per-watt

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Build the Rust energy monitor and stage the binary
uv run scripts/build_energy_monitor.py

# Install the base package
uv pip install -e intelligence-per-watt
```

The base install includes the CLI (`ipw`), core profiling engine, telemetry collector, and analysis tools. Inference clients and agent harnesses are installed separately as extras.

## Extras

IPW uses optional dependencies to keep the base install lightweight. Install each extra you need:

### Inference Clients

```bash
# Ollama client
uv pip install -e 'intelligence-per-watt[ollama]'

# vLLM offline client
uv pip install -e 'intelligence-per-watt[vllm]'
```

The OpenAI client (used for LLM judge evaluation) ships with the base install.

### Agent Harnesses

```bash
# ReAct agent (Agno framework)
uv pip install -e 'intelligence-per-watt[react]'

# OpenHands agent (SDK)
uv pip install -e 'intelligence-per-watt[openhands]'

# Terminus agent (Docker + tmux terminal agent)
uv pip install -e 'intelligence-per-watt[terminus]'

# All agents at once
uv pip install -e 'intelligence-per-watt[agents]'
```

### Additional Features

```bash
# Tavily web search tool for agents
uv pip install -e 'intelligence-per-watt[tavily]'

# FLOPs estimation via calflops
uv pip install -e 'intelligence-per-watt[flops]'

# Documentation site dependencies
uv pip install -e 'intelligence-per-watt[docs]'

# Everything (all clients, agents, and tools)
uv pip install -e 'intelligence-per-watt[all]'
```

## Building the Energy Monitor

The energy monitor is a Rust gRPC service that must be compiled for your platform. The build script handles compilation and stages the binary into the Python package:

```bash
uv run scripts/build_energy_monitor.py
```

This compiles the Rust binary from `energy-monitor/` and places it at `intelligence-per-watt/src/ipw/telemetry/bin/<platform>/energy-monitor`.

To verify the energy monitor works on your system:

```bash
uv run scripts/test_energy_monitor.py [--interval 2.0]
```

## Environment Variables

Create a `.env` file in the project root for API keys:

```bash
# Required for LLM judge evaluation (accuracy scoring)
IPW_EVAL_API_KEY=sk-...
# Or use the OpenAI key directly
OPENAI_API_KEY=sk-...

# Optional: Anthropic API key (for Anthropic-hosted models)
ANTHROPIC_API_KEY=sk-ant-...

# Optional: Tavily API key (for web search tools in agents)
TAVILY_API_KEY=tvly-...
```

IPW loads `.env` automatically via `python-dotenv`.

## Verifying the Installation

After installation, verify everything is working:

```bash
# Check the CLI is available
ipw --help

# List available components
ipw list all

# Test energy monitoring
uv run scripts/test_energy_monitor.py
```
