# Installation

## Requirements

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | >= 3.13 | Runtime |
| [uv](https://docs.astral.sh/uv/) | latest | Package management |
| [Rust](https://www.rust-lang.org/tools/install) | stable | Building the energy monitor |
| [protoc](https://protobuf.dev/installation/) | >= 3.0 | Protocol Buffer compiler for gRPC |

## Install

=== "Quick Setup"

    ```bash
    git clone https://github.com/HazyResearch/intelligence-per-watt.git
    cd intelligence-per-watt
    bash intelligence-per-watt/scripts/setup.sh   # (1)!
    source .venv/bin/activate
    ```

    1. Auto-installs `uv`, creates a Python 3.13 venv, installs the package, and builds the energy monitor. Pass extras as arguments: `bash intelligence-per-watt/scripts/setup.sh ollama react`

=== "Manual"

    ```bash
    git clone https://github.com/HazyResearch/intelligence-per-watt.git
    cd intelligence-per-watt
    uv venv && source .venv/bin/activate
    uv run scripts/build_energy_monitor.py
    uv pip install -e intelligence-per-watt
    ```

=== "From Source"

    ```bash
    git clone https://github.com/HazyResearch/intelligence-per-watt.git
    cd intelligence-per-watt
    uv venv && source .venv/bin/activate
    cd energy-monitor && cargo build --release && cd ..
    uv pip install -e intelligence-per-watt
    ```

### Extras

Install only what you need:

```bash
uv pip install -e 'intelligence-per-watt[ollama]'     # Ollama client
uv pip install -e 'intelligence-per-watt[vllm]'       # vLLM offline client
uv pip install -e 'intelligence-per-watt[react]'      # ReAct agent (Agno)
uv pip install -e 'intelligence-per-watt[openhands]'   # OpenHands agent
uv pip install -e 'intelligence-per-watt[terminus]'    # Terminus agent
uv pip install -e 'intelligence-per-watt[agents]'     # All agents
uv pip install -e 'intelligence-per-watt[tavily]'     # Tavily web search
uv pip install -e 'intelligence-per-watt[flops]'      # FLOPs estimation
uv pip install -e 'intelligence-per-watt[all]'        # Everything
```

## Platform Setup

=== "NVIDIA"

    Requires NVIDIA driver >= 525 (NVML ships with it).

    Telemetry: GPU power, energy, temperature, memory, utilization, tensor core utilization (Ampere+). Optional CPU energy via RAPL.

    ```bash
    # Enable RAPL CPU energy (optional, as root)
    chmod o+r /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj
    ```

=== "AMD"

    Requires ROCm >= 5.0 with `rocm-smi` accessible.

    Telemetry: GPU power, energy, temperature, memory, utilization. Optional CPU energy via RAPL.

=== "Apple Silicon"

    Requires macOS 13+ on M1/M2/M3/M4 with `sudo` access.

    Telemetry: GPU, CPU, and ANE power/energy via `powermetrics`. CPU memory usage.

    !!! note
        No GPU memory or utilization reporting (Apple Unified Memory). Requires password or passwordless sudo for `powermetrics`.

=== "Linux CPU-only"

    Falls back to RAPL for CPU energy, or a null collector (memory only) if RAPL is unavailable.

    ```bash
    # Load RAPL kernel module if needed
    sudo modprobe intel_rapl_common
    ```

## Inference Runtime

=== "Ollama"

    ```bash
    curl -fsSL https://ollama.ai/install.sh | sh
    ollama pull llama3.2:1b
    ollama serve
    ```

=== "vLLM"

    ```bash
    pip install vllm  # Requires NVIDIA GPU
    vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
    ```

=== "OpenAI API"

    No local setup needed -- set your API key:

    ```bash
    export OPENAI_API_KEY=sk-...
    ```

## Environment Variables

Create a `.env` file in the project root (loaded automatically via `python-dotenv`):

```bash
# Required for LLM judge evaluation
OPENAI_API_KEY=sk-...

# Optional
ANTHROPIC_API_KEY=sk-ant-...   # Anthropic models
TAVILY_API_KEY=tvly-...        # Web search in agents
```

## Verify Installation

```bash
ipw --help                              # CLI available
ipw list all                            # Registered components
uv run scripts/test_energy_monitor.py   # Hardware telemetry
pytest intelligence-per-watt            # Test suite
```
