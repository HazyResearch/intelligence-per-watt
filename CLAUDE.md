# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Intelligence Per Watt (IPW) is a benchmarking suite for LLM inference systems that measures accuracy alongside energy consumption, power, memory, temperature, and latency. It introduces Intelligence Per Joule (IPJ) and Intelligence Per Watt (IPW) metrics (accuracy / energy per query).

Dual-language project: Python package for orchestration + Rust gRPC service for energy telemetry.

## Build & Development Commands

### Initial Setup
```bash
uv venv && source .venv/bin/activate
uv run scripts/build_energy_monitor.py          # Compiles Rust energy monitor, stages binary
uv pip install -e intelligence-per-watt          # Install Python package
uv pip install -e 'intelligence-per-watt[ollama]'  # With Ollama client
uv pip install -e 'intelligence-per-watt[vllm]'    # With vLLM client
```

### Prerequisites
- Rust compiler + `protoc` (Protocol Buffer compiler)
- Python >=3.13, managed with `uv`
- Ollama or vLLM as inference runtime
- OpenAI-compatible API for LLM judge evaluation (set `IPW_EVAL_API_KEY` or `OPENAI_API_KEY` in `.env`)

### Running Tests
```bash
pytest intelligence-per-watt                             # Full test suite
pytest intelligence-per-watt/src/ipw/tests/              # All tests explicitly
pytest intelligence-per-watt/src/ipw/tests/clients/      # Single module
pytest intelligence-per-watt/src/ipw/tests/core/test_registry.py  # Single file
```

### CLI Usage
```bash
ipw profile --client ollama --model llama3.2:1b --client-base-url http://localhost:11434
ipw analyze ./runs/profile_*
ipw analyze ./runs/profile_* --analysis regression
ipw plot ./runs/profile_*
ipw list clients|datasets|analyses|visualizations|all
```

### Energy Monitor Testing
```bash
uv run scripts/test_energy_monitor.py [--interval 2.0]
```

## Architecture

### Python Package (`intelligence-per-watt/src/ipw/`)

**Registry pattern** is the central extensibility mechanism. All components self-register via decorators on their classes (see `core/registry.py`). Registries: `ClientRegistry`, `DatasetRegistry`, `AnalysisRegistry`, `VisualizationRegistry`, `EvaluationRegistry`. The CLI and runner resolve components by string key through registries, never by direct import.

**Profiling flow** (`execution/runner.py` — `ProfilerRunner`):
1. Resolve dataset + client from registries
2. Launch energy-monitor subprocess, start gRPC telemetry stream
3. Prime hardware metadata from initial telemetry samples
4. For each dataset record: invoke inference client → capture telemetry window → build `ProfilingRecord`
5. Flush to HuggingFace Arrow dataset every 100 records + write `summary.json`
6. Post-profiling: run accuracy analysis (LLM judge scoring → IPJ/IPW computation)

**Agentic benchmarking flow** (`execution/agentic_runner.py` — `AgenticRunner`):
1. For each dataset record: create task environment (e.g. Docker container for TerminalBench)
2. Run agent with energy telemetry capture
3. Build `QueryTrace` with per-turn energy correlation and `ProfilingRecord`
4. Save per-query artifacts (response, metadata, extracted patch)
5. Supports concurrent execution via `agent_factory` + thread pool
6. Cost: uses pricing tables for cloud models; automatically sets `cost = 0.0` for local models (`localhost`/`127.0.0.1` base URLs)

**Key modules:**
- `agents/` — Agent harnesses (`BaseAgent` ABC). OpenHands (with TerminalBench Docker support), ReAct, Terminus. OpenHands reads token metrics from trajectory files inside Docker (`/agent-logs/*.json`) with terminal-output parsing as fallback.
- `clients/` — Inference adapters (`InferenceClient` ABC). Ollama, vLLM (offline), OpenAI (judge only). Optional deps loaded lazily.
- `datasets/` — Dataset providers (`DatasetProvider` ABC). Built-in IPW (1k mixed), MMLU-Pro, SuperGPQA.
- `evaluation/` — Scoring handlers per dataset. Use LLM-as-judge (default: `gpt-5-nano-2025-08-07`) or exact match depending on dataset type.
- `analysis/` — Post-run analysis. `accuracy.py` computes IPJ/IPW; `regression.py` fits energy/latency curves.
- `visualization/` — Plotting plugins (regression scatter, KDE).
- `execution/` — `ProfilerRunner` (single-turn profiling) and `AgenticRunner` (multi-turn agent benchmarking) orchestrators, `TelemetrySession` (threaded sampling with rolling buffer + time-window queries), `QueryTrace`/`TurnTrace` for per-turn energy-correlated traces.
- `core/types.py` — Shared data types: `DatasetRecord`, `Response`, `TelemetryReading`, `ProfilerConfig`.
- `execution/types.py` — Profiling data model: `ProfilingRecord`, `ModelMetrics`, `EnergyMetrics`, etc.
- `cli/model_presets.py` — Named presets for common models (e.g. `glm-4.7-flash`, `qwen35-27b`) mapping short names to `model_id` + vLLM launch arguments.

### Rust Energy Monitor (`energy-monitor/`)

Standalone gRPC service streaming telemetry at 50ms intervals. Auto-detects platform collector:
- macOS: `powermetrics` (requires `sudo`)
- Linux/Windows: NVML (NVIDIA) → ROCm SMI (AMD) → null collector
- Proto definition: `energy-monitor/proto/energy.proto`
- Prebuilt binaries staged to `ipw/telemetry/bin/{platform}/`

Python side: `telemetry/launcher.py` manages the subprocess lifecycle; `telemetry/collector.py` is the gRPC client.

### Data Flow
```
Dataset → InferenceClient → Response + TelemetryWindow → ProfilingRecord → Arrow dataset on disk
                                                                              ↓
                                                          EvaluationHandler (LLM judge) → scored dataset
                                                                              ↓
                                                          AnalysisProvider → IPJ/IPW metrics + JSON report
```

### Output Structure
```
runs/profile_<hardware>_<model>_<dataset>/
├── data-*.arrow        # HuggingFace dataset format
├── summary.json        # Run metadata
├── analysis/           # JSON reports
└── plots/              # Visualizations
```

## Adding New Components

All follow the same pattern — implement the ABC, register with the appropriate registry decorator:

- **Client**: subclass `InferenceClient` in `clients/`, register with `@ClientRegistry.register("id")`, add optional dep to `pyproject.toml`
- **Dataset**: subclass `DatasetProvider` in `datasets/`, register with `@DatasetRegistry.register("id")`
- **Evaluation**: subclass `EvaluationHandler` in `evaluation/`, register with `@EvaluationRegistry.register("id")`
- **Analysis**: subclass `AnalysisProvider` in `analysis/`, register with `@AnalysisRegistry.register("id")`
- **Visualization**: subclass `VisualizationProvider` in `visualization/`, register with `@VisualizationRegistry.register("id")`
- **Platform collector** (Rust): implement `TelemetryCollector` trait in `energy-monitor/src/collectors/`, register with `#[cfg(target_os)]`

## Key Conventions

- Unavailable telemetry metrics use `-1` or `None`; validate with `math.isfinite()` before arithmetic
- Dataset persistence uses atomic temp-directory-then-rename pattern
- `ProfilingRecord` and all metric types use `@dataclass(slots=True)` for performance
- CLI is Click-based (`cli/` module), entry point is `ipw` command
- Tests live alongside source at `src/ipw/tests/`, mirroring the package structure
