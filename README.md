# Intelligence Per Watt

<p align="center">
  <img src="assets/intelligence_per_watt_mood.png" width="500" alt="Intelligence Per Watt">
</p>

![Python](https://img.shields.io/badge/python-%3E%3D3.13-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)

A benchmarking suite for LLM inference systems that measures accuracy alongside energy consumption, power usage, memory, temperature, and latency. Intelligence Per Watt introduces two key efficiency metrics:

- **Intelligence Per Joule (IPJ)** = accuracy / average energy per query
- **Intelligence Per Watt (IPW)** = accuracy / average power per query

## Features

- **Single-turn profiling** -- Profile any OpenAI-compatible inference server with per-query energy telemetry
- **Agentic profiling** -- Profile multi-turn agent workloads (ReAct, OpenHands, Terminus) with per-step trace collection
- **10+ benchmark datasets** -- From knowledge tests (MMLU-Pro, SuperGPQA) to agentic tasks (GAIA, FRAMES, HLE) and coding (SWE-bench)
- **Cross-platform energy telemetry** -- Rust gRPC service supporting NVIDIA (NVML), AMD (ROCm), Apple Silicon (powermetrics), and Linux RAPL
- **Cost tracking** -- Built-in pricing tables for OpenAI, Anthropic, and Google Gemini APIs
- **FLOPs estimation** -- Computational cost via model parameter lookup or calflops
- **MCP tool integration** -- Agent tools for inference servers, retrieval systems, and web search

## Installation

### Prerequisites
- [Python >= 3.13](https://www.python.org/) (managed with [uv](https://docs.astral.sh/uv/))
- [Rust compiler](https://www.rust-lang.org/tools/install) (for building energy monitor)
- [Protocol Buffer compiler](https://protobuf.dev/installation/) (`protoc`)
- [Ollama](https://ollama.ai/) or [vLLM](https://docs.vllm.ai/) (inference client)

### Setup
```bash
git clone https://github.com/HazyResearch/intelligence-per-watt.git
cd intelligence-per-watt

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Build energy monitoring
uv run scripts/build_energy_monitor.py

# Install Intelligence Per Watt
uv pip install -e intelligence-per-watt
```

### Extras

Install optional components as needed:

```bash
# Inference clients
uv pip install -e 'intelligence-per-watt[ollama]'
uv pip install -e 'intelligence-per-watt[vllm]'

# Agent harnesses
uv pip install -e 'intelligence-per-watt[react]'       # ReAct (Agno framework)
uv pip install -e 'intelligence-per-watt[openhands]'    # OpenHands SDK
uv pip install -e 'intelligence-per-watt[terminus]'     # Terminus (Docker + tmux)
uv pip install -e 'intelligence-per-watt[agents]'       # All agents

# Additional features
uv pip install -e 'intelligence-per-watt[tavily]'       # Web search tool
uv pip install -e 'intelligence-per-watt[flops]'        # FLOPs estimation (calflops)
uv pip install -e 'intelligence-per-watt[all]'          # Everything
```

Set up API keys in a `.env` file:

```bash
IPW_EVAL_API_KEY=sk-...       # For LLM judge evaluation
OPENAI_API_KEY=sk-...         # Alternative for OpenAI
ANTHROPIC_API_KEY=sk-ant-...  # For Anthropic models
TAVILY_API_KEY=tvly-...       # For web search tools
```

## Quick Start

### Single-Turn Profiling

```bash
# 1. List available inference clients
ipw list clients

# 2. Run a benchmark
ipw profile \
  --client ollama \
  --model llama3.2:1b \
  --client-base-url http://localhost:11434

# 3. Analyze the results
ipw analyze ./runs/profile_*

# 4. Generate plots
ipw plot ./runs/profile_*
```

### Agentic Profiling

```bash
# Run an agentic benchmark
ipw run \
  --agent react \
  --model gpt-4o \
  --dataset gaia \
  --max-queries 10

# Analyze results
ipw analyze ./runs/run_*
```

**What gets measured:** For each query, Intelligence Per Watt captures energy consumption, power draw, GPU/CPU memory usage, temperature, time-to-first-token, throughput, token counts, API cost, and FLOPs.

## Commands

### `ipw profile`

Send prompts to the device, profile hardware usage, and calculate IPW/IPJ.

```bash
ipw profile --client <client> --model <model> [options]
```

**Options:**
- `--client` - Inference client (e.g., `ollama`, `vllm`)
- `--model` - Model name
- `--client-base-url` - Client base URL
- `--dataset` - Workload dataset (default: `ipw`)
- `--max-queries` - Limit queries for testing
- `--eval-client` - Judge client for scoring (default: `openai`)
- `--eval-model` - Judge model (default: `gpt-5-nano-2025-08-07`)
- `--output-dir` - Where to save results

### `ipw run`

Run agentic workloads with multi-turn trace collection.

```bash
ipw run --agent <agent> --model <model> --dataset <dataset> [options]
```

**Options:**
- `--agent` - Agent harness (`react`, `openhands`, `terminus`)
- `--model` - Model for the agent's LLM backbone
- `--dataset` - Benchmark dataset
- `--max-queries` - Limit number of tasks
- `--max-turns` - Maximum agent turns per task (default: 20)

### `ipw analyze`

Compute accuracy and efficiency metrics, or fit regression curves.

```bash
ipw analyze <results_dir>
ipw analyze <results_dir> --analysis regression
```

### `ipw plot`

Visualize profiling data (scatter plots, regression lines, distributions).

```bash
ipw plot <results_dir> [--output <dir>]
```

### `ipw list`

Discover available components.

```bash
ipw list <clients|datasets|analyses|visualizations|all>
```

## Datasets

### Single-Turn
| ID | Name | Size | Evaluation |
|----|------|------|------------|
| `ipw` | IPW Mixed 1K | 1,000 | LLM judge |
| `mmlu-pro` | MMLU-Pro | ~12,000 | MCQ exact match |
| `supergpqa` | SuperGPQA | varies | MCQ exact match |
| `gpqa` | GPQA | varies | MCQ exact match |
| `math500` | MATH-500 | 500 | Exact match |
| `natural-reasoning` | Natural Reasoning | varies | LLM judge |
| `wildchat` | WildChat | varies | LLM judge |

### Agentic
| ID | Name | Source | Evaluation |
|----|------|--------|------------|
| `gaia` | GAIA | `gaia-benchmark/GAIA` | LLM judge |
| `simpleqa` | SimpleQA | `basicv8vc/SimpleQA` | LLM judge |
| `frames` | FRAMES | `google/frames-benchmark` | LLM judge |
| `hle` | HLE | `cais/hle` | LLM judge |
| `terminalbench` | TerminalBench | `terminal-bench/terminal-bench` | Terminal check |

### Coding
| ID | Name | Source | Evaluation |
|----|------|--------|------------|
| `swebench` | SWE-bench | `princeton-nlp/SWE-bench_Verified` | Test execution |
| `swefficiency` | SWEfficiency | HuggingFace | Speedup measurement |

## Agents

Three agent harnesses are built in, each wrapping an existing framework with energy telemetry:

| Agent | Framework | Install | Use Case |
|-------|-----------|---------|----------|
| `react` | [Agno](https://github.com/agno-agi/agno) | `ipw[react]` | Tool-augmented reasoning |
| `openhands` | [OpenHands SDK](https://github.com/All-Hands-AI/OpenHands) | `ipw[openhands]` | Autonomous task execution |
| `terminus` | [terminal-bench](https://github.com/terminal-bench/terminal-bench) | `ipw[terminus]` | Terminal/CLI tasks |

Agents capture per-turn telemetry via `EventRecorder`, recording LLM inference boundaries and tool call boundaries for energy attribution.

## Architecture

```
Python Package (ipw/)
    cli/                  Click-based CLI (ipw command)
    core/                 Registry pattern, shared types (DatasetRecord, Response, TelemetryReading)
    clients/              Inference adapters (Ollama, vLLM, OpenAI)
    datasets/             Dataset providers (IPW, MMLU-Pro, GAIA, SWE-bench, ...)
    agents/               Agent harnesses (ReAct, OpenHands, Terminus)
        mcp/              MCP tool servers (inference, retrieval)
    evaluation/           Scoring handlers (LLM judge, MCQ match, exact match)
    analysis/             Post-run analysis (accuracy/IPJ/IPW, regression)
    visualization/        Plotting (regression scatter, output KDE)
    execution/            ProfilerRunner orchestrator, TelemetrySession, traces
    telemetry/            Energy monitor launcher + gRPC collector
    cost/                 API pricing tables and cost calculation
    compute/              FLOPs estimation

Rust Service (energy-monitor/)
    collectors/           Platform collectors (NVIDIA, AMD, macOS, RAPL)
    proto/                gRPC proto definition (energy.proto)
    server                gRPC streaming server (50ms intervals)
```

**Registry pattern**: All components self-register via decorators (`@ClientRegistry.register("id")`, `@DatasetRegistry.register("id")`, etc.). The CLI resolves components by string key through registries. Registries: `ClientRegistry`, `DatasetRegistry`, `AgentRegistry`, `AnalysisRegistry`, `VisualizationRegistry`, `EvaluationRegistry`.

**Data flow**:
```
Dataset -> InferenceClient -> Response + TelemetryWindow -> ProfilingRecord -> Arrow dataset
                                                                                   |
                                                            EvaluationHandler -> scored dataset
                                                                                   |
                                                            AnalysisProvider -> IPJ/IPW metrics
```

## Energy Telemetry

The energy monitor is a Rust gRPC service streaming telemetry at 50ms intervals. It auto-detects the platform collector:

| Platform | Collector | GPU Power/Energy | CPU Power/Energy | Temperature | Utilization |
|----------|-----------|:---:|:---:|:---:|:---:|
| NVIDIA | NVML | yes | via RAPL | yes | yes |
| AMD | ROCm SMI | yes | via RAPL | yes | yes |
| Apple Silicon | powermetrics | yes | yes + ANE | -- | -- |
| Linux (CPU-only) | RAPL | -- | yes | -- | -- |

## Cost Tracking

Built-in pricing tables cover cloud API models:

- **OpenAI**: GPT-4o, GPT-4o-mini, o1, GPT-5 series
- **Anthropic**: Claude 4.5 Opus/Sonnet/Haiku, Claude 4, Claude 3.5, Claude 3
- **Google**: Gemini 3.0 Flash, Gemini 2.0, Gemini 1.5 Pro/Flash
- **Tools**: Tavily web search ($0.01/search)

Cost is computed per query and per turn, stored alongside energy and latency metrics.

## Output

Profiling runs save to `./runs/profile_<hardware>_<model>/`:

```
runs/profile_<hardware>_<model>/
    data-*.arrow        # Per-query metrics (HuggingFace dataset format)
    summary.json        # Run metadata and totals
    analysis/           # Regression coefficients, statistics
    plots/              # Graphs
```

Agentic runs additionally produce `traces.jsonl` with per-turn `TurnTrace` / `QueryTrace` data.

## Running Tests

```bash
pytest intelligence-per-watt                            # Full test suite
pytest intelligence-per-watt/src/ipw/tests/             # All tests explicitly
pytest intelligence-per-watt/src/ipw/tests/clients/     # Single module
pytest intelligence-per-watt/src/ipw/tests/core/test_registry.py  # Single file
```

## Energy Monitor Testing

```bash
uv run scripts/test_energy_monitor.py [--interval 2.0]
```

## Citation

If you use Intelligence Per Watt in your research, please cite:

```bibtex
@misc{saadfalcon2025intelligencewattmeasuringintelligence,
      title={Intelligence per Watt: Measuring Intelligence Efficiency of Local AI},
      author={Jon Saad-Falcon and Avanika Narayan and Hakki Orhun Akengin and J. Wes Griffin and Herumb Shandilya and Adrian Gamarra Lafuente and Medhya Goel and Rebecca Joseph and Shlok Natarajan and Etash Kumar Guha and Shang Zhu and Ben Athiwaratkun and John Hennessy and Azalia Mirhoseini and Christopher Ré},
      year={2025},
      eprint={2511.07885},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2511.07885},
}
```

## Sponsors

Intelligence Per Watt is supported by

- [Laude Institute](https://www.laude.org/)
- [Stanford Marlowe](https://datascience.stanford.edu/marlowe)
- [Google Cloud Platform](https://cloud.google.com/)
- [Lambda Labs](https://lambda.ai/)
