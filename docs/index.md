# Intelligence Per Watt

**A benchmarking suite for LLM inference systems that measures accuracy alongside energy consumption, power, memory, temperature, and latency.**

Intelligence Per Watt (IPW) introduces two key metrics for evaluating the efficiency of LLM inference:

- **Intelligence Per Joule (IPJ)** = accuracy / average energy per query (joules)
- **Intelligence Per Watt (IPW)** = accuracy / average power per query (watts)

These metrics let you compare models and hardware configurations on a single axis that captures both correctness and energy cost.

## Key Features

### Single-Turn Profiling

Profile any OpenAI-compatible inference server. IPW sends workloads from curated datasets, captures per-query energy telemetry via a Rust gRPC service, and computes efficiency metrics automatically.

```bash
ipw profile --client ollama --model llama3.2:1b --client-base-url http://localhost:11434
```

### Agentic Profiling

Profile multi-turn agent workloads with tool use. Three agent harnesses are built in -- ReAct (via Agno), OpenHands, and Terminus -- each with per-step trace collection and energy attribution.

```bash
ipw run --agent react --model gpt-4o --dataset gaia --max-queries 10
```

### 10+ Benchmark Datasets

From single-turn knowledge tests (MMLU-Pro, SuperGPQA, GPQA) to agentic benchmarks (GAIA, FRAMES, HLE, SimpleQA, TerminalBench) and coding tasks (SWE-bench, SWEfficiency).

### Cross-Platform Energy Telemetry

A standalone Rust gRPC service streams telemetry at 50ms intervals. Auto-detects the platform collector:

| Platform | Collector | GPU Metrics | CPU Metrics |
|----------|-----------|-------------|-------------|
| NVIDIA | NVML | Power, energy, temp, utilization, memory | Via RAPL |
| AMD | ROCm SMI | Power, energy, temp, utilization, memory | Via RAPL |
| Apple Silicon | powermetrics | GPU power/energy | CPU + ANE power/energy |
| Linux (CPU-only) | RAPL | -- | Package + core energy |

### Cost Tracking

Built-in pricing tables for OpenAI, Anthropic, and Google Gemini APIs. Per-query and per-turn cost is tracked automatically for cloud-hosted models.

### FLOPs Estimation

Estimate computational cost using known model parameter counts (2*P*T formula) or the optional `calflops` library for HuggingFace models.

## Quick Links

- [Installation](getting-started/installation.md) -- get IPW running on your system
- [Quickstart](getting-started/quickstart.md) -- run your first benchmark in minutes
- [Profiling Guide](user-guide/profiling.md) -- deep dive on `ipw profile`
- [Agentic Profiling](user-guide/agentic-profiling.md) -- profile multi-turn agents with `ipw run`
- [Datasets Overview](datasets/overview.md) -- browse available benchmark datasets
- [Platform Support](telemetry/platform-support.md) -- check what metrics are available on your hardware

## Architecture Overview

```
Dataset --> InferenceClient --> Response + TelemetryWindow --> ProfilingRecord --> Arrow dataset
                                                                                     |
                                                              EvaluationHandler (LLM judge) --> scored dataset
                                                                                     |
                                                              AnalysisProvider --> IPJ/IPW metrics + JSON report
```

All components (clients, datasets, agents, evaluators, analyses, visualizations) are registered via a decorator-based registry pattern. The CLI resolves them by string key, so extending IPW requires only implementing an ABC and adding a `@Registry.register("id")` decorator.

## Output Structure

Each profiling run produces a self-contained results directory:

```
runs/profile_<hardware>_<model>_<dataset>/
    data-*.arrow        # Per-query metrics (HuggingFace dataset format)
    summary.json        # Run metadata and configuration
    analysis/           # JSON reports (accuracy, regression)
    plots/              # Visualizations (scatter, KDE)
```
