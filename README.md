# Intelligence Per Watt

A benchmarking suite for LLM inference systems. Intelligence Per Watt sends workloads to your inference service and collects detailed telemetry—energy consumption, power usage, memory, temperature, and latency—to help you optimize performance and compare hardware configurations.

## Installation

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Build energy monitoring
uv run scripts/build_energy_monitor.py

# Install Intelligence Per Watt
uv pip install -e intelligence-per-watt
```

## Quick Start

```bash
# 1. List available inference clients
ipw list clients

# 2, Run a benchmark
ipw profile \
  --client ollama \
  --model llama3.2:1b \
  --client-base-url http://localhost:11434

# 3. Analyze the results
ipw analyze ./runs/profile_*

# 4. Generate plots
ipw plot ./runs/profile_*
```

**What gets measured:** For each query, Intelligence Per Watt captures energy consumption, power draw, GPU/CPU memory usage, temperature, time-to-first-token, throughput, and token counts.

## Commands

### `ipw profile`

Sends prompts to your service and measures performance.

```bash
ipw profile --client <client> --model <model> [options]
```

**Options:**
- `--client` - Inference client (e.g., `ollama`, `vllm`)
- `--model` - Model name
- `--client-base-url` - Service URL (e.g., `http://localhost:11434`)
- `--dataset` - Workload dataset (default: `ipw`)
- `--max-queries` - Limit queries for testing
- `--output-dir` - Where to save results

Example:
```bash
ipw profile \
  --client ollama \
  --model llama3.2:1b \
  --client-base-url http://localhost:11434 \
  --max-queries 100
```

### `ipw analyze`

Compute regression metrics (e.g., how energy scales with tokens, latency vs. input size).

```bash
ipw analyze <results_dir>
```

### `ipw plot`

Visualize profiling data (scatter plots, regression lines, distributions).

```bash
ipw plot <results_dir> [--output <dir>]
```

### `ipw list`

Discover available clients, datasets, and analysis types.

```bash
ipw list <clients|datasets|analyses|visualizations|all>
```

### Energy monitor test script

Validate that your system can collect energy telemetry before running full workloads.

```bash
uv run scripts/test_energy_monitor.py [--interval 2.0]
```

## Output

Profiling runs save to `./runs/profile_<hardware>_<model>/`:

```
runs/profile_<hardware>_<model>/
├── data-*.arrow        # Per-query metrics (HuggingFace dataset format)
├── summary.json        # Run metadata and totals
├── analysis/           # Regression coefficients, statistics
└── plots/              # Graphs
```
