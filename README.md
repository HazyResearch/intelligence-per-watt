# TrafficBench

A benchmarking suite for LLM inference systems. TrafficBench sends workloads to your inference service and collects detailed telemetry—energy consumption, power usage, memory, temperature, and latency—to help you optimize performance and compare hardware configurations.

## Installation

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Build energy monitoring
uv run scripts/build_energy_monitor.py

# Install TrafficBench
uv pip install -e trafficbench
```

## Quick Start

```bash
# 1. List available inference clients
trafficbench list clients

# 2, Run a benchmark
trafficbench profile \
  --client ollama \
  --model llama3.2:1b \
  --client-base-url http://localhost:11434

# 3. Analyze the results
trafficbench analyze ./runs/profile_*

# 4. Generate plots
trafficbench plot ./runs/profile_*
```

**What gets measured:** For each query, TrafficBench captures energy consumption, power draw, GPU/CPU memory usage, temperature, time-to-first-token, throughput, and token counts.

## Commands

### `trafficbench profile`

Sends prompts to your service and measures performance.

```bash
trafficbench profile --client <client> --model <model> [options]
```

**Options:**
- `--client` - Inference client (e.g., `ollama`, `vllm`)
- `--model` - Model name
- `--client-base-url` - Service URL (e.g., `http://localhost:11434`)
- `--dataset` - Workload dataset (default: `trafficbench`)
- `--max-queries` - Limit queries for testing
- `--output-dir` - Where to save results

Example:
```bash
trafficbench profile \
  --client ollama \
  --model llama3.2:1b \
  --client-base-url http://localhost:11434 \
  --max-queries 100
```

### `trafficbench analyze`

Compute regression metrics (e.g., how energy scales with tokens, latency vs. input size).

```bash
trafficbench analyze <results_dir>
```

### `trafficbench plot`

Visualize profiling data (scatter plots, regression lines, distributions).

```bash
trafficbench plot <results_dir> [--output <dir>]
```

### `trafficbench list`

Discover available clients, datasets, and analysis types.

```bash
trafficbench list <clients|datasets|analyses|visualizations|all>
```

### `trafficbench energy`

Test energy monitoring hardware (verify your system can collect power metrics).

```bash
trafficbench energy [--interval 2.0]
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
