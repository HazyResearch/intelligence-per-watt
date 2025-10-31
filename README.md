# TrafficBench

Profiling tool for inference services.

## Installation

```bash
cd trafficbench
uv sync
cd ..
uv run scripts/build_energy_monitor.py
uv pip install -e trafficbench
```

## Commands

### profile

Run profiling against an inference service.

```bash
trafficbench profile \
  --client <client_id> \
  --model <model_name> \
  --dataset <dataset_id> \
  --client-base-url <url> \
  --output-dir <path> \
  --max-queries <n>
```

Options:
- `--client` - Client identifier (required). Use `trafficbench list` to see available clients.
- `--model` - Model name (required)
- `--dataset` - Dataset identifier (default: trafficbench)
- `--client-base-url` - Base URL for the inference service
- `--client-param` - Client parameters as key=value (repeatable)
- `--dataset-param` - Dataset parameters as key=value (repeatable)
- `--output-dir` - Output directory for results
- `--max-queries` - Limit number of queries to profile

Example:
```bash
trafficbench profile \
  --client ollama \
  --model llama3.2:1b \
  --client-base-url http://localhost:11434 \
  --max-queries 100
```

### analyze

Analyze profiling results and compute regression metrics.

```bash
trafficbench analyze <results_dir> \
  --analysis <analysis_name> \
  --option key=value
```

Options:
- `<results_dir>` - Directory containing profiling results (required)
- `--analysis` - Analysis type (default: regression)
- `--option` - Analysis options as key=value (repeatable)
- `--verbose` - Show detailed output

Example:
```bash
trafficbench analyze ./runs/profile_H200_model \
  --analysis regression \
  --option model=vllm::meta-llama/llama-3.1-8b
```

### plot

Generate visualization plots from analysis results.

```bash
trafficbench plot <results_dir> \
  --visualization <viz_id> \
  --output <output_dir> \
  --option key=value
```

Options:
- `<results_dir>` - Directory containing profiling results (required)
- `--visualization` - Visualization type (default: regression)
- `--output` - Output directory for plots (default: <results_dir>/plots)
- `--option` - Visualization options as key=value (repeatable)

Example:
```bash
trafficbench plot ./runs/profile_H200_model
```

### list

List available components.

```bash
trafficbench list <subcommand>
```

Subcommands:
- `clients` - List available inference clients
- `datasets` - List available datasets
- `analyses` - List available analysis types
- `visualizations` - List available visualization types
- `all` - List all components

Examples:
```bash
trafficbench list clients
trafficbench list all
```

### energy

Start energy monitor and display telemetry stream.

```bash
trafficbench energy --target <host:port> --interval <seconds>
```

Options:
- `--target` - Energy monitor gRPC target address (default: 127.0.0.1:50053)
- `--interval` - Seconds between printed samples (default: 1.0)

Example:
```bash
trafficbench energy --interval 2.0
```

Note: Energy monitor binary must be running separately (requires sudo for hardware access):
```bash
./trafficbench/src/trafficbench/telemetry/bin/macos-arm64/energy-monitor
```

## Workflow

1. List available clients and datasets:
```bash
trafficbench list clients
trafficbench list datasets
```

2. Run profiling:
```bash
trafficbench profile \
  --client ollama \
  --model llama3.2:1b \
  --client-base-url http://localhost:11434 \
  --max-queries 500
```

3. Analyze results:
```bash
trafficbench analyze ./runs/profile_* --analysis regression
```

4. Generate plots:
```bash
trafficbench plot ./runs/profile_*
```

## Output

Profiling creates a directory with:
- `data-*.arrow` - Raw profiling data (HuggingFace dataset format)
- `dataset_info.json` - Dataset metadata
- `summary.json` - Run summary and aggregate metrics
- `analysis/` - Analysis outputs
- `plots/` - Generated visualizations
