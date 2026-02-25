# Single-Turn Profiling

The `ipw profile` command sends prompts from a dataset to an inference server one at a time, capturing energy telemetry for each query. This is the primary workflow for benchmarking single-turn LLM inference.

## Command Reference

```bash
ipw profile --client <client> --model <model> [options]
```

### Required Options

| Option | Description |
|--------|-------------|
| `--client` | Inference client ID (`ollama`, `vllm`) |
| `--model` | Model name as known to the inference server |

### Optional Options

| Option | Default | Description |
|--------|---------|-------------|
| `--client-base-url` | client-specific | Base URL of the inference server |
| `--dataset` | `ipw` | Dataset to use for prompts |
| `--max-queries` | all | Limit the number of queries |
| `--output-dir` | `./runs/` | Directory for results |
| `--eval-client` | `openai` | Client for LLM judge evaluation |
| `--eval-base-url` | `https://api.openai.com/v1` | Judge service URL |
| `--eval-model` | `gpt-5-nano-2025-08-07` | Model for evaluation judging |

## How It Works

The profiling flow is managed by `ProfilerRunner` in `ipw/execution/runner.py`:

1. **Resolve components** -- Look up the dataset and client from their registries by string ID.
2. **Launch energy monitor** -- Start the Rust gRPC energy monitor as a subprocess. Begin streaming telemetry at 50ms intervals.
3. **Prime hardware metadata** -- Collect initial telemetry samples to identify GPU, CPU, and platform information.
4. **Execute queries** -- For each dataset record:
    - Send the prompt to the inference client via `stream_chat_completion()`
    - Capture the telemetry window between request start and end
    - Build a `ProfilingRecord` with token counts, latency, energy, power, memory, and temperature
5. **Flush results** -- Write to HuggingFace Arrow dataset every 100 records. Write `summary.json` with run metadata.
6. **Post-profiling analysis** -- Run accuracy analysis using an LLM judge to score each response, then compute IPJ and IPW.

## Example Workflows

### Basic Ollama Profiling

```bash
# Profile Llama 3.2 1B on the default IPW dataset
ipw profile \
  --client ollama \
  --model llama3.2:1b \
  --client-base-url http://localhost:11434
```

### Limiting Query Count

Useful for quick testing:

```bash
ipw profile \
  --client ollama \
  --model llama3.2:1b \
  --client-base-url http://localhost:11434 \
  --max-queries 20
```

### Using a Specific Dataset

```bash
# Profile on MMLU-Pro academic questions
ipw profile \
  --client ollama \
  --model llama3.2:1b \
  --client-base-url http://localhost:11434 \
  --dataset mmlu-pro

# Profile on SuperGPQA
ipw profile \
  --client vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --client-base-url http://localhost:8000 \
  --dataset supergpqa
```

### Custom Evaluation Model

```bash
ipw profile \
  --client ollama \
  --model llama3.2:1b \
  --client-base-url http://localhost:11434 \
  --eval-model gpt-4o \
  --eval-base-url https://api.openai.com/v1
```

### Comparing Models

Run the same dataset against different models and compare:

```bash
# Profile model A
ipw profile --client ollama --model llama3.2:1b \
  --client-base-url http://localhost:11434

# Profile model B
ipw profile --client ollama --model llama3.2:3b \
  --client-base-url http://localhost:11434

# Compare results
ipw analyze ./runs/profile_*_llama3.2_1b*
ipw analyze ./runs/profile_*_llama3.2_3b*
```

## Output Structure

Each profiling run creates a results directory:

```
runs/profile_<hardware>_<model>_<dataset>/
    data-00000-of-00001.arrow   # Per-query metrics
    summary.json                # Run metadata
    analysis/
        accuracy.json           # IPJ/IPW and scoring results
        regression.json         # Energy/latency regression (if requested)
    plots/
        regression.png          # Scatter plots with regression lines
        output_kde.png          # Output length distribution
```

### summary.json

Contains run metadata including:

- Profiler configuration (client, model, dataset)
- Hardware information (GPU name, CPU brand, platform)
- Timing (start/end timestamps, total duration)
- Aggregate token counts

### Arrow Dataset

Each row in the Arrow dataset represents one query and contains:

- `problem` -- the input prompt
- `answer` -- the reference answer
- `model_answers` -- map of model name to generated response
- `model_metrics` -- map of model name to `ModelMetrics` (energy, power, latency, memory, temperature, tokens, compute, cost)

## Telemetry Session

The `TelemetrySession` (in `ipw/execution/telemetry_session.py`) manages a threaded connection to the energy monitor. It maintains a rolling buffer of telemetry readings and supports time-window queries to extract the readings that correspond to a specific inference request.

Key behaviors:

- Samples are buffered in memory with timestamps
- `get_window(start_time, end_time)` returns all readings within that interval
- Thread-safe for concurrent access
- Automatically reconnects if the gRPC stream drops
