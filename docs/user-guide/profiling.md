# Profiling

IPW supports two profiling modes: **single-turn** (`ipw profile`) for benchmarking raw inference, and **agentic** (`ipw run`) for multi-turn agent workloads with tool use. For full benchmark evaluations with efficiency scoring, see [`ipw bench`](#benchmarking-ipw-bench). For managing local inference servers, see [`ipw servers`](#server-management-ipw-servers).

---

## Single-Turn Profiling

Send prompts to an inference server one at a time, capturing energy telemetry for each query.

```bash
ipw profile --client <client> --model <model> [options]
```

### Required Options

| Option | Description |
|--------|-------------|
| `--client` | Inference client ID (`ollama`, `vllm`, `openai-server`, `mlx`, `afm`) |
| `--model` | Model name as known to the inference server |

Run `ipw list clients` to see which are available in your install; the
`mlx` and `afm` clients are Apple Silicon only and need their optional extras.

### Optional Options

| Option | Default | Description |
|--------|---------|-------------|
| `--client-base-url` | client-specific | Base URL of the inference server |
| `--dataset` | `ipw` | Dataset to use for prompts |
| `--max-queries` | all | Limit the number of queries |
| `--warmup-queries` | `3` | Number of warmup queries to discard before measurement (0 to disable) |
| `--output-dir` | `./runs/` | Directory for results |
| `--dataset-param` | none | Dataset params as `key=value` (repeatable) |
| `--client-param` | none | Client params as `key=value` (repeatable) |
| `--eval-client` | `openai` | Client for LLM judge evaluation |
| `--eval-base-url` | `https://api.openai.com/v1` | Judge service URL |
| `--eval-model` | `gpt-5-nano-2025-08-07` | Model for evaluation judging |

### Example Workflows

```bash
# Basic: profile Llama 3.2 1B via Ollama
ipw profile \
  --client ollama \
  --model llama3.2:1b \
  --client-base-url http://localhost:11434

# Quick test with limited queries
ipw profile \
  --client ollama \
  --model llama3.2:1b \
  --client-base-url http://localhost:11434 \
  --max-queries 20

# Use a specific dataset (MMLU-Pro)
ipw profile \
  --client vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --client-base-url http://localhost:8000 \
  --dataset mmlu-pro
```

### Apple Silicon: on-device backends

Both Apple backends run **in-process**, so `--client-base-url` is unused. Energy
is captured by the same system-wide `powermetrics` collector, which needs `sudo`.

```bash
# MLX: any model from the mlx-community Hub org
ipw profile --client mlx --model mlx-community/Qwen3-4B-bf16

# Apple Foundation Models (AFM 3), the model behind Apple Intelligence
ipw profile --client afm --model afm-3-core-advanced \
  --dataset fixed_length --dataset-param prompt_length=512 num_samples=6 \
  --client-param max_tokens=128
```

`--client afm` accepts three `--model` labels: `afm-3`, `afm-3-core` and
`afm-3-core-advanced`. **These only label the run.** The Foundation Models
framework's dynamic profile chooses AFM 3 Core (dense ~3B) or AFM 3 Core
Advanced (20B sparse MoE) based on the host device, and the SDK exposes no
override — so pick the label matching the tier you believe your device runs, and
check `client_info` in `summary.json` for the SDK version, context size and host
chip that produced the numbers.

AFM-specific `--client-param` keys:

| Param | Default | Description |
|-------|---------|-------------|
| `max_tokens` | SDK default | Maps to the SDK's `maximum_response_tokens` |
| `temperature` | SDK default | Only meaningful with `sampling=random` |
| `sampling` | `greedy` | `greedy` or `random`. Greedy by default for reproducible runs |
| `top`, `probability_threshold`, `seed` | none | `sampling=random` only |
| `instructions` | none | System prompt; counts against the 4096-token context |
| `use_case` | `general` | `general` or `content_tagging` |
| `guardrails` | `default` | `default` or `permissive_content_transformations` |

Two measurement caveats when comparing AFM against other backends:

- **Latency percentiles are per-chunk, not per-token.** The SDK streams
  cumulative text snapshots batching roughly 8–10 tokens each, so
  `time_to_first_token_seconds` is really time-to-first-*chunk* (~450 ms on an
  M1 Pro) and the ITL percentiles are inter-chunk gaps. Token counts,
  throughput and per-token energy are unaffected.
- **Context is 4096 tokens**, including instructions and the response. Queries
  that overflow it are recorded with empty content and counted under
  `client_info.skipped_queries` rather than aborting the run; `fixed_length`
  and short-prompt datasets suit AFM better than long-context ones.

---

## Agentic Profiling

Profile multi-turn agent workloads — multiple LLM calls, tool invocations, and reasoning steps per task.

```bash
ipw run --agent <agent> --model <model> --dataset <dataset> [options]
```

### Required Options

| Option | Description |
|--------|-------------|
| `--agent` | Agent harness ID (`react`, `openhands`, `terminus`) |
| `--model` | Model name for the agent's LLM backbone (or use `--preset`) |
| `--dataset` | Dataset ID for the workload |

### Optional Options

| Option | Default | Description |
|--------|---------|-------------|
| `--preset` | none | Model preset name (e.g., `glm-4.7-flash`); alternative to `--model` |
| `--client-base-url` | `http://localhost:8000` | Inference server base URL |
| `--api-key` | `EMPTY` | API key for the inference server |
| `--max-queries` | all | Limit number of tasks to run |
| `--output-dir` | `./runs/` | Directory for results |
| `--concurrency` | 1 | Number of tasks to run in parallel |
| `--query-timeout` | none | Wall-clock timeout in seconds per query |
| `--max-retries` | 3 | Per-turn retry budget on transient errors (0 disables retry) |
| `--require-dedicated-hardware` | off | Abort if preflight detects competing GPU/CPU workloads (precision mode) |
| `--export-format` | `jsonl,hf` | Comma-separated export formats (`jsonl`, `hf`) |
| `--estimate-flops` | off | Enable FLOPs estimation |
| `--dataset-kwargs` | none | JSON string of extra dataset arguments |
| `--agent-kwargs` | none | JSON string of extra agent arguments |
| `--eval-client` | `openai` | Client for evaluation judging |
| `--eval-model` | `gpt-5-nano-2025-08-07` | Model for evaluation |

### Agent Setup

=== "ReAct"

    Uses the [Agno](https://github.com/agno-agi/agno) framework for tool-augmented reasoning.

    ```bash
    uv pip install -e 'intelligence-per-watt[react]'

    ipw run \
      --agent react \
      --model gpt-4o \
      --dataset gaia \
      --max-queries 10
    ```

=== "OpenHands"

    Uses the [OpenHands SDK](https://github.com/All-Hands-AI/OpenHands) for autonomous task execution.

    ```bash
    uv pip install -e 'intelligence-per-watt[openhands]'

    ipw run \
      --agent openhands \
      --model gpt-4o \
      --dataset swebench \
      --max-queries 30
    ```

=== "Terminus"

    Runs tasks inside Docker containers for terminal/CLI benchmarking.

    ```bash
    uv pip install -e 'intelligence-per-watt[terminus]'

    ipw run \
      --agent terminus \
      --model gpt-4o \
      --dataset terminalbench \
      --max-queries 10
    ```

=== "TerminalBench Native"

    Any agent can use TerminalBench tasks via the `terminalbench-native` dataset. The runner creates a per-task Docker container automatically.

    ```bash
    ipw run \
      --agent openhands \
      --model gpt-4o \
      --dataset terminalbench-native \
      --concurrency 4 \
      --dataset-kwargs '{"n_tasks": 20}'
    ```

---

## Concurrent Execution

Use `--concurrency N` to run multiple agentic tasks in parallel. Each concurrent task gets its own agent instance to avoid shared state conflicts.

```bash
ipw run \
  --agent openhands \
  --model gpt-4o \
  --dataset terminalbench-native \
  --concurrency 4 \
  --max-queries 20
```

Concurrency is most useful for agentic workloads where each task takes minutes (e.g., TerminalBench, SWE-bench). For fast single-turn benchmarks, sequential execution is usually sufficient.

---

## Tool Configuration

Agents can use MCP (Model Context Protocol) tools for accessing inference servers and retrieval systems.

### Inference Server Tools

| Tool | Description |
|------|-------------|
| `openai_server` | OpenAI API |
| `anthropic_server` | Anthropic API |
| `gemini_server` | Google Gemini API |
| `ollama_server` | Local Ollama |
| `vllm_server` | Local vLLM |
| `openrouter_server` | OpenRouter API |

### Retrieval Tools

| Tool | Description |
|------|-------------|
| `bm25_server` | BM25 sparse retrieval |
| `dense_server` | Dense vector retrieval |
| `grep_server` | Grep-based text search |
| `hybrid_server` | Hybrid BM25 + dense retrieval |

All MCP tool servers are in `ipw/agents/mcp/` and implement the `BaseMCPServer` interface.

---

## Output

### Single-Turn Output

```
runs/profile_<hardware>_<model>_<dataset>/
    data-00000-of-00001.arrow   # Per-query metrics (Arrow dataset)
    summary.json                # Run metadata
    analysis/
        accuracy.json           # IPJ/IPW and scoring results
```

### Agentic Output

```
runs/run_<agent>_<model>_<dataset>/
    traces.jsonl               # One QueryTrace per line (per-turn details)
    data-*.arrow               # HuggingFace dataset format
    summary.json               # Run metadata
    analysis/
        accuracy.json          # Scoring results
```

### summary.json

Contains run configuration (client/agent, model, dataset), aggregate totals (queries, tokens, wall clock, energy, cost), per-query averages, per-metric statistics (avg, median, min, max, std), and a generation timestamp.

### Arrow Dataset Schema

Each row represents one query with fields: `problem` (input prompt), `answer` (reference answer), `model_answers` (generated responses), and `model_metrics` (energy, power, latency, memory, temperature, tokens, compute, cost).

### JSONL Traces (Agentic Only)

Each line is a `QueryTrace` containing per-turn `TurnTrace` records with: token counts, tools called, per-tool latencies, wall-clock time, GPU/CPU energy and power, API cost, and any errors.

---

## Benchmarking (`ipw bench`)

`ipw bench` is the primary command for running full benchmark evaluations with energy telemetry. It orchestrates the entire pipeline from server management through results export, and is the recommended way to produce Intelligence Per Joule (IPJ) and Intelligence Per Watt (IPW) scores.

### Pipeline

The benchmark pipeline executes these stages in order:

1. **Server startup** (optional, `--auto-server`) — automatically launches inference servers using vLLM, Ollama, or preset configurations. Startup time is excluded from profiling measurements.
2. **Warmup** (default on, skip with `--skip-warmup`) — sends warmup queries to initialize model weights and KV caches. Warmup time is excluded from profiling measurements.
3. **Benchmark execution** — runs the selected agent against the dataset with energy telemetry streaming from the Rust energy monitor via gRPC. Each query is evaluated inline (`is_resolved` is computed per query).
4. **Results export** — writes `summary.json` (aggregate metrics, efficiency scores, statistics), `traces.jsonl` (per-query details), and per-query artifacts to the output directory.
5. **Server shutdown** (if `--auto-server` was used) — stops managed servers. Shutdown time is excluded from profiling measurements.

```bash
ipw bench --agent <agent> --model <model> --dataset <dataset> [options]
```

### CLI Options

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--agent` | Yes | — | Agent type: `react`, `openhands`, `terminus`, `terminus-tb` |
| `--model` | One of `--model` or `--preset` | — | HuggingFace model ID (e.g., `Qwen/Qwen3-4B`) |
| `--preset` | One of `--model` or `--preset` | — | Model preset name (e.g., `glm-4.7-flash`); see [Model Presets](#model-presets) |
| `--dataset` | Yes | — | Dataset to benchmark (`gaia`, `hle`, `simpleqa`, `terminalbench`, etc.) |
| `--limit` | No | all | Maximum number of queries to evaluate |
| `--output` | No | `./outputs/bench/` | Output directory for results |
| `--client` | No | `vllm` | Model provider: `vllm`, `openai`, `ollama` |
| `--vllm-url` | No | `http://localhost:8000/v1` | Override the inference server URL |
| `--api-key` | No | `EMPTY` | API key for the inference server |
| `--per-action` | No | off | Enable per-action energy breakdown (tool calls vs. LM inference) |
| `--no-telemetry` | No | off | Disable energy telemetry collection entirely |
| `--skip-warmup` | No | off | Skip warmup phase (cold-start costs will be included in measurements) |
| `--auto-server` | No | off | Auto-manage inference server lifecycle (start before, stop after) |
| `--submodel` | No | — | Submodel specification in `alias:backend:model_id` format (repeatable) |
| `--base-port` | No | `8000` | Base port for auto-managed vLLM servers |
| `--seed` | No | — | Random seed for reproducible benchmark sampling |

### Example Commands

```bash
# Simple benchmark against a running server
ipw bench --agent react --model Qwen/Qwen3-4B --dataset gaia --limit 5

# Full auto-managed pipeline with a preset
ipw bench --agent openhands --preset qwen35-397b-a17b-fp8 --dataset gaia --auto-server

# With per-action energy breakdown
ipw bench --agent react --model Qwen/Qwen3-4B --dataset gaia --per-action

# Without energy telemetry (accuracy-only run)
ipw bench --agent react --model Qwen/Qwen3-4B --dataset gaia --no-telemetry

# Terminus agent on TerminalBench
ipw bench --agent terminus --model openai/gpt-oss-120b --dataset terminalbench --limit 10
```

### Output Format

Results are written to a timestamped directory under `--output` (default `./outputs/bench/`):

```
outputs/bench/gaia_Qwen_Qwen3-4B_20260308_143022/
    summary.json          # Aggregate metrics and efficiency scores
    traces.jsonl          # Per-query trace details
    results.json          # Full benchmark result (energy, hardware, metadata)
    artifacts/            # Per-query artifacts (agent logs, tool outputs)
```

**`summary.json`** contains:

- **`config`** — run configuration (agent, model, dataset, client, telemetry settings)
- **`totals`** — aggregate counts: queries, completed, resolved, unresolved, turns, tool calls, tokens, wall clock, energy, cost, accuracy
- **`efficiency`** — IPJ (accuracy / total GPU energy in joules), IPW (accuracy / average GPU power in watts), total energy, accuracy
- **`averages`** — per-query means for turns, wall clock, and GPU energy
- **`statistics`** — per-metric distributions (avg, median, min, max, std) for wall clock, energy, power, tokens, throughput, cost, turns, tool calls, and memory bandwidth utilization
- **`normalized_statistics`** — same statistics recomputed after trimming the top and bottom 5% of queries by wall clock time (outlier removal)
- **`normalized_efficiency`** — IPJ/IPW recomputed on the trimmed query set

**`traces.jsonl`** contains one JSON object per line, each a serialized `QueryTrace` with per-turn `TurnTrace` records including token counts, tool calls, wall-clock time, GPU/CPU energy and power, MBU, cost, and resolution status.

### Model Presets

The `--preset` flag is a shorthand for common model configurations. A preset maps a short name to a HuggingFace model ID plus vLLM launch arguments (tensor parallel size, tool/reasoning parsers, memory limits, etc.). Presets are especially useful with `--auto-server`, which uses the preset's vLLM arguments to launch the server automatically.

Use `--preset` or `--model`, but not both.

Available presets include:

| Preset | Model ID | TP Size |
|--------|----------|---------|
| `glm-4.7-flash` | `zai-org/GLM-4.7-FP8` | 8 |
| `gpt-oss-120b` | `openai/gpt-oss-120b` | 8 |
| `gpt-oss-20b` | `openai/gpt-oss-20b` | 1 |
| `qwen3-30b-a3b` | `Qwen/Qwen3-30B-A3B` | 1 |
| `qwen35-397b-a17b-fp8` | `Qwen/Qwen3.5-397B-A17B-FP8` | 8 |
| `minimax-m2.5` | `MiniMaxAI/MiniMax-M2.5` | 4 |
| `kimi-k2.5` | `moonshotai/Kimi-K2.5` | 8 |

The full list is defined in `src/ipw/cli/model_presets.py`. To see all available presets, run:

```bash
ipw list all
```

---

## Server Management (`ipw servers`)

`ipw servers` provides commands for managing local inference servers (Ollama, vLLM). This is useful when you want to start servers manually before benchmarking, rather than using `ipw bench --auto-server`.

### Commands

#### `ipw servers start` — Start a server in the background

```bash
# Start Ollama
ipw servers start --ollama

# Start vLLM with a specific model
ipw servers start --vllm --model Qwen/Qwen3-4B

# Start vLLM with tensor parallelism
ipw servers start --vllm --model Qwen/Qwen3-4B --tensor-parallel-size 4
```

| Option | Default | Description |
|--------|---------|-------------|
| `--ollama` | — | Start Ollama server |
| `--vllm` | — | Start vLLM server |
| `--model` | — | Model to load (required for vLLM) |
| `--port` | 11434 (Ollama), 8000 (vLLM) | Port to run the server on |
| `--gpu-memory-utilization` | 0.9 | GPU memory utilization for vLLM |
| `--tensor-parallel-size` | 1 | Number of GPUs for tensor parallelism |

#### `ipw servers launch` — Start and block until ready

Like `start`, but waits for the server to respond to health checks and runs a warmup query. Recommended before benchmarking to ensure server startup is excluded from measurements.

```bash
# Launch vLLM and wait up to 2 minutes for it to be ready
ipw servers launch --vllm --model Qwen/Qwen3-4B --wait-timeout 120

# Launch using a model preset
ipw servers launch --vllm --preset glm-4.7-flash

# Launch Ollama and pre-pull a model
ipw servers launch --ollama --model llama3.2:1b
```

| Option | Default | Description |
|--------|---------|-------------|
| `--ollama` / `--vllm` | — | Server type to launch |
| `--model` | — | Model to load |
| `--preset` | — | Model preset (resolves model ID and vLLM args) |
| `--port` | auto | Server port |
| `--gpu-memory-utilization` | 0.9 | GPU memory utilization for vLLM |
| `--tensor-parallel-size` | auto from preset or 1 | Tensor parallelism |
| `--wait-timeout` | 60 | Seconds to wait for the server to become ready |

#### `ipw servers stop` — Stop running servers

```bash
# Stop all inference servers
ipw servers stop --all

# Stop only vLLM
ipw servers stop --vllm

# Stop only Ollama
ipw servers stop --ollama
```

#### `ipw servers status` — Check server status

```bash
ipw servers status
```

Displays whether Ollama and vLLM are running, the loaded model (if detectable), and any registered server lock files with port, model, PID, and owner information.

## Preflight hardware check

Before the first query, `AgenticRunner` samples GPU and CPU utilization to
detect whether the hardware is shared with other workloads. Whole-device
energy measurement inflates per-query attribution proportionally to any
contaminating workload — a baseline check catches the most common case before
results are spent.

By default the check is **informational**: if baseline utilization exceeds
thresholds (GPU > 5%, CPU > 10%, or foreign GPU processes detected via NVML),
every subsequent `TurnTrace` and `QueryTrace` is tagged `shared_device_warning:
true` and warnings are logged. Runs proceed normally — the flag is
informational, surfacing in the JSON output so downstream analysis can filter
contaminated runs.

Strict mode (`--require-dedicated-hardware`) aborts the run if contamination
is detected, ensuring publishable energy numbers are never silently inflated.

The check skips gracefully when `pynvml` or `psutil` is unavailable — only the
GPU half (or CPU half) is gated by the missing dependency.

## EventBus

Internal telemetry now flows through an in-process `EventBus` pub/sub layer.
The legacy `EventRecorder.record()` API is preserved verbatim — every call
appends an `AgentEvent` to the recorder's list (existing `get_events()`
contract intact) *and* publishes an `Event` to the bus so additional
subscribers can observe without changing the recorder.

A single shadow subscriber attaches by default: `EnergyAttribution`,
which pairs `TOOL_CALL_START`/`TOOL_CALL_END` and `LM_INFERENCE_START`/
`LM_INFERENCE_END` events by correlation_id, computes the energy window via
the `TelemetrySession`, and emits `ENERGY_ATTRIBUTED` events. Additional
subscribers can plug in without further plumbing changes. Cloud LM inferences
explicitly emit
`gpu_energy_j=None` (no local GPU involved); only `cpu_energy_j` is populated
for the local-CPU request marshaling.

## Trace schema additions

The trace types carry these fields on `TurnTrace` (all default to `None` or `False`):

- `dram_energy_joules: float | None`
- `peak_watts: float | None`
- `reasoning_tokens: int | None`
- `cached_tokens: int | None`
- `has_parallel_tools: bool`  (set when an executor dispatches multiple tools concurrently)
- `shared_device_warning: bool`  (set when the preflight or per-window check detects contamination)

And to `QueryTrace`:

- `lm_energy_measurable: bool`  (True by default; runner sets `False` for cloud LMs where GPU telemetry is unavailable)
- `shared_device_warning: bool`
- `accuracy_score: float | None`  (evaluators may return float, bool, or dict; annotated `float` for the common case)
- `accuracy_metadata: dict[str, Any]`
- `n_retries: int`

These fields are **additive**: existing analysis tooling and JSONL exports
continue to work without modification, and the new fields default to their
safe values when no subscriber populates them.
