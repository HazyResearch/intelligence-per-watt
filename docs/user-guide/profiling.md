# Profiling

IPW supports two profiling modes: **single-turn** (`ipw profile`) for benchmarking raw inference, and **agentic** (`ipw run`) for multi-turn agent workloads with tool use.

---

## Single-Turn Profiling

Send prompts to an inference server one at a time, capturing energy telemetry for each query.

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
| `--model` | Model name for the agent's LLM backbone |
| `--dataset` | Dataset ID for the workload |

### Optional Options

| Option | Default | Description |
|--------|---------|-------------|
| `--max-queries` | all | Limit number of tasks to run |
| `--output-dir` | `./runs/` | Directory for results |
| `--max-turns` | 20 | Maximum agent turns per task |
| `--concurrency` | 1 | Number of tasks to run in parallel |
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
      --max-turns 30
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

Contains run metadata: profiler configuration (client/agent, model, dataset), hardware information (GPU, CPU, platform), timing (start/end timestamps, total duration), and aggregate token counts.

### Arrow Dataset Schema

Each row represents one query with fields: `problem` (input prompt), `answer` (reference answer), `model_answers` (generated responses), and `model_metrics` (energy, power, latency, memory, temperature, tokens, compute, cost).

### JSONL Traces (Agentic Only)

Each line is a `QueryTrace` containing per-turn `TurnTrace` records with: token counts, tools called, per-tool latencies, wall-clock time, GPU/CPU energy and power, API cost, and any errors.
