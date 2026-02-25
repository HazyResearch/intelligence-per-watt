# Agentic Profiling

The `ipw run` command profiles multi-turn agent workloads. Unlike single-turn profiling (`ipw profile`), agentic profiling tracks the full lifecycle of an agent solving a task: multiple LLM calls, tool invocations, and reasoning steps.

## Command Reference

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
| `--dataset-kwargs` | none | JSON string of extra dataset keyword arguments |
| `--agent-kwargs` | none | JSON string of extra agent keyword arguments |
| `--eval-client` | `openai` | Client for evaluation judging |
| `--eval-model` | `gpt-5-nano-2025-08-07` | Model for evaluation |

## Lifecycle

An agentic profiling run follows this sequence:

1. **Start energy monitor** -- Launch the Rust gRPC telemetry service.
2. **Initialize agent** -- Create the agent instance with the specified model and any MCP tools. Attach an `EventRecorder` for per-action telemetry.
3. **Iterate dataset** -- For each dataset record:
    - Create a `QueryTrace` to track per-turn telemetry
    - Call `agent.run(input)` which internally:
        - Makes LLM inference calls (recorded as `lm_inference_start`/`end` events)
        - Calls tools (recorded as `tool_call_start`/`end` events)
        - Accumulates token counts, latencies, and tool metadata per turn
    - Capture the telemetry window for energy attribution
    - Build a `TurnTrace` for each agent turn with tokens, tools, latency, energy, and cost
    - Save the `QueryTrace` as a JSONL line
4. **Export results** -- Write traces as JSONL and HuggingFace Arrow datasets.
5. **Run analysis** -- Evaluate responses with the LLM judge and compute accuracy/efficiency metrics.

## Concurrent Execution

Use `--concurrency N` to run multiple tasks in parallel via a thread pool. Each concurrent task gets its own agent instance (created from `--agent-kwargs`) to avoid shared state conflicts. Results are collected in original index order.

```bash
ipw run \
  --agent openhands \
  --model gpt-4o \
  --dataset terminalbench-native \
  --concurrency 4 \
  --max-queries 20
```

Concurrency is most useful for agentic workloads where each task takes minutes (e.g., TerminalBench, SWE-bench). For fast single-turn benchmarks, sequential execution is usually sufficient.

## Agent Setup

### ReAct Agent

The ReAct agent uses the [Agno](https://github.com/agno-agi/agno) framework for tool-augmented reasoning:

```bash
uv pip install -e 'intelligence-per-watt[react]'

ipw run \
  --agent react \
  --model gpt-4o \
  --dataset gaia \
  --max-queries 10
```

The ReAct agent wraps tools with telemetry instrumentation. Each tool call is recorded as a `tool_call_start`/`tool_call_end` event pair, enabling per-tool energy attribution.

### OpenHands Agent

The OpenHands agent uses the [OpenHands SDK](https://github.com/All-Hands-AI/OpenHands) for autonomous task execution:

```bash
uv pip install -e 'intelligence-per-watt[openhands]'

ipw run \
  --agent openhands \
  --model gpt-4o \
  --dataset swebench \
  --max-turns 30
```

OpenHands integrates via an instrumented callback system. The `_instrumented_callback` method emits telemetry events for every action and observation in the agent loop. If the agent exhausts its turn budget without calling `FinishTool`, IPW sends a synthesis nudge to extract a final answer.

### Terminus Agent

The Terminus agent runs tasks inside Docker containers for terminal/CLI benchmarking:

```bash
uv pip install -e 'intelligence-per-watt[terminus]'

ipw run \
  --agent terminus \
  --model gpt-4o \
  --dataset terminalbench \
  --max-queries 10
```

The Terminus agent:

1. Creates or reuses a Docker container
2. Installs tmux inside the container
3. Runs the agent's `perform_task()` in a tmux session
4. Captures the terminal output as the response

### Terminus-TB Agent

The Terminus-TB agent is a lightweight wrapper around Terminus2 designed for the `terminalbench-native` dataset. Unlike the original Terminus agent, it does not manage Docker containers — that lifecycle is handled by the runner via `TerminalBenchTaskEnv`:

```bash
ipw run \
  --agent terminus-tb \
  --model gpt-4o \
  --dataset terminalbench-native \
  --concurrency 4 \
  --max-queries 10
```

### TerminalBench Native Mode

Any agent can work with TerminalBench tasks via the `terminalbench-native` dataset. The runner creates a per-task Docker container, sets up a tmux session, and runs test scripts after the agent finishes:

```bash
# OpenHands with TerminalBench
ipw run \
  --agent openhands \
  --preset glm-4.7-flash \
  --dataset terminalbench-native \
  --concurrency 4 \
  --dataset-kwargs '{"n_tasks": 20}'
```

The `--dataset-kwargs` option passes arguments to the dataset constructor. Supported keys for `terminalbench-native`:

- `name` — dataset name (default: `terminal-bench-core`)
- `version` — dataset version (default: `0.1.1`)
- `path` — local path to dataset directory (overrides name/version)
- `task_ids` — list of specific task IDs to include
- `n_tasks` — limit total number of tasks

## Tool Configuration

Agents can use MCP (Model Context Protocol) tools for accessing inference servers and retrieval systems. IPW ships with several built-in MCP servers:

### Inference Server Tools

These connect to LLM inference servers for sub-queries:

- `openai_server` -- OpenAI API
- `anthropic_server` -- Anthropic API
- `gemini_server` -- Google Gemini API
- `ollama_server` -- Local Ollama
- `vllm_server` -- Local vLLM
- `openrouter_server` -- OpenRouter API

### Retrieval Tools

These provide document retrieval capabilities:

- `bm25_server` -- BM25 sparse retrieval
- `dense_server` -- Dense vector retrieval
- `grep_server` -- Grep-based text search
- `hybrid_server` -- Hybrid BM25 + dense retrieval

All MCP tool servers are in `ipw/agents/mcp/` and implement the `BaseMCPServer` interface.

## Per-Turn Traces

Each agent turn produces a `TurnTrace` (defined in `ipw/execution/trace.py`) containing:

| Field | Type | Description |
|-------|------|-------------|
| `turn_index` | int | Sequential turn number |
| `input_tokens` | int | Tokens consumed in this turn |
| `output_tokens` | int | Tokens generated in this turn |
| `tool_result_tokens` | int | Tokens from tool results |
| `tools_called` | list[str] | Names of tools invoked |
| `tool_latencies_s` | dict[str, float] | Per-tool wall-clock time |
| `wall_clock_s` | float | Total wall-clock time for the turn |
| `gpu_energy_joules` | float | GPU energy consumed |
| `cpu_energy_joules` | float | CPU energy consumed |
| `gpu_power_avg_watts` | float | Average GPU power |
| `cpu_power_avg_watts` | float | Average CPU power |
| `cost_usd` | float | API cost for this turn |
| `error` | str | Error message if the turn failed |

Turns are aggregated into a `QueryTrace` for the full task, with computed properties for `total_input_tokens`, `total_output_tokens`, `total_tool_calls`, `total_gpu_energy_joules`, and `total_cost_usd`.

## Output Structure

Agentic runs produce both JSONL traces and Arrow datasets:

```
runs/run_<agent>_<model>_<dataset>/
    traces.jsonl               # One QueryTrace per line
    data-*.arrow               # HuggingFace dataset format
    summary.json               # Run metadata
    analysis/
        accuracy.json          # Scoring results
```

See [Export Formats](export-formats.md) for schema details.
