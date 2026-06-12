# Agents

## Overview

IPW profiles multi-turn agent workloads through pluggable agent harnesses. Each agent wraps an existing framework and adds energy telemetry instrumentation. All agents inherit from `BaseAgent` and are registered with `AgentRegistry`.

### Available Agents

| Agent ID | Framework | Install Extra | Use Case |
|----------|-----------|---------------|----------|
| `react` | [Agno](https://github.com/agno-agi/agno) | `ipw[react]` | General tool-augmented reasoning |
| `openhands` | [OpenHands SDK](https://github.com/All-Hands-AI/OpenHands) | `ipw[openhands]` | Autonomous task execution, coding |
| `terminus` | [terminal-bench](https://github.com/terminal-bench/terminal-bench) | `ipw[terminus]` | Terminal/CLI task benchmarking |
| `terminus-tb` | [terminal-bench](https://github.com/terminal-bench/terminal-bench) | `ipw[terminus]` | TerminalBench native (Docker managed by runner) |

### Event Types

The `EventRecorder` captures timestamped events during agent execution, correlated with energy telemetry to compute per-action energy costs.

| Event Type | Description |
|------------|-------------|
| `lm_inference_start` / `lm_inference_end` | LLM call boundaries |
| `tool_call_start` / `tool_call_end` | Tool invocation boundaries |
| `prefill_start` / `prefill_end` | Prefill phase (if detectable) |
| `decode_start` / `decode_end` | Decode phase (if detectable) |
| `submodel_call_start` / `submodel_call_end` | Sub-model calls from MCP tools |

---

## ReAct (Agno)

```bash
uv pip install -e 'intelligence-per-watt[react]'
```

The ReAct agent uses [Agno](https://github.com/agno-agi/agno) to implement Reasoning + Acting (ReAct) style tool-augmented reasoning. It wraps Agno's `Agent` class and instruments tool calls for energy tracking.

```bash
ipw run \
  --agent react \
  --model gpt-4o \
  --dataset gaia \
  --max-queries 10
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | Any | required | Agno Model instance (e.g., `OpenAIChat`) |
| `tools` | list[Callable] | required | List of callable tool functions |
| `instructions` | str | built-in | Custom system instructions |
| `max_turns` | int | -- | Maximum reasoning iterations |

---

## OpenHands

```bash
uv pip install -e 'intelligence-per-watt[openhands]'
```

The OpenHands agent uses the [OpenHands SDK](https://github.com/All-Hands-AI/OpenHands) for autonomous task execution with per-tool energy tracking. It is designed for complex, multi-step tasks such as software engineering, research, and document analysis.

```bash
ipw run \
  --agent openhands \
  --model gpt-4o \
  --dataset swebench \
  --max-turns 30
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | Any | required | LLM model instance |
| `tools` | list | None | OpenHands Tool specs |
| `mcp_tools` | dict | None | MCP server instances for sub-queries |
| `max_turns` | int | 20 | Maximum iterations per run |

---

## Terminus

```bash
uv pip install -e 'intelligence-per-watt[terminus]'
```

The Terminus agent uses [terminal-bench](https://github.com/terminal-bench/terminal-bench) to run tasks inside Docker containers with tmux, enabling benchmarking of terminal/CLI task execution.

**Prerequisites:** Docker Engine installed and running; current user in the `docker` group (or sudo access).

```bash
ipw run \
  --agent terminus \
  --model gpt-4o \
  --dataset terminalbench \
  --max-queries 10
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | required | Model name (e.g., `"gpt-4o"`) |
| `docker_image` | str | `"ubuntu:22.04"` | Docker image for the container |
| `container_name` | str | `"terminus-container"` | Name for the Docker container |
| `max_turns` | int | -- | Maximum agent turns |

---

## MCP Tools

Model Context Protocol servers provide tool capabilities to agents. Each MCP server wraps an external service with a standard interface.

**Inference servers** -- wrap LLM APIs so agents can make sub-queries:

| Server | Backend |
|--------|---------|
| `OpenAIServer` | OpenAI API |
| `AnthropicServer` | Anthropic API |
| `GeminiServer` | Google Gemini API |
| `OllamaServer` | Local Ollama instance |
| `VLLMServer` | Local vLLM instance |
| `OpenRouterServer` | OpenRouter API |

**Retrieval servers** -- provide document retrieval for RAG-style agents:

| Server | Method |
|--------|--------|
| `BM25Server` | Sparse BM25 retrieval |
| `DenseServer` | Dense vector retrieval |
| `GrepServer` | Grep-based text search |
| `HybridServer` | Combined BM25 + dense retrieval |

---

## Writing a Custom Agent

See [Extending IPW](../extending/index.md) for how to implement and register a custom agent.

## Native ReAct (`react-native`)

`react-native` is the new ReAct implementation backed by the Executor. It
replaces the Agno-based `react` for new benchmarks. Key differences:

| | react (Agno) | react-native |
|---|---|---|
| Loop owner | Agno Agent | IPW Executor |
| Parser | Agno tool spec | structured-text (THOUGHT/ACTION/INPUT) |
| Retry | Agno internal | Executor: up to 3 attempts, exponential backoff |
| Parallel tools | Agno's dispatch | asyncio.gather with side_effect_conflict fallback |
| Telemetry | Per-tool _record_event | EventBus + EnergyAttribution subscriber |

### Usage

```bash
ipw run --agent react-native --model openai/gpt-4o-mini --dataset swebench --max-queries 5
```

### System tools

- `shell_exec` — bash via subprocess (side-effect-conflict, no network)
- `git_tool` — allowlisted git subcommands (status / diff / log / show / branch / add / commit / checkout / rev-parse / ls-files / stash)
- `apply_patch` — apply unified diff via GNU patch
- `http_request` — HTTP via httpx (network required)
- `repl` — persistent Python REPL subprocess (state persists across calls within a single agent instance)
- `code_interpreter_docker` — sandboxed one-shot Python execution in a Docker container (`--network=none --memory=512m --cpus=1`)

For `ipw run --agent react-native`, all registered tools are made available to
the agent (SWE-Bench tasks, for example, lean on `shell_exec`, `git_tool`,
`apply_patch`, and `repl`).

### Retry policy

The Executor distinguishes two error classes via `classify_error`:

- **RetryableError** (`TimeoutError`, `ConnectionError`, network glitches) —
  retried up to 3 times with exponential backoff (1s / 2s / 4s). After
  exhaustion, the error is reraised as `RetryExhaustedError` (distinct from
  a fatal-from-start `FatalError`).
- **FatalError** (malformed agent output, missing tool, assertion errors,
  default for unknown exceptions) — no retry; the run aborts immediately.

Each retry publishes a `RETRY_ATTEMPT` event on the bus so telemetry can
distinguish "transient retry survived" from "fatal-from-attempt-1" runs.

### Parallel tool dispatch

When a turn emits multiple tool calls, the Executor runs them concurrently
via `asyncio.gather`. Tools tagged `side_effect_conflict=True` (shell_exec,
git_tool, apply_patch, repl) force sequential dispatch within a turn — two
shell commands stepping on each other's workspace would be a real bug,
not a feature.

### Migration from `react`

Agno's `react` is preserved for existing benchmark configurations and is
unaffected by this change. New benchmark work should target `react-native`
to get retry, parallel dispatch, and bus-driven telemetry. GAIA, TerminalBench,
and FRAMES also run on the new infra.

## Browser and PDF tools

Three tools support web/document tasks. They are optional dependencies
and ship under the `[browser]` and `[pdf]` extras:

```bash
uv pip install -e "intelligence-per-watt[browser,pdf]"
python -m playwright install chromium  # one-time, ~200MB
```

### Available tools

- `browser` — Headless Chromium fetch via Playwright. Returns rendered page
  text (`page.inner_text("body")`). Parameters: `url`, `wait_for` (CSS
  selector), `timeout` (seconds, default 30). Network required.
- `browser_axtree` — Same Playwright stack, but returns the accessibility
  tree as indented text (role + name + value per node). Better than raw
  HTML for agent navigation tasks. Same parameters as `browser`.
- `pdf_tool` — Extract text and metadata from a PDF. Parameters: `path`
  (local file or http(s) URL), `page_range` (e.g. `"1-5"` or `"1,3,5"`),
  `extract_metadata` (bool, default False). Uses pdfplumber + pypdf.

### GAIA + react-native

When invoked via `react-native`, GAIA web/document tasks use the browser, PDF,
shell, and repl tools:

```
gaia: browser, browser_axtree, pdf_tool, shell_exec, repl
```

FRAMES uses `browser`, `browser_axtree`, `shell_exec`, `repl` (no PDFs).

### Workspace isolation

All these tool calls execute inside a per-query temp dir managed by
`AgenticRunner._run_with_executor`. The browser tool's downloads, the PDF
tool's URL-fetched temp files, and any shell-side effects stay inside that temp
dir, which is cleaned up automatically after each query.

## Image and audio tools

Two multimodal tools support GAIA's image/audio tasks. Install via the
`[multimodal]` extra:

```bash
uv pip install -e "intelligence-per-watt[multimodal]"
```

### Available tools

- `image_tool` — Analyze a local image file (or URL) and answer a question
  about it using a vision model (gpt-4o-mini). Parameters: `path` (image file
  or http(s) URL), `question` (default "Describe this image in detail.").
  Supports png, jpg/jpeg, gif, webp. Images larger than ~2000px are downscaled
  before encoding. Network required.
- `audio_tool` — Transcribe a local audio file to text via OpenAI Whisper.
  Parameters: `path` (audio file), `language` (optional ISO code). Supports
  mp3, wav, m4a, ogg, flac, webm; max 25MB. Network required.

### Divergence from OpenJarvis

OpenJarvis's `image_tool` is an image *generation* tool (DALL-E). GAIA needs
image *understanding* — analyzing images that appear in tasks — so IPW's
`image_tool` was built fresh as a vision-understanding tool rather than ported.
The audio tool IS a direct port of OpenJarvis's Whisper transcription tool.

### GAIA full tool set

When invoked via `react-native`, GAIA uses:

```
gaia: browser, browser_axtree, pdf_tool, image_tool, audio_tool, shell_exec, repl
```

This covers GAIA's text, web, document, image, and audio task types. All tool
calls run inside the per-query temp workspace and are bounded by the Executor's
per-step (300s) and per-tool (120s) timeouts.

## docker_shell_exec, FRAMES, and the wrapper-agent bridge

This adds the 12th Tier-1 tool, wires FRAMES through `react-native`, and bridges
the SDK-wrapped agents (OpenHands, Terminus) onto the EventBus.

### docker_shell_exec

- `docker_shell_exec` — run a **shell** command in a fresh `docker run --rm`
  container, distinct from `code_interpreter_docker` (which runs Python).
  Parameters: `command`, `image` (default `ubuntu:22.04`), `timeout` (seconds).
  Hardened by default: `--network=none --memory=512m --cpus=1`. When the
  Executor sets a per-query workspace (`_default_cwd`), the directory is mounted
  at `/workspace` (`-v {cwd}:/workspace -w /workspace`) so file side effects stay
  contained. Requires Docker; unit tests skip without it.

The full 12-tool set: `shell_exec`, `git_tool`, `apply_patch`, `http_request`,
`repl`, `code_interpreter_docker`, `browser`, `browser_axtree`, `pdf_tool`,
`image_tool`, `audio_tool`, `docker_shell_exec`.

### FRAMES on react-native

FRAMES (web-research QA) runs through the `react-native` + Executor stack with
the `browser`, `browser_axtree`, `shell_exec`, and `repl` tools.

### Wrapper-agent telemetry bridge (best-effort)

`WrapperAgentBridge` (`ipw.telemetry.wrapper_agent_bridge`) republishes an
SDK-wrapped agent's event stream onto the IPW EventBus — `AGENT_START/END`,
`TURN_START/END`, `TOOL_CALL_START/END`, `LM_INFERENCE_START/END` — assigning
paired `correlation_id`s so `EnergyAttribution` pairs tool/LM energy windows
just as it does for native agents.

- **OpenHands mode** (`open_turn_on_start=False`): each action/observation pair
  is one bus turn; retry is task-level (whole-task retry on error status).
- **Terminus mode** (`open_turn_on_start=True`): the whole container run is a
  single window (`start()` → `TURN_START`, `finish()` → `TURN_END`).

The bridge never imports `openhands-sdk` or `terminal-bench`; `default_classify`
duck-types both stub dicts and OpenHands-shaped objects (class names containing
`Action`/`Observation`). Fidelity is intentionally lower than native agents
(spec §4.7) — the right tradeoff for SDK-wrapped harnesses. Live-SDK wiring of
the bridge into `OpenHands.run` / `Terminus.run` is a documented follow-up; the
legacy non-bus agent path is unaffected.

### Final benchmark coverage

All four target benchmarks now run with unified energy telemetry:

| Benchmark | Agent | Tools |
|---|---|---|
| SWE-Bench | `react-native` | shell, git, patch, repl |
| GAIA | `react-native` | browser, axtree, pdf, image, audio, shell, repl |
| FRAMES | `react-native` | browser, axtree, shell, repl |
| TerminalBench | `terminus-tb` | (terminal env) |

`react-native` benchmarks get telemetry via the native EventBus path;
TerminalBench via `terminus-tb` plus the `WrapperAgentBridge` where applicable.

## Robustness and run-time controls

### Agent termination robustness

The `react-native` loop was strengthened to stop a real failure mode: on
multi-hop tasks the agent could call a tool every turn until `max_turns` without
ever emitting `FINAL_ANSWER`, yielding an empty answer. Three changes fixed it:

1. **Chain continuity** — `_build_messages` now replays the agent's own prior
   `THOUGHT/ACTION` outputs (not just tool observations), so the model can build
   on its reasoning across turns instead of re-deriving and looping.
2. **Turn-budget awareness** — the `Executor` exposes `turn_index`/`max_turns`
   on `ExecutorContext`, and the agent injects a per-turn budget hint
   ("you are on turn N of M") plus a strong final-turn directive.
3. **Forced final** — on the last turn the agent always returns a non-empty
   `FINAL_ANSWER` (best-effort from its current reasoning) rather than spending
   the turn on another tool call.

Effect on a FRAMES sample (gpt-4o-mini): completion rate rose from ~1-in-4 to
all instances answering, with clean concise answers.

### Retry + dedicated-hardware controls

Two `ipw run` flags (see [profiling](profiling.md)) surface existing infra:

- `--max-retries N` — per-turn retry budget on transient errors (`0` disables
  retry; default 3). The CLI count is *retries*; the runner/Executor uses the
  *attempt count* `N + 1`.
- `--require-dedicated-hardware` — turns the startup preflight (which samples
  baseline GPU/CPU utilization) into strict mode: the run aborts if competing
  workloads are detected, instead of only flagging `shared_device_warning`.
  Use this for publishable energy numbers — whole-device measurement inflates
  per-query attribution in proportion to any contaminating workload.

### Stress + per-tool energy validation

- **Concurrency stress** (`stress` marker) — runs 20 queries through
  `AgenticRunner` at `concurrency=20` with isolated per-query agents, asserting
  index-ordered results and zero cross-talk (validates the isolation that
  multi-query energy profiling depends on).
- **Per-tool energy attribution** (`test_per_tool_energy_integration.py`) —
  confirms each local-compute tool (`shell_exec`, `repl`,
  `code_interpreter_docker`) emits an `ENERGY_ATTRIBUTED` event with non-zero
  joules end-to-end; a hardware-integration test checks a real CPU-spin window
  against the live energy monitor.

### CI

`.github/workflows/core-infra-regression.yml` runs the offline unit-level
regression subset on every PR touching `execution/`, `telemetry/`, or `tools/`.
Tests that need a running server or GPU are marked `integration` and excluded
from CI; the concurrency stress test (`stress`) is opt-in.
