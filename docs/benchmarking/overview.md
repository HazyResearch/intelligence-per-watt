# Benchmarking Overview

IPW benchmarks LLM inference efficiency by measuring accuracy alongside energy consumption. A Rust gRPC service streams hardware telemetry at 50ms intervals while the Python profiler runs queries, enabling per-query energy attribution.

## Architecture

```
Python (ProfilerRunner)                    Rust (energy-monitor)
    |                                           |
    |--- Launch subprocess --->                 |
    |                                      Start gRPC server
    |                                      Auto-detect collector
    |                                           |
    |<-- StreamTelemetry (gRPC stream) -------->|
    |    TelemetryReading every 50ms            |
    |                                           |
    |--- Health (gRPC unary) ----------------->|
    |<-- HealthResponse -----------------------|
```

**Python launcher** (`ipw/telemetry/launcher.py`) manages the subprocess lifecycle -- starting, health-checking, and stopping the energy monitor.

**Python collector** (`ipw/telemetry/collector.py`) connects to the gRPC streaming endpoint and converts protobuf messages into Python `TelemetryReading` objects.

## Energy Monitor

The energy monitor is a standalone Rust binary that auto-detects the best hardware collector at startup:

1. **macOS** -- `powermetrics` for CPU/GPU/ANE power (requires `sudo`)
2. **Linux/Windows + NVIDIA GPU** -- NVML; falls through if initialization fails
3. **Linux + AMD GPU** -- ROCm SMI; falls through if unavailable
4. **Linux + RAPL** -- Intel RAPL for CPU energy counters
5. **Null collector** -- fallback reporting only CPU memory usage (all power/energy = -1)

The selected platform is reported in each `TelemetryReading.platform` field (e.g., `"nvidia"`, `"macos"`, `"amd"`, `"rapl"`, `"null"`).

### Proto Definition

The service is defined in `energy-monitor/proto/energy.proto`:

```protobuf
service EnergyMonitor {
  rpc Health(HealthRequest) returns (HealthResponse);
  rpc StreamTelemetry(StreamRequest) returns (stream TelemetryReading);
}
```

`StreamTelemetry` returns a continuous stream of `TelemetryReading` messages at a fixed 50ms interval. Fields that the current platform cannot provide are set to `-1`. The interval is not configurable -- it is hardcoded to ensure consistent measurement across deployments.

## Building the Energy Monitor

=== "Automated"

    Works for NVIDIA and Apple Silicon:
    ```bash
    uv run scripts/build_energy_monitor.py
    ```

=== "NVIDIA"

    ```bash
    cd energy-monitor && cargo build --release
    ```
    Requires NVIDIA drivers and `protoc`. See [Installation](../getting-started/installation.md).

=== "AMD"

    ```bash
    export LIBRARY_PATH=/opt/rocm/lib:$LIBRARY_PATH
    export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
    cd energy-monitor && cargo build --release --features amd
    cp ./target/release/energy-monitor ../intelligence-per-watt/src/ipw/telemetry/bin/linux-x86_64/energy-monitor
    ```
    Requires ROCm (`sudo apt install rocm rocm-smi`) and `protoc`.

    !!! note
        The automated build script does not pass `--features amd`, so AMD requires a manual build.

=== "Apple Silicon"

    ```bash
    brew install protobuf   # install protoc
    uv run scripts/build_energy_monitor.py
    ```
    Configure passwordless sudo for `powermetrics`:
    ```bash
    sudo sh -c "echo \"$(whoami) ALL=(ALL) NOPASSWD: /usr/bin/powermetrics\" > /etc/sudoers.d/powermetrics"
    ```

### Verify Your Build

```bash
uv run scripts/test_energy_monitor.py [--interval 2.0]
```

This starts the energy monitor, connects via gRPC, and prints telemetry readings to verify the binary works on your platform.

## Telemetry Session

The `TelemetrySession` (`ipw/execution/telemetry_session.py`) wraps the gRPC client in a threaded sampling loop:

- Maintains a rolling buffer of `TelemetryReading` objects
- Provides `get_window(start_time, end_time)` to extract readings for a specific time range
- Thread-safe for concurrent access from the profiling runner

Energy is attributed to individual queries by matching `get_window()` time ranges to query start/end timestamps.

## Traces

IPW records per-step telemetry during agent execution using two dataclasses defined in `ipw/execution/trace.py`.

**TurnTrace** captures a single agent turn (one LLM call plus tool calls):

| Field | Description |
|-------|-------------|
| `turn_index` | Sequential turn number |
| `input_tokens`, `output_tokens` | Token counts |
| `tools_called` | Tool names invoked this turn |
| `wall_clock_s` | Total wall-clock time |
| `gpu_energy_joules`, `cpu_energy_joules` | Energy consumed |
| `cost_usd` | API cost |

**QueryTrace** aggregates all turns for a single task:

| Field | Description |
|-------|-------------|
| `query_id` | Unique query identifier |
| `turns` | List of `TurnTrace` objects |
| `total_wall_clock_s` | Total execution time |
| `completed` | Whether the task finished |

QueryTrace provides computed properties: `total_input_tokens`, `total_output_tokens`, `total_gpu_energy_joules`, `total_cost_usd`, `tool_call_count`.

### Serialization

Traces support three serialization formats:

- **JSONL** -- `trace.save_jsonl(path)` / `QueryTrace.load_jsonl(path)` (one JSON object per line)
- **Dict** -- `trace.to_dict()` / `QueryTrace.from_dict(data)`
- **HuggingFace Dataset** -- `QueryTrace.to_hf_dataset(traces)` (flat dataset with `trace_json` column for full detail)

## Execution Pipeline

Two runners orchestrate benchmarking workloads:

**ProfilerRunner** (`ipw/execution/profiler_runner.py`) handles single-turn profiling: dataset provides queries, runner sends each to the inference client, telemetry session captures energy data, per-query metrics are computed.

**AgenticRunner** (`ipw/execution/agentic_runner.py`) handles multi-turn agent evaluation: dataset provides tasks, agent harness executes multi-step reasoning with tool calls, each turn is recorded as a `TurnTrace`, and the full `QueryTrace` is saved.

Both runners write results to `./runs/` as timestamped JSON files.
