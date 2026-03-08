# Energy Monitor

The energy monitor is a standalone Rust gRPC service that streams hardware telemetry at 50ms intervals. It runs as a subprocess alongside the Python profiler and provides real-time power, energy, temperature, memory, and utilization readings.

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

### Components

**Rust binary** (`energy-monitor/`): The compiled binary is located at `ipw/telemetry/bin/<platform>/energy-monitor`. Platform variants:

- `linux-x86_64/energy-monitor`
- `macos-arm64/energy-monitor`

**Python launcher** (`ipw/telemetry/launcher.py`): Manages the subprocess lifecycle -- starting, monitoring, and stopping the energy monitor process.

**Python collector** (`ipw/telemetry/collector.py`): gRPC client that connects to the energy monitor's streaming endpoint and converts protobuf messages into Python `TelemetryReading` objects.

## Proto Definition

The service is defined in `energy-monitor/proto/energy.proto`:

```protobuf
service EnergyMonitor {
  rpc Health(HealthRequest) returns (HealthResponse);
  rpc StreamTelemetry(StreamRequest) returns (stream TelemetryReading);
}

message TelemetryReading {
  double power_watts = 1;
  double energy_joules = 2;
  double temperature_celsius = 3;
  double gpu_memory_usage_mb = 4;
  double cpu_memory_usage_mb = 5;
  double cpu_power_watts = 10;
  double cpu_energy_joules = 11;
  double ane_power_watts = 12;
  double ane_energy_joules = 13;
  double gpu_compute_utilization_pct = 14;
  double gpu_memory_bandwidth_utilization_pct = 15;
  double gpu_tensor_core_utilization_pct = 16;
  double gpu_memory_total_mb = 17;
  string platform = 6;
  int64 timestamp_nanos = 7;
  SystemInfo system_info = 8;
  GpuInfo gpu_info = 9;
}
```

### Unavailable Metrics

Fields that the current platform cannot provide are set to `-1`. Python code should check for this sentinel with `math.isfinite()` or compare against 0 before using values in calculations.

## Collector Auto-Detection

The Rust service auto-detects the appropriate collector at startup:

1. **macOS** (`cfg(target_os = "macos")`): Uses `powermetrics` for CPU/GPU/ANE power readings. Requires `sudo`.

2. **Linux/Windows with NVIDIA GPU**: Uses NVML. Falls through to next option if NVML initialization fails.

3. **Linux with AMD GPU**: Uses ROCm SMI. Falls through if unavailable.

4. **Linux with RAPL**: Uses Intel RAPL for CPU energy counters. Available on Intel and some AMD CPUs.

5. **Null collector**: Fallback that reports only CPU memory usage. All power/energy/temperature fields return -1.

The selected platform is reported in each `TelemetryReading.platform` field (e.g., `"nvidia"`, `"macos"`, `"amd"`, `"rapl"`, `"null"`).

## Platform Setup

=== "NVIDIA"

    **Prerequisites:** Rust and `protoc` must be installed. See [Installation](../getting-started/installation.md).

    1. Install NVIDIA drivers (if not already present):
    ```bash
    sudo apt install nvidia-driver-560
    ```

    2. Build and stage the energy monitor:
    ```bash
    uv run scripts/build_energy_monitor.py
    ```

    3. Test the energy monitor:
    ```bash
    uv run scripts/test_energy_monitor.py
    ```

=== "AMD"

    **Prerequisites:** Rust and `protoc` must be installed. See [Installation](../getting-started/installation.md).

    1. Install ROCm:
    ```bash
    sudo apt install rocm rocm-smi
    ```

    2. Export paths:
    ```bash
    export LIBRARY_PATH=/opt/rocm/lib:$LIBRARY_PATH
    export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
    ```

    3. Build the energy monitor with AMD support:
    ```bash
    cd energy-monitor && cargo build --release --features amd
    ```

    4. Copy the binary to the staging directory:
    ```bash
    cp ./target/release/energy-monitor ../intelligence-per-watt/src/ipw/telemetry/bin/linux-x86_64/energy-monitor
    cd ..
    ```

    !!! note
        The AMD build requires the `--features amd` flag to enable ROCm SMI support. The build script (`scripts/build_energy_monitor.py`) does not currently pass feature flags, so a manual build is needed here.

    5. Test the energy monitor:
    ```bash
    uv run scripts/test_energy_monitor.py
    ```

=== "Apple"

    Tested on Apple M4 Pro.

    **Prerequisites:** Rust and `protoc` must be installed. See [Installation](../getting-started/installation.md). On macOS, install `protoc` via Homebrew: `brew install protobuf`.

    1. Build and stage the energy monitor:
    ```bash
    uv run scripts/build_energy_monitor.py
    ```

    2. Configure passwordless `sudo` for `powermetrics`:

    The macOS collector requires `sudo` to run `powermetrics`. Without this configuration, you will be prompted for your password each time the energy monitor starts.

    ```bash
    sudo sh -c "echo \"$(whoami) ALL=(ALL) NOPASSWD: /usr/bin/powermetrics\" > /etc/sudoers.d/powermetrics"
    ```

    !!! warning
        This grants your user passwordless sudo access to `powermetrics` only. Review your organization's security policies before modifying sudoers.

    3. Test the energy monitor:
    ```bash
    uv run scripts/test_energy_monitor.py
    ```

## Testing

```bash
uv run scripts/test_energy_monitor.py [--interval 2.0]
```

This starts the energy monitor, connects via gRPC, and prints telemetry readings at the specified interval (default: 1 second). It helps verify that the binary works correctly on your platform.

## Streaming Interval

The service streams telemetry at a fixed 50ms interval. This provides sufficient time resolution to:

- Attribute energy to individual inference queries (typically 0.5-30 seconds)
- Capture power spikes during GPU-intensive prefill phases
- Track memory usage changes during batch processing

The interval is not configurable -- it is hardcoded in the `StreamRequest` handler to ensure consistent measurement across deployments.

## Python Integration

### TelemetrySession

The `TelemetrySession` (`ipw/execution/telemetry_session.py`) wraps the gRPC client in a threaded sampling loop:

- Maintains a rolling buffer of `TelemetryReading` objects
- Provides `get_window(start_time, end_time)` to extract readings for a specific time range
- Thread-safe for concurrent access from the profiling runner

### Launcher

The `Launcher` (`ipw/telemetry/launcher.py`) manages the energy monitor process:

- Finds the correct binary for the current platform
- Starts the process and waits for the gRPC server to become healthy
- Stops the process when profiling is complete
- Handles process crashes and cleanup
