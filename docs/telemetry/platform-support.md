# Platform Support

The energy monitor auto-detects the best available collector for your hardware. This page documents what metrics are available on each platform.

## Platform Matrix

| Metric | NVIDIA (NVML) | AMD (ROCm) | Apple Silicon | Linux RAPL | Null |
|--------|:---:|:---:|:---:|:---:|:---:|
| GPU power (W) | yes | yes | yes | -- | -- |
| GPU energy (J) | yes | yes | yes | -- | -- |
| GPU temperature (C) | yes | yes | -- | -- | -- |
| GPU memory usage (MB) | yes | yes | -- | -- | -- |
| GPU memory total (MB) | yes | yes | -- | -- | -- |
| GPU compute utilization (%) | yes | yes | -- | -- | -- |
| GPU memory bandwidth util (%) | yes | yes | -- | -- | -- |
| GPU tensor core util (%) | yes* | -- | -- | -- | -- |
| CPU power (W) | via RAPL | via RAPL | yes | -- | -- |
| CPU energy (J) | via RAPL | via RAPL | yes | yes | -- |
| ANE power (W) | -- | -- | yes | -- | -- |
| ANE energy (J) | -- | -- | yes | -- | -- |
| CPU memory usage (MB) | yes | yes | yes | yes | yes |
| System info | yes | yes | yes | yes | yes |
| GPU info | yes | yes | -- | -- | -- |

*Tensor core utilization requires NVIDIA Ampere architecture (A100, RTX 30xx) or newer.

## NVIDIA (NVML)

**Collector**: `energy-monitor/src/collectors/nvidia.rs`

NVML (NVIDIA Management Library) provides comprehensive GPU telemetry. The collector queries:

- `nvmlDeviceGetPowerUsage()` -- instantaneous power draw
- `nvmlDeviceGetTotalEnergyConsumption()` -- cumulative energy counter
- `nvmlDeviceGetTemperature()` -- GPU die temperature
- `nvmlDeviceGetMemoryInfo()` -- memory used/total
- `nvmlDeviceGetUtilizationRates()` -- GPU and memory controller utilization
- Tensor core utilization (Ampere+)

**Energy calculation**: NVML provides a cumulative energy counter in millijoules. The monitor computes per-interval energy by taking the difference between successive readings.

**GPU info**: Reports GPU name, vendor ("NVIDIA"), device ID, and backend ("NVML").

### RAPL Integration

On Linux with Intel or AMD CPUs, NVIDIA systems can additionally report CPU energy via RAPL (Running Average Power Limit). The RAPL collector reads from:

```
/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj
```

This provides:

- Package-level energy in microjoules
- Per-core domain energy (if available)

RAPL readings are added to the `cpu_energy_joules` and `cpu_power_watts` fields.

## AMD (ROCm SMI)

**Collector**: `energy-monitor/src/collectors/amd.rs`

ROCm SMI provides GPU telemetry for AMD GPUs. Metrics include:

- Power usage (instantaneous)
- Energy consumption (cumulative)
- Temperature
- Memory usage and total
- GPU compute utilization
- Memory bandwidth utilization

**GPU info**: Reports GPU name, vendor ("AMD"), device ID, and backend ("ROCm").

RAPL integration for CPU energy works the same as with NVIDIA.

## Apple Silicon (powermetrics)

**Collector**: `energy-monitor/src/collectors/macos.rs`

On macOS with Apple Silicon, the collector uses the `powermetrics` system utility to capture power data for:

- **GPU**: Integrated GPU power and energy
- **CPU**: CPU cluster power and energy
- **ANE**: Apple Neural Engine power and energy (for ML workloads that use ANE)

**Requirements**: `powermetrics` requires root privileges. The energy monitor must be run with `sudo` or the user must configure passwordless sudo for `powermetrics`.

**Limitations**:

- No GPU memory reporting (Apple uses unified memory; system memory is reported instead)
- No GPU utilization percentage
- No temperature reporting
- No GPU info (integrated GPU has no separate device ID)

**Energy calculation**: `powermetrics` reports power in milliwatts. Energy is computed by integrating power over the sampling interval.

## Linux RAPL (CPU-Only)

**Collector**: `energy-monitor/src/collectors/linux_rapl.rs`

When no GPU is detected, the monitor falls back to a RAPL-only collector that reads CPU energy from sysfs:

```
/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj     # Package energy
/sys/class/powercap/intel-rapl/intel-rapl:0:0/energy_uj   # Core energy (if available)
```

**What's available**:

- CPU package energy (joules)
- CPU power (derived from energy / time)
- CPU memory usage

**Requirements**: Read access to the RAPL sysfs interface. May need:

```bash
sudo chmod o+r /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj
```

## Null Collector

**Collector**: `energy-monitor/src/collectors/mod.rs` (fallback)

When no hardware-specific collector is available, the null collector reports:

- CPU memory usage from the OS
- All power, energy, temperature, and GPU fields set to -1
- Platform reported as `"null"`

This allows IPW to run profiling for latency and accuracy metrics even without energy telemetry.

## Checking Your Platform

Run the test script to see which collector is active:

```bash
uv run scripts/test_energy_monitor.py

# Expected output includes:
# Platform: nvidia (or macos, amd, rapl, null)
# GPU: NVIDIA RTX 4090 (or empty if no GPU)
# Power: 45.2 W
# Energy: 2.26 J
# Temperature: 42.0 C
```

The `platform` field in every `TelemetryReading` tells you which collector is active.
