# Platform Support

The energy monitor auto-detects the best available collector for your hardware. This page documents metric availability per platform and the complete metrics reference.

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

Provides comprehensive GPU telemetry via NVML: power, energy (cumulative counter in millijoules), temperature, memory, compute utilization, and tensor core utilization (Ampere+). GPU info includes name, vendor, device ID, and backend.

On Linux with Intel or AMD CPUs, CPU energy is additionally reported via RAPL by reading `/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj`.

## AMD (ROCm SMI)

Provides GPU power, energy, temperature, memory, compute utilization, and memory bandwidth utilization via ROCm SMI. GPU info includes name, vendor, device ID, and backend.

RAPL integration for CPU energy works the same as with NVIDIA.

## Apple Silicon

Uses the `powermetrics` system utility to capture power data for CPU, GPU, and ANE (Apple Neural Engine). Requires root privileges -- configure passwordless sudo for `powermetrics` (see [Benchmarking Overview](overview.md#building-the-energy-monitor)).

**Limitations:** No GPU memory reporting (unified memory architecture), no GPU utilization percentage, no temperature reporting, no GPU info.

## Linux RAPL (CPU-Only)

Fallback when no GPU is detected. Reads CPU energy from sysfs:

```
/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj
```

Provides CPU package energy (joules) and derived CPU power. May require read permissions:

```bash
sudo chmod o+r /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj
```

## Null Collector

Reports only CPU memory usage from the OS. All power, energy, temperature, and GPU fields return -1. Platform reported as `"null"`. Allows IPW to run profiling for latency and accuracy metrics without energy telemetry.

## Metrics Reference

### Energy Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| `power_watts` | W | Instantaneous GPU power draw |
| `energy_joules` | J | Accumulated GPU energy since baseline |
| `cpu_power_watts` | W | Instantaneous CPU power draw |
| `cpu_energy_joules` | J | Accumulated CPU energy since baseline |
| `ane_power_watts` | W | Apple Neural Engine power (macOS only) |
| `ane_energy_joules` | J | Accumulated ANE energy (macOS only) |

Per-query variants: `energy_metrics.per_query_joules`, `energy_metrics.cpu_per_query_joules`, `energy_metrics.ane_per_query_joules`.

### Temperature Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| `temperature_metrics.{avg,max,min}` | C | GPU temperature stats during query |

Available on: NVIDIA, AMD. Returns -1 on Apple Silicon and null collector.

### Memory Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| `gpu_memory_usage_mb` / `gpu_memory_total_mb` | MB | GPU memory used / total |
| `cpu_memory_usage_mb` | MB | Current system memory used |

### Latency Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| `latency_metrics.per_token_ms` | ms | Average time per output token |
| `latency_metrics.throughput_tokens_per_sec` | tok/s | Output token throughput |
| `latency_metrics.time_to_first_token_seconds` | s | Time from request to first token |
| `latency_metrics.total_query_seconds` | s | Total wall-clock time for the query |

### Token Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| `token_metrics.input` | tokens | Number of input/prompt tokens |
| `token_metrics.output` | tokens | Number of output/completion tokens |
| `token_metrics.total` | tokens | Total tokens (input + output) |

### Power Metrics

| Metric | Description |
|--------|-------------|
| `power_metrics.gpu.per_query_watts.{avg,max,min}` | GPU power stats during query |
| `power_metrics.cpu.per_query_watts.{avg,max,min}` | CPU power stats during query |

### GPU Utilization Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| `gpu_compute_utilization_pct` | % | SM/compute core utilization |
| `gpu_memory_bandwidth_utilization_pct` | % | Memory controller utilization |
| `gpu_tensor_core_utilization_pct` | % | Tensor core utilization (Ampere+) |

Per-query utilization is available in `hardware_utilization.gpu.*` fields.

### Derived Utilization

| Metric | Description |
|--------|-------------|
| `hardware_utilization.derived.mfu` | Model FLOPs Utilization |
| `hardware_utilization.derived.mbu` | Model Bandwidth Utilization |
| `hardware_utilization.derived.arithmetic_intensity` | FLOPs per byte ratio |

### Phase Metrics

Break down energy and latency into prefill (processing input) and decode (generating output):

| Metric | Unit | Description |
|--------|------|-------------|
| `phase_metrics.{prefill,decode}_energy_j` | J | Energy per phase |
| `phase_metrics.{prefill,decode}_duration_ms` | ms | Wall-clock time per phase |
| `phase_metrics.{prefill,decode}_energy_per_*_token_j` | J/tok | Energy efficiency per phase |

### Compute Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| `compute_metrics.total_flops` | FLOPs | Estimated total FLOPs for the query |
| `compute_metrics.flops_per_token` | FLOPs/tok | Estimated FLOPs per token |

FLOPs are estimated using the 2\*P\*T formula (P = model parameters, T = tokens).

### Cost Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| `cost.input_cost_usd` | USD | Cost of input tokens |
| `cost.output_cost_usd` | USD | Cost of output tokens |
| `cost.tool_cost_usd` | USD | Cost of tool calls (e.g., Tavily) |
| `cost.total_cost_usd` | USD | Total API cost |

### Efficiency Metrics (Computed)

| Metric | Formula | Description |
|--------|---------|-------------|
| Intelligence Per Joule (IPJ) | accuracy / avg_energy_per_query | Accuracy per joule of energy |
| Intelligence Per Watt (IPW) | accuracy / avg_power_per_query | Accuracy per watt of power |

## Sentinel Values

The energy monitor uses these sentinel values for unavailable metrics:

- **-1**: metric not available on this platform (set by Rust/proto layer)
- **None**: metric not captured or not applicable (Python layer)
- **0**: genuine zero reading (metric is available but value is zero)

Validate before using in efficiency calculations:

```python
math.isfinite(value) and value > 0
```
