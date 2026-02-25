# Metrics Reference

Complete catalog of all metrics collected by IPW, organized by category.

## Energy Metrics

| Metric | Unit | Type | Description |
|--------|------|------|-------------|
| `power_watts` | W | float | Instantaneous GPU power draw |
| `energy_joules` | J | float | Accumulated GPU energy since baseline |
| `cpu_power_watts` | W | float | Instantaneous CPU power draw |
| `cpu_energy_joules` | J | float | Accumulated CPU energy since baseline |
| `ane_power_watts` | W | float | Apple Neural Engine power (macOS only) |
| `ane_energy_joules` | J | float | Accumulated ANE energy (macOS only) |

### Per-Query Energy (in ProfilingRecord)

| Metric | Unit | Description |
|--------|------|-------------|
| `energy_metrics.per_query_joules` | J | GPU energy consumed for this query |
| `energy_metrics.total_joules` | J | Cumulative GPU energy |
| `energy_metrics.cpu_per_query_joules` | J | CPU energy consumed for this query |
| `energy_metrics.cpu_total_joules` | J | Cumulative CPU energy |
| `energy_metrics.ane_per_query_joules` | J | ANE energy for this query (macOS) |
| `energy_metrics.ane_total_joules` | J | Cumulative ANE energy |

### Platform Availability

| Metric | NVIDIA | AMD | Apple | RAPL | Null |
|--------|:---:|:---:|:---:|:---:|:---:|
| GPU power/energy | yes | yes | yes | -- | -- |
| CPU power/energy | RAPL | RAPL | yes | yes | -- |
| ANE power/energy | -- | -- | yes | -- | -- |

## Temperature Metrics

| Metric | Unit | Type | Description |
|--------|------|------|-------------|
| `temperature_celsius` | C | float | GPU die temperature |

### Per-Query Temperature (in ProfilingRecord)

| Metric | Unit | Description |
|--------|------|-------------|
| `temperature_metrics.avg` | C | Average GPU temp during query |
| `temperature_metrics.max` | C | Peak GPU temp during query |
| `temperature_metrics.median` | C | Median GPU temp during query |
| `temperature_metrics.min` | C | Minimum GPU temp during query |

Available on: NVIDIA, AMD. Returns -1 on Apple Silicon and null collector.

## Memory Metrics

| Metric | Unit | Type | Description |
|--------|------|------|-------------|
| `gpu_memory_usage_mb` | MB | float | Current GPU memory used |
| `gpu_memory_total_mb` | MB | float | Total GPU memory |
| `cpu_memory_usage_mb` | MB | float | Current system memory used |

### Per-Query Memory (in ProfilingRecord)

| Metric | Description |
|--------|-------------|
| `memory_metrics.gpu_mb.{avg,max,median,min}` | GPU memory stats during query |
| `memory_metrics.cpu_mb.{avg,max,median,min}` | CPU memory stats during query |

## Latency Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| `latency_metrics.per_token_ms` | ms | Average time per output token |
| `latency_metrics.throughput_tokens_per_sec` | tok/s | Output token throughput |
| `latency_metrics.time_to_first_token_seconds` | s | Time from request to first token |
| `latency_metrics.total_query_seconds` | s | Total wall-clock time for the query |

All latency metrics are computed from Python-side timestamps, not from the energy monitor.

## Token Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| `token_metrics.input` | tokens | Number of input/prompt tokens |
| `token_metrics.output` | tokens | Number of output/completion tokens |
| `token_metrics.total` | tokens | Total tokens (input + output) |

Token counts come from the inference client's response (`ChatUsage`).

## Power Metrics

| Metric | Description |
|--------|-------------|
| `power_metrics.gpu.per_query_watts.{avg,max,median,min}` | GPU power stats during query |
| `power_metrics.gpu.total_watts.{avg,max,median,min}` | Cumulative GPU power stats |
| `power_metrics.cpu.per_query_watts.{avg,max,median,min}` | CPU power stats during query |
| `power_metrics.cpu.total_watts.{avg,max,median,min}` | Cumulative CPU power stats |

## GPU Utilization Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| `gpu_compute_utilization_pct` | % | SM/compute core utilization |
| `gpu_memory_bandwidth_utilization_pct` | % | Memory controller utilization |
| `gpu_tensor_core_utilization_pct` | % | Tensor core utilization (Ampere+) |

### Per-Query Utilization (in ProfilingRecord)

| Metric | Description |
|--------|-------------|
| `hardware_utilization.gpu.compute_utilization_pct` | Avg compute util during query |
| `hardware_utilization.gpu.memory_bandwidth_utilization_pct` | Avg memory BW util |
| `hardware_utilization.gpu.tensor_core_utilization_pct` | Avg tensor core util |
| `hardware_utilization.gpu.memory_used_gb` | GPU memory used (GB) |
| `hardware_utilization.gpu.memory_total_gb` | GPU memory total (GB) |

### Derived Utilization

| Metric | Description |
|--------|-------------|
| `hardware_utilization.derived.mfu` | Model FLOPs Utilization |
| `hardware_utilization.derived.mbu` | Model Bandwidth Utilization |
| `hardware_utilization.derived.arithmetic_intensity` | FLOPs per byte ratio |

## Phase Metrics

Phase metrics break down energy and latency into prefill (processing input) and decode (generating output) phases:

| Metric | Unit | Description |
|--------|------|-------------|
| `phase_metrics.prefill_energy_j` | J | Energy during prefill |
| `phase_metrics.decode_energy_j` | J | Energy during decode |
| `phase_metrics.prefill_duration_ms` | ms | Prefill wall-clock time |
| `phase_metrics.decode_duration_ms` | ms | Decode wall-clock time |
| `phase_metrics.prefill_power_avg_w` | W | Average power during prefill |
| `phase_metrics.decode_power_avg_w` | W | Average power during decode |
| `phase_metrics.prefill_energy_per_input_token_j` | J/tok | Prefill energy efficiency |
| `phase_metrics.decode_energy_per_output_token_j` | J/tok | Decode energy efficiency |

Phase separation requires detecting the time-to-first-token boundary.

## Compute Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| `compute_metrics.total_flops` | FLOPs | Estimated total FLOPs for the query |
| `compute_metrics.flops_per_token` | FLOPs/tok | Estimated FLOPs per token |
| `compute_metrics.flops_per_request` | FLOPs | Same as total_flops |

FLOPs are estimated using the 2*P*T formula (where P = model parameters, T = tokens) or the optional `calflops` library.

## Cost Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| `cost.input_cost_usd` | USD | Cost of input tokens |
| `cost.output_cost_usd` | USD | Cost of output tokens |
| `cost.tool_cost_usd` | USD | Cost of tool calls (e.g., Tavily) |
| `cost.total_cost_usd` | USD | Total API cost |

Cost is computed from the pricing tables in `ipw/cost/pricing.py` based on provider, model, and token counts.

## Efficiency Metrics (Computed)

These are computed during accuracy analysis, not captured per-query:

| Metric | Formula | Description |
|--------|---------|-------------|
| Intelligence Per Joule (IPJ) | accuracy / avg_energy_per_query | Accuracy per joule of energy |
| Intelligence Per Watt (IPW) | accuracy / avg_power_per_query | Accuracy per watt of power |

## Sentinel Values

The energy monitor uses these sentinel values for unavailable metrics:

- **-1** (Rust/proto): The metric is not available on this platform
- **None** (Python): The metric was not captured or is not applicable
- **0** : The metric is available but the reading was zero

Always validate with `math.isfinite(value) and value > 0` before using energy/power values in efficiency calculations.
