# ipw bench Pipeline Completion — Design

Date: 2026-03-08

## Goal

Make `ipw bench --auto-server` a single command that handles the full profiling lifecycle — server spin-up, telemetry, benchmark execution, accuracy evaluation, results export, and server shutdown — with complete, well-structured output data.

## Current State

`ipw bench --auto-server` already orchestrates: server start → warmup → telemetry-wrapped benchmark → display results → server stop. However:

- Accuracy evaluation requires a separate `ipw analyze` call
- `is_resolved` is null for non-TerminalBench datasets
- IPJ/IPW cannot be computed without accuracy
- MBU is sampled by the Rust energy monitor but never aggregated
- Statistics section exists in code but isn't always populated
- Outlier-sensitive metrics have no trimmed variant
- `ipw bench` is undocumented

## Changes

### 1. Integrated Accuracy Evaluation

At the end of `execute_benchmark()`, after traces are collected but before `summary.json` is written:

1. For each query trace, evaluate accuracy using the dataset's `score()` method (same logic currently in `AccuracyAnalysis`)
2. Set `is_resolved` on each `QueryTrace`
3. Compute aggregate accuracy, IPJ, and IPW
4. Write into `summary.json`

Per-dataset scoring:
- **TerminalBench**: Already sets `is_resolved` during the run — no change
- **GAIA, GPQA, etc.**: Use the dataset's `score(record, response)` method (exact/fuzzy match or LLM-as-judge)

For LLM-as-judge datasets, this adds eval cost identical to what `ipw analyze` would cost separately.

### 2. MBU Enrichment

The Rust energy monitor already samples `gpu_memory_bandwidth_utilization_pct` at 50ms intervals via NVML. This data is available in `TelemetrySession` samples but is discarded during metric computation.

Extract MBU from existing samples in `_compute_energy_metrics()` and the agentic runner's energy correlation:

Per-query in `traces.jsonl`:
- `query_mbu_avg_pct` — mean MBU during query time window
- `query_mbu_max_pct` — peak MBU during query time window

In `summary.json` statistics:
- `mbu_avg_pct`: `{avg, median, min, max, std}` across queries

### 3. Terminal Output Changes

**Metrics table** — add one new row:
- `MBU` (avg/median/min/max/std) with unit `%`

**Efficiency panel** — now always populated:
- Accuracy (was sometimes N/A, now always present)
- IPJ (was sometimes N/A, now always computed)
- IPW (was sometimes N/A, now always computed)
- Resolved count: `Resolved: 35/165`
- Total Energy, Avg Power (unchanged)

### 4. `summary.json` Schema

```json
{
  "config": {
    "agent": "openhands",
    "model": "Qwen/Qwen3.5-397B-A17B-FP8",
    "dataset": "gaia",
    "client": "vllm",
    "client_base_url": "http://localhost:8000/v1",
    "max_queries": null,
    "concurrency": 1,
    "telemetry_granularity": "benchmark",
    "auto_server": true,
    "seed": null
  },
  "totals": {
    "queries": 165,
    "completed": 165,
    "resolved": 35,
    "unresolved": 130,
    "accuracy": 0.212,
    "turns": 165,
    "tool_calls": 0,
    "input_tokens": 15509890,
    "output_tokens": 1649314,
    "total_tokens": 17159204,
    "wall_clock_s": 8656.14,
    "gpu_energy_joules": 969264.69,
    "cpu_energy_joules": null,
    "cost_usd": 0.0
  },
  "efficiency": {
    "accuracy": 0.212,
    "total_gpu_energy_joules": 969264.69,
    "total_cpu_energy_joules": null,
    "avg_gpu_power_watts": 2128.30,
    "avg_cpu_power_watts": null,
    "ipj": 2.19e-07,
    "ipw": 9.96e-05
  },
  "normalized_statistics": {
    "_description": "Statistics recomputed after removing the top 5% and bottom 5% of queries by wall_clock_s",
    "_outliers_removed": { "top_pct": 5, "bottom_pct": 5, "queries_before": 165, "queries_after": 149 }
  },
  "normalized_efficiency": {
    "_description": "Efficiency recomputed on the trimmed query set",
    "_outliers_removed": { "top_pct": 5, "bottom_pct": 5, "queries_before": 165, "queries_after": 149 }
  },
  "statistics": {
    "wall_clock_s": { "avg": 52.46, "median": 24.49, "min": 2.52, "max": 2301.94, "std": 191.81 },
    "gpu_energy_joules": { "avg": 37279.41, "median": 38245.59, "min": 3191.26, "max": 64837.71, "std": 21020.40 },
    "cpu_energy_joules": null,
    "gpu_power_watts": { "avg": 2128.30, "median": 2163.93, "min": 1482.42, "max": 2396.98, "std": 199.17 },
    "cpu_power_watts": null,
    "input_tokens": { "avg": 93999.33, "median": 59455.0, "min": 3343.0, "max": 1432871.0, "std": 158943.21 },
    "output_tokens": { "avg": 9995.84, "median": 4946.0, "min": 124.0, "max": 205381.0, "std": 22611.51 },
    "total_tokens": { "avg": 103995.18, "median": 65346.0, "min": 3467.0, "max": 1590603.0, "std": 180340.19 },
    "throughput_tokens_per_sec": { "avg": 223.77, "median": 200.36, "min": 28.16, "max": 804.37, "std": 124.82 },
    "energy_per_token_joules": { "avg": 4.73, "median": 2.97, "min": 0.21, "max": 41.24, "std": 7.83 },
    "mbu_avg_pct": { "avg": null, "median": null, "min": null, "max": null, "std": null },
    "cost_usd": { "avg": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "std": 0.0 },
    "turns": { "avg": 1.0, "median": 1.0, "min": 1.0, "max": 1.0, "std": 0.0 },
    "tool_calls": { "avg": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "std": 0.0 }
  },
  "generated_at": "2026-03-08T01:53:00"
}
```

`normalized_statistics` has the same keys as `statistics`. `normalized_efficiency` has the same keys as `efficiency`. Both are recomputed on the query set after removing the top 5% and bottom 5% of queries sorted by `wall_clock_s`.

### 5. Per-Query Data (`traces.jsonl`)

New/changed fields per query:
- `is_resolved` — now populated for all datasets (was null for non-TerminalBench)
- `query_mbu_avg_pct` — mean MBU during query time window
- `query_mbu_max_pct` — peak MBU during query time window

All existing fields unchanged.

### 6. Documentation

- **Document `ipw bench`**: New section covering the full pipeline, all flags, model presets, `--auto-server`, `--submodel`, output format
- **Document `ipw servers`**: Brief section covering `start`, `launch`, `stop`, `status` subcommands
- **Update CLI flag docs**: Add missing flags (`--per-action`, `--auto-server`, `--submodel`, `--preset`, `--query-timeout`, `--warmup-queries`)
- **Fix minor issues**: Create abbreviations file, fix nav label ("Telemetry" → "Benchmarking")

## Not Doing

- TTFT tracking
- MFU computation (requires model architecture configs)
- Markdown report generation
- Cross-run aggregation command
- HTML reports
- New CLI commands

## Key Files to Modify

| File | Change |
|------|--------|
| `src/ipw/cli/bench.py` | Integrate accuracy eval, MBU extraction, normalized stats, updated terminal display |
| `src/ipw/cli/_display.py` | Add MBU row, resolved count in efficiency panel |
| `src/ipw/execution/exporters.py` | Add efficiency, normalized_statistics, normalized_efficiency sections to summary.json |
| `src/ipw/execution/trace.py` | Add `is_resolved`, `query_mbu_avg_pct`, `query_mbu_max_pct` fields to QueryTrace |
| `src/ipw/execution/agentic_runner.py` | Pass MBU data through, populate is_resolved |
| `docs/user-guide/profiling.md` | Document ipw bench |
| `docs/user-guide/servers.md` (or inline) | Document ipw servers |
