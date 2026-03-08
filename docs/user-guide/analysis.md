# Analysis & Visualization

## Analyzing Results

The `ipw analyze` command runs post-profiling analysis on results directories.

```bash
ipw analyze <results_dir> [--analysis <type>]
```

IPW ships with two analysis providers:

- **accuracy** (default) -- computes accuracy, IPJ, IPW, and energy/power statistics.
- **regression** -- fits models for energy, power, and latency vs. token counts.

Run multiple analyses on the same results:

```bash
ipw analyze ./runs/profile_* --analysis accuracy --analysis regression
```

Each analysis writes to a separate file in the `analysis/` subdirectory without overwriting the others.

## Key Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Accuracy** | correct / total_scored | Fraction of correctly answered queries |
| **IPJ** | accuracy / avg_energy_per_query | Intelligence Per Joule -- higher is better |
| **IPW** | accuracy / avg_power_per_query | Intelligence Per Watt -- higher is better |

**Energy imputation:** When per-query energy readings are zero or negative (common on platforms without hardware energy counters), energy is imputed from power and latency:

```
imputed_energy = avg_power_watts × query_latency_seconds
```

## Regression Analysis

The regression provider fits statistical models to understand how energy, power, and latency scale with input/output length:

```bash
ipw analyze ./runs/profile_nvidia_llama3.2_1b_ipw/ --analysis regression
```

This produces `analysis/regression.json` with coefficients for:

- Energy (joules) vs. input token count
- Energy (joules) vs. output token count
- Power (watts) vs. input token count
- Latency (seconds) vs. output token count

## Visualization

The `ipw plot` command generates charts from profiling results.

```bash
ipw plot <results_dir> [--output <dir>]
```

Built-in plots:

- **Regression scatter plots** -- Energy, power, and latency vs. input/output tokens with fitted regression lines.
- **Output KDE** -- Kernel density estimate of the output token length distribution.

Output files are PNG images saved to `<results_dir>/plots/` by default, or to a custom directory via `--output`.

### Custom Visualizations

Subclass `VisualizationProvider` and register with the decorator:

```python
from ipw.core.registry import VisualizationRegistry

@VisualizationRegistry.register("my-plot")
class MyPlot(VisualizationProvider):
    ...
```

## Output Formats

### Arrow Dataset

The primary output format. Location: `data-00000-of-00001.arrow` inside the results directory.

Each row represents one query. Key fields:

| Field | Description |
|-------|-------------|
| `problem` | The input query text |
| `answer` | The ground-truth answer |
| `subject` | Dataset category or subject |
| `model_answers` | Model-generated responses |
| `model_metrics` | Nested struct with `energy_metrics`, `latency_metrics`, `power_metrics`, `token_metrics`, `cost` |

Loading an Arrow dataset:

```python
from datasets import load_from_disk

ds = load_from_disk("runs/profile_nvidia_llama3.2_1b_ipw/")
energy = ds[0]["model_metrics"]["model"]["energy_metrics"]["per_query_joules"]
```

### JSONL Traces (Agentic Runs)

Agentic runs produce `traces.jsonl` -- one `QueryTrace` per line.

- **QueryTrace**: `query_id`, `turns[]`, `total_wall_clock_s`, `completed`
- **TurnTrace**: `turn_index`, `tokens`, `tools_called`, `energy`, `power`, `cost`

Loading traces:

```python
from pathlib import Path
from ipw.execution.traces import QueryTrace

traces = QueryTrace.load_jsonl(Path("runs/run_react_gpt4o_gaia/traces.jsonl"))
```

### Summary & Reports

Each results directory contains:

| File | Contents |
|------|----------|
| `summary.json` | Dataset, model, client, hardware info, timing |
| `analysis/accuracy.json` | Accuracy, IPJ, IPW, energy/power statistics |
| `analysis/regression.json` | Regression coefficients (after regression analysis) |
