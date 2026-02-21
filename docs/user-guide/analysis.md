# Analysis

The `ipw analyze` command runs post-profiling analysis on results directories. IPW ships with two analysis providers: `accuracy` (default) and `regression`.

## Command Reference

```bash
ipw analyze <results_dir> [--analysis <type>]
```

| Option | Default | Description |
|--------|---------|-------------|
| `<results_dir>` | required | Path to the profiling results directory |
| `--analysis` | `accuracy` | Analysis type (`accuracy` or `regression`) |

## Accuracy Analysis

The default analysis computes IPJ and IPW metrics. It works by:

1. Loading the Arrow dataset from the results directory.
2. Checking each record for evaluation data. If evaluation is missing, it instantiates the original dataset provider and scores unevaluated records using `dataset.score()`.
3. Scoring uses either exact match (for MCQ datasets like MMLU-Pro) or an LLM judge (for open-ended datasets).
4. Aggregating results into accuracy, energy, and power statistics.
5. Computing the final efficiency metrics.

```bash
ipw analyze ./runs/profile_nvidia_llama3.2_1b_ipw/
```

### Output Metrics

The accuracy analysis produces a JSON report at `analysis/accuracy.json`:

```json
{
  "analysis": "accuracy",
  "summary": {
    "model": "llama3.2:1b",
    "correct": 42,
    "incorrect": 58,
    "total_scored": 100,
    "accuracy": 0.42,
    "intelligence_per_joule": 0.0084,
    "intelligence_per_watt": 0.0028,
    "avg_per_query_energy_joules": 50.0,
    "avg_per_query_power_watts": 150.0
  }
}
```

### Key Metrics Explained

| Metric | Formula | Description |
|--------|---------|-------------|
| Accuracy | correct / total_scored | Fraction of correctly answered queries |
| IPJ | accuracy / avg_energy_per_query | Intelligence Per Joule -- higher is better |
| IPW | accuracy / avg_power_per_query | Intelligence Per Watt -- higher is better |

### Energy Imputation

When per-query energy readings are zero or negative (common on platforms without energy counters), IPW attempts to impute energy from power and latency:

```
imputed_energy = avg_power_watts * query_latency_seconds
```

The analysis report notes how many values were imputed.

### Evaluation Retry

Failed evaluations are retried up to 3 times (configurable via `AccuracyAnalysis.MAX_EVALUATION_ATTEMPTS`). Each attempt is tracked in the evaluation metadata.

## Regression Analysis

The regression analysis fits statistical models to understand how energy, power, and latency scale with input/output length:

```bash
ipw analyze ./runs/profile_nvidia_llama3.2_1b_ipw/ --analysis regression
```

This produces `analysis/regression.json` with coefficients for:

- Energy (joules) vs. input token count
- Energy (joules) vs. output token count
- Power (watts) vs. input token count
- Latency (seconds) vs. output token count

## Custom Analysis Providers

You can create custom analysis providers by subclassing `AnalysisProvider` and registering with `@AnalysisRegistry.register("id")`:

```python
from ipw.analysis.base import AnalysisContext, AnalysisProvider, AnalysisResult
from ipw.core.registry import AnalysisRegistry


@AnalysisRegistry.register("my-analysis")
class MyAnalysis(AnalysisProvider):
    analysis_id = "my-analysis"

    def run(self, context: AnalysisContext) -> AnalysisResult:
        # Load data from context.results_dir
        # Perform analysis
        # Return AnalysisResult
        ...
```

Then use it via the CLI:

```bash
ipw analyze ./runs/profile_* --analysis my-analysis
```

## Running Multiple Analyses

You can run different analysis types on the same results:

```bash
# Default accuracy analysis
ipw analyze ./runs/profile_nvidia_llama3.2_1b_ipw/

# Then regression analysis
ipw analyze ./runs/profile_nvidia_llama3.2_1b_ipw/ --analysis regression
```

Each analysis writes to a separate file in the `analysis/` subdirectory without overwriting the others.
