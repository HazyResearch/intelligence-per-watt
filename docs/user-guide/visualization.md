# Visualization

The `ipw plot` command generates visualizations from profiling results. IPW ships with two visualization providers: `regression` (scatter plots with regression lines) and `output-kde` (output length distributions).

## Command Reference

```bash
ipw plot <results_dir> [--output <dir>]
```

| Option | Default | Description |
|--------|---------|-------------|
| `<results_dir>` | required | Path to the profiling results directory |
| `--output` | `<results_dir>/plots/` | Output directory for plots |

## Built-in Visualizations

### Regression Scatter Plots

Generates scatter plots of energy, power, and latency against input and output token counts, with fitted regression lines:

- Energy (J) vs. Input Tokens
- Energy (J) vs. Output Tokens
- Power (W) vs. Input Tokens
- Latency (s) vs. Output Tokens

Each plot shows individual data points and the best-fit line, making it easy to see how resource consumption scales with workload.

### Output KDE

Generates a kernel density estimate (KDE) plot showing the distribution of output token lengths across all queries. This helps identify whether the model tends to produce short or long responses for the given dataset.

## Output

Plots are saved as PNG files in the output directory:

```
runs/profile_<hardware>_<model>/
    plots/
        regression.png
        output_kde.png
```

## Custom Visualization Providers

Create custom visualizations by subclassing `VisualizationProvider` and registering:

```python
from ipw.visualization.base import VisualizationProvider
from ipw.core.registry import VisualizationRegistry


@VisualizationRegistry.register("my-viz")
class MyVisualization(VisualizationProvider):
    visualization_id = "my-viz"

    def run(self, results_dir, output_dir=None):
        # Load data from results_dir
        # Generate plots with matplotlib
        # Save to output_dir
        ...
```

## Listing Available Visualizations

```bash
ipw list visualizations
```
