# Intelligence Per Watt

Python package for profiling and analyzing inference service performance across hardware configurations.

CLI quick start: `ipw profile --client <id> --model <name> --dataset <dataset_id>` runs profiling and then immediately scores the run to produce Intelligence Per Watt metrics in the output directory.

## Project Structure

```
src/ipw/
├── analysis/          Analysis implementations (regression)
├── cli/               Command-line interface (profile, analyze, plot, list, energy)
├── clients/           Inference service clients (ollama, vLLM, OpenAI eval)
├── core/              Component registration and shared types
├── datasets/          Dataset providers (built-in Intelligence Per Watt dataset)
├── execution/         Profiling orchestration and telemetry collection
├── evaluation/        Judge handlers used during scoring
├── telemetry/         Energy monitoring integration (gRPC client, launcher)
├── tests/             Test suite
└── visualization/     Plotting and visualization (KDE, regression, telemetry timeline plots)
```

## Installation

Install from the repository root:

```bash
# from the repo root
uv pip install -e intelligence-per-watt
```

Optional client dependencies live in extras so you only install what you need:

```bash
# from the repo root
uv pip install -e 'intelligence-per-watt[ollama]'       # Enable the Ollama client
uv pip install -e 'intelligence-per-watt[vllm]'         # Enable the vLLM offline client
uv pip install -e 'intelligence-per-watt[all]'          # Pull in everything
```

Development installation:

```bash
cd intelligence-per-watt
uv venv
source .venv/bin/activate
uv pip install -e .
```

Build energy-monitor binary:

```bash
uv run ../scripts/build_energy_monitor.py
```

## Running Tests

Run the full test suite:

```bash
pytest
```

Run specific test modules:

```bash
pytest src/ipw/tests/clients/
pytest src/ipw/tests/analysis/test_regression.py
```

## Evaluation and accuracy metrics

Accuracy is produced by the `accuracy` analysis. Profiling collects responses,
then runs this analysis to score each answer with a judge model and persist the
results back into the saved dataset.

### Requirements

- Set `IPW_EVAL_API_KEY` or `OPENAI_API_KEY` for the judge endpoint. If both are
  set and differ, `IPW_EVAL_API_KEY` is used. Keys in a `.env` file are loaded
  automatically. The `ipw` dataset will fail fast if a key is missing.
- The default judge provider is OpenAI-compatible at
  `https://api.openai.com/v1` with model
  `gpt-5-nano-2025-08-07`. This is the recommended default for IPW scoring; if
  you need to use a different judge, set `--eval-base-url`, `--eval-model`, and
  `--eval-client`.

### Typical flow

Run profiling as usual; when it finishes, the CLI immediately runs the accuracy
analysis on the output directory:

```bash
ipw profile \
  --client <client_id> \
  --model <model_name> \
  --eval-client openai \
  --eval-base-url https://api.openai.com/v1 \
  --eval-model gpt-5-nano-2025-08-07 \
  --dataset ipw \
  --output-dir /path/to/runs
```

What gets written:
- Each record gains `model_metrics[<model>]["evaluation"]` with `is_correct` and
  a JSON `metadata` payload from the scorer.
- `<run>/analysis/accuracy.json` summarizes `correct`, `incorrect`,
  `unevaluated`, and overall accuracy.

### Manually re-running accuracy

If you need to backfill (for example, after adding an API key) or re-score a
run, execute:

```bash
ipw analyze /path/to/profile_RUN --analysis accuracy --option model=<model_name>
```

When only one model exists in the dataset you can omit `--option model=...`.
The analysis only evaluates rows where `evaluation.is_correct` is missing, so
reruns are safe and idempotent.

## Extending Intelligence Per Watt

Add matching tests under `src/ipw/tests/` whenever you add a client, dataset, analysis, or visualization so automated runs cover the new component.

### Registry System

Intelligence Per Watt uses a centralized registry (`core/registry.py`) for component discovery. All clients, datasets, analyses, and visualizations register themselves on import, making them available via CLI commands. Components can be listed via:

```bash
ipw list all
```

When optional client dependencies (e.g., `ollama`, `vllm`) are not installed, `ipw list clients` reports them under “Unavailable clients” with the extra (e.g., `uv pip install '.[ollama]'`) needed to enable them.

### Adding a New Client

Create a new client in `clients/`:

```python
# clients/custom.py
from ..core.registry import ClientRegistry
from ..core.types import Response
from .base import InferenceClient

@ClientRegistry.register("custom")
class CustomClient(InferenceClient):
    client_id = "custom"
    client_name = "Custom"
    
    def __init__(self, base_url: str | None = None, **config):
        super().__init__(base_url or "http://localhost:8000", **config)
        
    def stream_chat_completion(self, model: str, prompt: str, **params) -> Response:
        # Implement streaming chat completion, return as a Response type.
        pass
        
    def list_models(self) -> list[str]:
        # Return available models
        pass
        
    def health(self) -> bool:
        # Check service health
        return True
```

Import in `clients/__init__.py`:

```python
from .custom import CustomClient

__all__ = [..., "CustomClient"]
```

### Adding a New Dataset

Create a dataset provider in `datasets/`:

```python
# datasets/custom.py
from ..core.registry import DatasetRegistry
from ..core.types import DatasetRecord
from .base import DatasetProvider

@DatasetRegistry.register("custom")
class CustomDataset(DatasetProvider):
    dataset_id = "custom"
    dataset_name = "Custom Dataset"
    
    def iter_records(self):
        # Yield DatasetRecord objects
        pass
        
    def size(self) -> int:
        # Return total record count
        return 0
```

Import in `datasets/__init__.py`:

```python
from . import custom

__all__ = [..., "custom"]
```

### Adding a New Analysis

Create an analyzer in `analysis/`:

```python
# analysis/custom.py
from ..core.registry import AnalysisRegistry
from .base import AnalysisContext, AnalysisProvider, AnalysisResult

@AnalysisRegistry.register("custom")
class CustomAnalysis(AnalysisProvider):
    analysis_id = "custom"
    
    def run(self, context: AnalysisContext) -> AnalysisResult:
        # Perform analysis on profiling data in context.results_dir
        # Return AnalysisResult with summary, data, and artifacts
        pass
```

Import in `analysis/__init__.py`:

```python
from . import custom
```

### Adding a New Visualization

Create a visualizer in `visualization/`:

```python
# visualization/custom.py
from ..core.registry import VisualizationRegistry
from .base import VisualizationContext, VisualizationProvider, VisualizationResult

@VisualizationRegistry.register("custom")
class CustomVisualization(VisualizationProvider):
    visualization_id = "custom"
    
    def render(self, context: VisualizationContext) -> VisualizationResult:
        # Generate plots using matplotlib, save to context.output_dir
        # Return VisualizationResult with artifact paths
        pass
```

Import in `visualization/__init__.py`:

```python
from . import custom
```
