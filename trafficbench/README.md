# trafficbench

Python package for profiling and analyzing inference service performance across hardware configurations.

## Project Structure

```
src/trafficbench/
├── analysis/          Analysis implementations (regression)
├── cli/               Command-line interface (profile, analyze, plot, list, energy)
├── clients/           Inference service clients (ollama)
├── core/              Component registration and shared types
├── datasets/          Dataset providers (built-in TrafficBench dataset)
├── execution/         Profiling orchestration and telemetry collection
├── telemetry/         Energy monitoring integration (gRPC client, launcher)
├── tests/             Test suite
└── visualization/     Plotting and visualization (KDE, regression plots)
```

## Installation

Development installation:

```bash
cd trafficbench
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
pytest src/trafficbench/tests/clients/
pytest src/trafficbench/tests/analysis/test_regression.py
```

## Extending TrafficBench

Add matching tests under `src/trafficbench/tests/` whenever you add a client, dataset, analysis, or visualization so automated runs cover the new component.

### Registry System

TrafficBench uses a centralized registry (`core/registry.py`) for component discovery. All clients, datasets, analyses, and visualizations register themselves on import, making them available via CLI commands. Components can be listed via:

```bash
trafficbench list all
```

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
