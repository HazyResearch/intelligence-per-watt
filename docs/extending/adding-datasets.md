# Adding Datasets

Datasets provide the workload for profiling. To add a new dataset, subclass `DatasetProvider` and register it with `DatasetRegistry`.

## Step 1: Create the Dataset File

Create a new file in `intelligence-per-watt/src/ipw/datasets/`:

```python
# ipw/datasets/my_benchmark.py
from __future__ import annotations

import os
from typing import Dict, Iterable, List, MutableMapping, Optional, Tuple

from datasets import load_dataset

from ..clients.base import InferenceClient
from ..core.registry import ClientRegistry, DatasetRegistry, EvaluationRegistry
from ..core.types import DatasetRecord
from .base import DatasetProvider


@DatasetRegistry.register("my-benchmark")
class MyBenchmarkDataset(DatasetProvider):
    """My custom benchmark dataset."""

    dataset_id = "my-benchmark"
    dataset_name = "My Benchmark"
    evaluation_method = "my-benchmark"  # Maps to EvaluationRegistry

    # Evaluation settings
    eval_client = "openai"
    eval_base_url = "https://api.openai.com/v1"
    eval_model = "gpt-5-nano-2025-08-07"

    _hf_path = "my-org/my-benchmark"  # HuggingFace dataset path
    _default_split = "test"

    def __init__(
        self,
        *,
        split: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        self._split = split or self._default_split
        self._max_samples = max_samples
        self._records: Tuple[DatasetRecord, ...] = tuple(self._build_records())

    def iter_records(self) -> Iterable[DatasetRecord]:
        return iter(self._records)

    def size(self) -> int:
        return len(self._records)

    def verify_requirements(self) -> list[str]:
        issues: list[str] = []
        if not (os.getenv("IPW_EVAL_API_KEY") or os.getenv("OPENAI_API_KEY")):
            issues.append(
                "Missing evaluation API key. Set IPW_EVAL_API_KEY or OPENAI_API_KEY."
            )
        return issues

    def score(
        self,
        record: DatasetRecord,
        response: str,
        *,
        eval_client: Optional[InferenceClient] = None,
    ) -> Tuple[Optional[bool], Dict[str, object]]:
        handler = self._resolve_handler(eval_client)
        return handler.evaluate(
            problem=record.problem,
            reference=record.answer,
            model_answer=response,
            metadata=record.dataset_metadata,
        )

    def _resolve_handler(self, eval_client: Optional[InferenceClient]):
        judge_client = eval_client or ClientRegistry.create(
            self.eval_client or "openai",
            base_url=self.eval_base_url or "https://api.openai.com/v1",
            model=self.eval_model or "gpt-5-nano-2025-08-07",
        )
        return EvaluationRegistry.create(self.evaluation_method, client=judge_client)

    def _build_records(self) -> List[DatasetRecord]:
        dataset = load_dataset(self._hf_path, split=self._split)
        rows = list(dataset)
        if self._max_samples is not None:
            rows = rows[: self._max_samples]

        records: List[DatasetRecord] = []
        for raw in rows:
            record = self._convert_row(raw)
            if record is not None:
                records.append(record)
        return records

    def _convert_row(self, raw: dict) -> Optional[DatasetRecord]:
        question = str(raw.get("question", "")).strip()
        answer = str(raw.get("answer", "")).strip()

        if not question or not answer:
            return None

        return DatasetRecord(
            problem=question,
            answer=answer,
            subject=str(raw.get("category", "general")),
            dataset_metadata={
                "dataset_name": self.dataset_name,
                "id": raw.get("id"),
            },
        )
```

## Step 2: Register the Import

Add your module to `ipw/datasets/__init__.py`:

```python
def ensure_registered() -> None:
    from . import (
        # ... existing imports ...
        my_benchmark,  # Add this
    )
```

If your dataset has optional dependencies, wrap it in a try/except:

```python
try:
    from . import my_benchmark  # noqa: F401
except ImportError:
    pass
```

## Step 3: Create an Evaluation Handler (Optional)

If your dataset needs custom scoring logic, create an evaluation handler:

```python
# ipw/evaluation/my_benchmark.py
from ..core.registry import EvaluationRegistry
from .base import EvaluationHandler


@EvaluationRegistry.register("my-benchmark")
class MyBenchmarkEvaluator(EvaluationHandler):
    evaluation_method = "my-benchmark"

    def evaluate(self, *, problem, reference, model_answer, metadata):
        # Custom scoring logic
        is_correct = reference.lower().strip() == model_answer.lower().strip()
        return is_correct, {"method": "exact_match"}
```

Register it in `ipw/evaluation/__init__.py`.

## Step 4: Test

```bash
# Verify registration
ipw list datasets

# Run a profile
ipw profile --dataset my-benchmark --client ollama --model llama3.2:1b \
  --client-base-url http://localhost:11434 --max-queries 10
```

## Key Methods

### `iter_records()`

Must yield `DatasetRecord` objects. Each record contains:

- `problem` -- the prompt to send to the model
- `answer` -- the reference/ground-truth answer
- `subject` -- a category label (used for grouping in analysis)
- `dataset_metadata` -- arbitrary metadata (preserved in output)

### `size()`

Return the total number of records. Used for progress bars and validation.

### `score()`

Score a single model response. Returns `(is_correct, metadata)`:

- `is_correct`: `True`, `False`, or `None` (if unscorable)
- `metadata`: dictionary with evaluation details

For MCQ datasets, scoring can be done without an LLM judge (exact match on the selected option). For open-ended datasets, delegate to an `EvaluationHandler` that uses an LLM judge.

### `verify_requirements()`

Return a list of unmet requirements (e.g., missing API keys, missing packages). An empty list means the dataset is ready. Called before profiling starts.

## Prompt Formatting

Format prompts to get the best results from the model. Common patterns:

```python
# Multiple choice
prompt = f"{question}\n\nOptions:\n{formatted_options}\n\nRespond with the correct letter."

# Open-ended QA
prompt = f"Answer the following question concisely:\n\n{question}"

# With context
prompt = f"Given the following context:\n{context}\n\nAnswer: {question}"
```

## Managed Execution Environments

For datasets that need a managed environment per task (e.g., a Docker container), override `create_task_env()`:

```python
class MyDataset(DatasetProvider):
    def create_task_env(self, record):
        """Return a context manager for per-task environment setup."""
        return MyTaskEnv(record.dataset_metadata)
```

The returned context manager is used by the runner to wrap each agent call:

```python
with dataset.create_task_env(record):
    agent.set_task_metadata(record.dataset_metadata)
    agent.run(record.problem)
    env.run_tests()  # If the env supports post-task testing
```

The context manager should:

1. Set up the environment in `__enter__` (e.g., spin up a Docker container)
2. Populate `dataset_metadata` with environment handles (e.g., session, container name)
3. Clean up in `__exit__` (e.g., tear down the container)

See `ipw/datasets/terminalbench_native.py` and `ipw/execution/terminalbench_env.py` for a complete example.

## Existing Datasets for Reference

- `ipw/datasets/mmlu_pro.py` -- MCQ with option formatting
- `ipw/datasets/gaia.py` -- Agentic with file attachments and caching
- `ipw/datasets/swebench.py` -- Coding with variant support
- `ipw/datasets/simpleqa.py` -- Simple factual QA
- `ipw/datasets/terminalbench_native.py` -- Docker-managed tasks with test-based scoring
