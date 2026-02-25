# Datasets Overview

IPW ships with 10+ benchmark datasets spanning single-turn knowledge tests, multi-turn agentic tasks, and coding challenges. All datasets implement the `DatasetProvider` ABC and are registered via `@DatasetRegistry.register("id")`.

## Dataset Categories

### Single-Turn Datasets

These datasets send one prompt per query and evaluate the response directly. Best used with `ipw profile`.

| Dataset ID | Name | Size | Evaluation | Description |
|------------|------|------|------------|-------------|
| `ipw` | IPW Mixed 1K | 1,000 | LLM judge | Built-in mixed workload (bundled, no download) |
| `mmlu-pro` | MMLU-Pro | ~12,000 | MCQ exact match | Professional-level multiple-choice knowledge |
| `supergpqa` | SuperGPQA | varies | MCQ exact match | Graduate-level academic questions |
| `gpqa` | GPQA | varies | MCQ exact match | Graduate-level science questions |
| `math500` | MATH-500 | 500 | Exact match | Mathematical problem solving |
| `natural-reasoning` | Natural Reasoning | varies | LLM judge | Natural language reasoning |
| `wildchat` | WildChat | varies | LLM judge | Real user conversations |

### Agentic Datasets

These datasets require multi-turn reasoning, tool use, or external knowledge retrieval. Best used with `ipw run`.

| Dataset ID | Name | Source | Evaluation | Description |
|------------|------|--------|------------|-------------|
| `gaia` | GAIA | `gaia-benchmark/GAIA` | LLM judge | Multi-step factual questions with file attachments |
| `simpleqa` | SimpleQA | `basicv8vc/SimpleQA` | LLM judge | Short-form factual QA |
| `frames` | FRAMES | `google/frames-benchmark` | LLM judge | Multi-hop reasoning across Wikipedia articles |
| `hle` | HLE | `cais/hle` | LLM judge | Expert-level academic questions (Humanity's Last Exam) |
| `terminalbench` | TerminalBench | `terminal-bench/terminal-bench` | Terminal output check | Terminal/CLI task completion |

### Coding Datasets

These datasets evaluate code generation and software engineering capabilities.

| Dataset ID | Name | Source | Evaluation | Description |
|------------|------|--------|------------|-------------|
| `swebench` | SWE-bench | `princeton-nlp/SWE-bench_Verified` | Test execution | Real GitHub issues requiring patches |
| `swefficiency` | SWEfficiency | HuggingFace | Speedup measurement | Code performance optimization |

## Common Interface

All datasets implement `DatasetProvider` (`ipw/datasets/base.py`):

```python
class DatasetProvider(ABC):
    dataset_id: str
    dataset_name: str

    # Preferred evaluation settings
    eval_client: str | None = "openai"
    eval_base_url: str | None = "https://api.openai.com/v1"
    eval_model: str | None = "gpt-5-nano-2025-08-07"

    @abstractmethod
    def iter_records(self) -> Iterable[DatasetRecord]:
        """Yield dataset records."""

    @abstractmethod
    def size(self) -> int:
        """Return the number of records."""

    def score(self, record, response, *, eval_client=None):
        """Compute correctness for a single response."""

    def verify_requirements(self) -> list[str]:
        """Return unmet requirements (e.g., missing API keys)."""
```

## DatasetRecord

Each record yielded by a dataset contains:

```python
@dataclass(slots=True)
class DatasetRecord:
    problem: str                              # Input prompt
    answer: str                               # Reference answer
    subject: str                              # Subject/category
    dataset_metadata: MutableMapping[str, Any] # Dataset-specific fields
```

The `dataset_metadata` field carries dataset-specific information such as task IDs, difficulty levels, file paths (GAIA), option lists (MMLU-Pro), or repository information (SWE-bench).

## Listing Available Datasets

```bash
ipw list datasets
```

## Using a Dataset

### With `ipw profile` (single-turn)

```bash
ipw profile --dataset mmlu-pro --client ollama --model llama3.2:1b \
  --client-base-url http://localhost:11434
```

### With `ipw run` (agentic)

```bash
ipw run --agent react --model gpt-4o --dataset gaia --max-queries 10
```

### Limiting Samples

Most datasets accept a `max_samples` parameter during construction. Via the CLI, use `--max-queries`:

```bash
ipw profile --dataset supergpqa --max-queries 50 ...
```
