# Coding Datasets

Coding datasets evaluate code generation, bug fixing, and performance optimization capabilities. These are designed for use with agent harnesses that can interact with code repositories.

## SWE-bench

Real GitHub issues from popular Python repositories, requiring the model to produce a patch that fixes the issue and passes the test suite.

| Property | Value |
|----------|-------|
| Dataset ID | `swebench` |
| Source | `princeton-nlp/SWE-bench_Verified` |
| Evaluation | Test execution |
| Requirements | Agent with code editing capabilities |

### Variants

SWE-bench ships with two variants:

| Variant | Source | Size | Description |
|---------|--------|------|-------------|
| `verified` | `princeton-nlp/SWE-bench_Verified` | 500 | Full verified dataset |
| `verified_mini` | `MariusHobbhahn/swe-bench-verified-mini` | 50 | Smaller subset for quick testing |

The default variant is `verified_mini`.

### Usage

```bash
# Use the mini subset (default)
ipw run --agent openhands --model gpt-4o --dataset swebench --max-queries 10

# Full verified set (slow)
ipw run --agent openhands --model gpt-4o --dataset swebench
```

### How It Works

1. Each record contains a `problem_statement` (the GitHub issue text) and metadata about the repository.
2. The agent reads the issue, explores the codebase, and generates a fix.
3. The `answer` field contains the ground-truth patch for reference.
4. Correctness is determined by running the repository's test suite against the agent's patch.

### Record Metadata

Each `DatasetRecord.dataset_metadata` includes:

| Field | Description |
|-------|-------------|
| `instance_id` | Unique SWE-bench instance identifier |
| `repo` | GitHub repository (e.g., `django/django`) |
| `base_commit` | Commit hash for the issue |
| `hints_text` | Optional hints from the issue |
| `version` | Repository version |
| `test_patch` | Patch for the test that validates the fix |
| `fail_to_pass` | Tests that should go from failing to passing |
| `pass_to_pass` | Tests that should continue passing |
| `difficulty` | Estimated difficulty level |
| `variant` | Which SWE-bench variant this came from |

### Evaluation

SWE-bench uses test execution rather than LLM judging. The evaluation checks whether the agent's patch causes the `fail_to_pass` tests to pass while keeping `pass_to_pass` tests green. This means `eval_client`, `eval_base_url`, and `eval_model` are set to `None`.

## SWEfficiency

Code performance optimization benchmark. Given a repository and a workload, the goal is to produce optimizations that achieve a target speedup.

| Property | Value |
|----------|-------|
| Dataset ID | `swefficiency` |
| Source | HuggingFace |
| Evaluation | Speedup measurement |
| Requirements | Agent with code editing capabilities |

### Usage

```bash
ipw run --agent openhands --model gpt-4o --dataset swefficiency --max-queries 5
```

### How It Works

1. Each record describes a repository, a performance problem, a workload to benchmark, and an expected speedup target.
2. The agent analyzes the codebase, identifies bottlenecks, and implements optimizations.
3. The optimization is provided as a unified diff patch.
4. Correctness is measured by whether the optimization achieves the target speedup while keeping tests passing.

### Prompt Format

```
You are a software performance engineer. Your task is to optimize the code
in the repository to improve performance.

Repository: django/django

## Problem Statement
[Description of the performance issue]

## Workload Description
[How to benchmark the change]

## Expected Speedup
The optimization should achieve approximately 2.5x speedup.

## Instructions
1. Analyze the codebase to identify performance bottlenecks
2. Implement optimizations that improve performance
3. Ensure all existing tests still pass
4. Generate a git patch with your changes
```

### Record Metadata

| Field | Description |
|-------|-------------|
| `instance_id` | Unique identifier |
| `repo` | Repository name |
| `workload` | Workload description for benchmarking |
| `expected_speedup` | Target speedup multiplier |
| `base_commit` | Starting commit hash |
| `test_patch` | Test validation patch |

## Choosing Between Coding Datasets

| Need | Dataset | Why |
|------|---------|-----|
| Bug fixing | `swebench` (verified_mini) | Real bugs, test-validated, fast subset |
| Full evaluation | `swebench` (verified) | Comprehensive 500-task benchmark |
| Performance optimization | `swefficiency` | Tests optimization skill, not just correctness |
