# Agentic Datasets

Agentic datasets require multi-turn reasoning, tool use, or retrieval. They are designed for use with `ipw run` and an agent harness.

## GAIA

The GAIA benchmark tests general AI assistant capabilities with real-world questions that often require file reading, web search, or multi-step reasoning.

| Property | Value |
|----------|-------|
| Dataset ID | `gaia` |
| Source | `gaia-benchmark/GAIA` on HuggingFace |
| Split | `validation` (default), subset `2023_all` |
| Evaluation | LLM judge (`gaia` handler) |
| Requirements | `IPW_EVAL_API_KEY` or `OPENAI_API_KEY` |

### Features

- Questions span three difficulty levels (Level 1, 2, 3)
- Many questions include attached files (PDFs, images, spreadsheets) that the agent must read
- File artifacts are downloaded and cached at `~/.cache/gaia_benchmark/`
- Prompts include file paths so agents with file-reading tools can access them

### Usage

```bash
ipw run --agent react --model gpt-4o --dataset gaia --max-queries 10
```

### Prompt Format

Questions are formatted with file context when applicable:

```
Please answer the question below. You should:
- Return only your answer...

The following file is referenced in the question below...
File name: document.pdf
File path: /home/user/.cache/gaia_benchmark/GAIA/2023/validation/document.pdf
Use the file reading tools to access this file.

Here is the question:
What is the total revenue mentioned in the attached report?
```

### Metadata

Each record's `dataset_metadata` includes:

- `task_id` -- unique GAIA task identifier
- `level` -- difficulty level (1, 2, or 3)
- `file_name` -- name of the attached file (if any)
- `file_path` -- local path to the downloaded file

## SimpleQA

Short-form factual QA from OpenAI's SimpleQA benchmark, testing parametric knowledge.

| Property | Value |
|----------|-------|
| Dataset ID | `simpleqa` |
| Source | `basicv8vc/SimpleQA` on HuggingFace |
| Split | `test` (default) |
| Evaluation | LLM judge (`simpleqa` handler) |
| Requirements | `IPW_EVAL_API_KEY` or `OPENAI_API_KEY` |

### Features

- Short factual questions with concise reference answers
- Tests whether models can recall specific facts (dates, names, numbers)
- Can be used for both single-turn and agentic profiling

### Usage

```bash
# Single-turn
ipw profile --dataset simpleqa --client ollama --model llama3.2:1b \
  --client-base-url http://localhost:11434

# Agentic (with retrieval tools)
ipw run --agent react --model gpt-4o --dataset simpleqa --max-queries 50
```

### Prompt Format

```
Please answer the question below with a short, factual response.
- Return only your answer, which should be a word, phrase, name, number, or date.
- If the answer is a number, return only the number without any units unless specified.

Question: Who was the first person to walk on the Moon?
```

## FRAMES

Multi-hop factual retrieval from Google's FRAMES benchmark. Questions require synthesizing information across 2-15 Wikipedia articles.

| Property | Value |
|----------|-------|
| Dataset ID | `frames` |
| Source | `google/frames-benchmark` on HuggingFace |
| Split | `test` (default) |
| Evaluation | LLM judge (`frames` handler) |
| Requirements | `IPW_EVAL_API_KEY` or `OPENAI_API_KEY` |

### Features

- Questions require multi-hop reasoning across multiple Wikipedia sources
- Some records include relevant Wikipedia context in the prompt
- Tests retrieval-augmented generation (RAG) capabilities
- Particularly suitable for agents with web search or retrieval tools

### Usage

```bash
ipw run --agent react --model gpt-4o --dataset frames --max-queries 20
```

### Prompt Format

```
Please answer the question below. You should:
- Return only your answer...
- This question may require multi-hop reasoning across multiple Wikipedia articles.

[Optional Wikipedia context]

Here is the question:
Which country has a higher population: the birthplace of Einstein or the birthplace of Newton?
```

## HLE (Humanity's Last Exam)

Expert-level academic questions from the Center for AI Safety (CAIS). Designed to be the hardest publicly available benchmark.

| Property | Value |
|----------|-------|
| Dataset ID | `hle` |
| Source | `cais/hle` on HuggingFace |
| Split | `test` (default) |
| Evaluation | LLM judge (`hle` handler) |
| Requirements | `IPW_EVAL_API_KEY` or `OPENAI_API_KEY` |

### Features

- Expert-level questions across many academic disciplines
- By default, only text-only samples are loaded (set `text_only=False` for multimodal items)
- State-of-the-art models typically score below 10% on this benchmark
- Useful for measuring the ceiling of model capabilities

### Usage

```bash
ipw run --agent openhands --model gpt-4o --dataset hle --max-queries 10
```

### Configuration

The `HLEDataset` constructor accepts:

- `split` -- HuggingFace split (default: `test`)
- `max_samples` -- limit number of samples
- `text_only` -- if `True` (default), exclude samples with images

## TerminalBench

Terminal/CLI task completion benchmark from the terminal-bench project.

| Property | Value |
|----------|-------|
| Dataset ID | `terminalbench` |
| Source | `terminal-bench/terminal-bench` on HuggingFace |
| Split | `test` (default) |
| Evaluation | Terminal output check (`terminalbench` handler) |
| Requirements | `terminal-bench` package (`ipw[terminus]`) |

### Features

- Tasks that must be solved by executing terminal commands
- Designed to pair with the Terminus agent and Docker containers
- Evaluation checks the terminal state after task execution

### Usage

```bash
ipw run --agent terminus --model gpt-4o --dataset terminalbench --max-queries 10
```

This requires Docker to be available for the Terminus agent's container management.
