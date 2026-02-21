# Single-Turn Datasets

Single-turn datasets provide one prompt per query. The model generates a single response, which is scored by an LLM judge or exact match. These are used with `ipw profile`.

## IPW Mixed 1K

The default built-in dataset. A curated mix of 1,000 prompts across diverse topics, bundled with the package (no download required).

| Property | Value |
|----------|-------|
| Dataset ID | `ipw` |
| Source | Bundled in `ipw/datasets/ipw/data/` |
| Size | 1,000 |
| Evaluation | LLM judge |
| Download | Not required |

```bash
ipw profile --dataset ipw --client ollama --model llama3.2:1b \
  --client-base-url http://localhost:11434
```

This is the default dataset when `--dataset` is not specified.

## MMLU-Pro

Professional-level multiple-choice questions spanning academic subjects. An enhanced version of MMLU with harder questions and more answer choices.

| Property | Value |
|----------|-------|
| Dataset ID | `mmlu-pro` |
| Source | `TIGER-Lab/MMLU-Pro` on HuggingFace |
| Size | ~12,000 |
| Evaluation | MCQ exact match |
| Split | `test` (default) |

```bash
ipw profile --dataset mmlu-pro --client ollama --model llama3.2:1b \
  --client-base-url http://localhost:11434 --max-queries 100
```

**Prompt format**: Questions are presented with lettered options (A, B, C, ...) and the model is asked to respond with the correct letter.

**Scoring**: The evaluation handler (`ipw/evaluation/mmlu_pro.py`) extracts the letter answer from the model's response and compares it to the reference answer. No LLM judge is needed.

**Subjects**: biology, business, chemistry, computer science, economics, engineering, health, history, law, math, philosophy, physics, psychology, and more.

## SuperGPQA

Graduate-level academic questions designed to test deep subject knowledge.

| Property | Value |
|----------|-------|
| Dataset ID | `supergpqa` |
| Source | HuggingFace |
| Evaluation | MCQ exact match |
| Split | `test` (default) |

```bash
ipw profile --dataset supergpqa --client vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --client-base-url http://localhost:8000
```

## GPQA

Graduate-level science questions (physics, chemistry, biology) curated by domain experts.

| Property | Value |
|----------|-------|
| Dataset ID | `gpqa` |
| Source | HuggingFace |
| Evaluation | MCQ exact match |

## MATH-500

500 mathematical problems covering algebra, geometry, number theory, and calculus.

| Property | Value |
|----------|-------|
| Dataset ID | `math500` |
| Source | HuggingFace |
| Size | 500 |
| Evaluation | Exact match |

## Natural Reasoning

Natural language reasoning problems that test logical deduction, common sense, and causal reasoning.

| Property | Value |
|----------|-------|
| Dataset ID | `natural-reasoning` |
| Source | HuggingFace |
| Evaluation | LLM judge |

## WildChat

Real user conversations from the WildChat dataset, covering a wide range of topics and interaction styles.

| Property | Value |
|----------|-------|
| Dataset ID | `wildchat` |
| Source | HuggingFace |
| Evaluation | LLM judge |

## Evaluation Methods

### LLM Judge

For open-ended datasets (IPW, WildChat, Natural Reasoning), an LLM judge compares the model's response to the reference answer:

- Default judge: `gpt-5-nano-2025-08-07` via OpenAI API
- Requires `IPW_EVAL_API_KEY` or `OPENAI_API_KEY` in your environment
- The judge produces a binary correct/incorrect determination with metadata

### MCQ Exact Match

For multiple-choice datasets (MMLU-Pro, SuperGPQA, GPQA), the evaluation handler extracts the selected letter from the response and compares to the correct answer. No API key is needed for scoring.

### Exact Match

For datasets with precise answers (MATH-500), the response is compared directly to the reference answer after normalization.

## API Key Requirements

Datasets that use LLM judge scoring require an evaluation API key:

```bash
# Set in .env or environment
export IPW_EVAL_API_KEY=sk-...
# Or
export OPENAI_API_KEY=sk-...
```

MCQ datasets do not require an API key for scoring but still need one to run inference (unless using a local model).
