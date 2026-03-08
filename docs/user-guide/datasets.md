# Datasets

IPW includes 10+ benchmark datasets spanning single-turn knowledge tests, multi-turn agentic tasks, and coding challenges.

## Available Datasets

| ID | Type | Size | Evaluation | Description |
|---|---|---|---|---|
| `ipw` | Single-turn | 1,000 | LLM judge | Mixed knowledge workload (bundled, no download) |
| `mmlu-pro` | Single-turn | ~12,000 | MCQ exact match | Professional-level multiple-choice |
| `supergpqa` | Single-turn | varies | MCQ exact match | Graduate-level academic questions |
| `gpqa` | Single-turn | varies | MCQ exact match | Graduate-level science questions |
| `math500` | Single-turn | 500 | Exact match | Mathematical problem solving |
| `natural-reasoning` | Single-turn | varies | LLM judge | Natural language reasoning |
| `wildchat` | Single-turn | varies | LLM judge | Real user conversations |
| `gaia` | Agentic | varies | LLM judge | Multi-step factual questions with file attachments |
| `simpleqa` | Agentic | varies | LLM judge | Short-form factual QA |
| `frames` | Agentic | varies | LLM judge | Multi-hop reasoning across Wikipedia articles |
| `hle` | Agentic | varies | LLM judge | Expert-level academic questions (Humanity's Last Exam) |
| `terminalbench` | Agentic | varies | Task-specific | Terminal/CLI task completion |
| `terminalbench-native` | Agentic | varies | Task-specific | TerminalBench with per-task Docker lifecycle |
| `swebench` | Coding | 50/500 | Test execution | Real GitHub issues requiring patches |
| `swefficiency` | Coding | varies | Speedup measurement | Code performance optimization |

## Single-Turn Datasets

**IPW Mixed 1K** (`ipw`) -- The default dataset. 1,000 curated prompts across diverse topics, bundled with the package (no download required). Used when `--dataset` is not specified.

**MMLU-Pro** (`mmlu-pro`) -- Enhanced MMLU with harder professional-level multiple-choice questions across 14+ academic subjects. Scored by letter-answer extraction (no LLM judge needed).

**SuperGPQA** (`supergpqa`) -- Graduate-level academic questions testing deep subject knowledge. MCQ format with exact-match scoring.

**GPQA** (`gpqa`) -- Graduate-level science questions (physics, chemistry, biology) curated by domain experts.

**MATH-500** (`math500`) -- 500 math problems covering algebra, geometry, number theory, and calculus. Scored by exact match after normalization.

**Natural Reasoning** (`natural-reasoning`) -- Tests logical deduction, common sense, and causal reasoning with open-ended responses.

**WildChat** (`wildchat`) -- Real user conversations covering a wide range of topics and interaction styles.

## Agentic Datasets

**GAIA** (`gaia`) -- General AI assistant benchmark with real-world questions spanning three difficulty levels. Many questions include file attachments (PDFs, images, spreadsheets) that agents must read. Files are cached at `~/.cache/gaia_benchmark/`.

**SimpleQA** (`simpleqa`) -- Short factual questions testing parametric knowledge (dates, names, numbers). Can also be used single-turn with `ipw profile`.

**FRAMES** (`frames`) -- Multi-hop factual retrieval requiring synthesis across 2-15 Wikipedia articles. Particularly suited for agents with web search or retrieval tools.

**HLE** (`hle`) -- Humanity's Last Exam from CAIS. Expert-level questions where state-of-the-art models typically score below 10%. By default loads text-only samples.

**TerminalBench** (`terminalbench`) -- Terminal/CLI tasks paired with the Terminus agent. Requires Docker for container management.

**TerminalBench Native** (`terminalbench-native`) -- Decoupled TerminalBench integration with per-task Docker lifecycle. Works with any agent (not just Terminus). Supports concurrent execution with `--concurrency`.

## Coding Datasets

**SWE-bench** (`swebench`) -- Real GitHub issues from popular Python repositories. The agent must produce a patch that fixes the issue and passes the test suite. Two variants: `verified` (500 tasks) and `verified_mini` (50 tasks, default).

**SWEfficiency** (`swefficiency`) -- Code performance optimization benchmark. Given a repository and workload, the agent must implement optimizations achieving a target speedup while keeping tests passing.

## Evaluation Methods

| Method | Datasets | Requires API Key |
|---|---|---|
| LLM judge | ipw, natural-reasoning, wildchat, gaia, simpleqa, frames, hle | Yes (`gpt-5-nano` default) |
| MCQ exact match | mmlu-pro, supergpqa, gpqa | No |
| Exact match | math500 | No |
| Task-specific | terminalbench, terminalbench-native, swebench, swefficiency | No |

LLM judge scoring requires `IPW_EVAL_API_KEY` or `OPENAI_API_KEY` in your environment.

## Usage

```bash
# Single-turn profiling
ipw profile --dataset mmlu-pro --client ollama --model llama3.2:1b \
  --client-base-url http://localhost:11434

# Agentic benchmarking
ipw run --agent react --model gpt-4o --dataset gaia --max-queries 10

# Limit sample count
ipw profile --dataset supergpqa --max-queries 50 ...

# List available datasets
ipw list datasets
```
