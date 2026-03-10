---
title: Intelligence Per Watt
description: Benchmarking Intelligence Efficiency of LM Inference
hide:
  - navigation
---

# *Intelligence* Per Watt

<p class="hero-tagline">Benchmarking Intelligence Efficiency of LM Inference.</p>

<p class="install-command">pip install intelligence-per-watt</p>

[Get Started](getting-started/installation.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/HazyResearch/intelligence-per-watt){ .md-button }

---

<div class="grid cards" markdown>

-   **Profile**

    ---

    Single-turn and agentic inference profiling with per-query telemetry across any OpenAI-compatible endpoint.

    [:octicons-arrow-right-24: Profiling guide](user-guide/profiling.md)

-   **Measure**

    ---

    Real-time energy, power, temperature, and memory telemetry via a Rust gRPC service sampling at 50ms.

    [:octicons-arrow-right-24: Benchmarking overview](benchmarking/overview.md)

-   **Analyze**

    ---

    Intelligence Per Joule and Intelligence Per Watt metrics with accuracy scoring, regression analysis, and plots.

    [:octicons-arrow-right-24: Analysis guide](user-guide/analysis.md)

-   **Extend**

    ---

    Plug in custom inference clients, benchmark datasets, agent harnesses, and platform collectors.

    [:octicons-arrow-right-24: Extending IPW](extending/index.md)

</div>

---

## Key Metrics

- **Intelligence Per Joule (IPJ)** = accuracy / average energy per query (joules)
- **Intelligence Per Watt (IPW)** = accuracy / average power per query (watts)

## What's Included

| Component | Options |
|---|---|
| **Clients** | Ollama, vLLM, OpenAI-compatible (OpenAI, OpenRouter, Gemini, local servers) |
| **Agents** | ReAct (Agno), OpenHands, Terminus |
| **Datasets** | MMLU-Pro, GPQA, SuperGPQA, MATH-500, GAIA, SimpleQA, FRAMES, HLE, TerminalBench, SWE-bench, SWEfficiency |
| **Telemetry** | NVIDIA (NVML), AMD (ROCm), Apple Silicon (powermetrics), Linux (RAPL) |
| **Evaluation** | LLM judge, MCQ exact match, task-specific scoring |

## About

Built by [Stanford Hazy Research](https://hazyresearch.stanford.edu/) and the [Scaling Intelligence Lab](https://scalingintelligence.stanford.edu/).

Paper: [arXiv:2511.07885](https://arxiv.org/abs/2511.07885)

## Acknowledgements

[Stanford HAI](https://hai.stanford.edu/) &bull;
[IBM Research](https://research.ibm.com/) &bull;
[Ollama](https://ollama.com/)
