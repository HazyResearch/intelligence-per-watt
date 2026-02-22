<p align="center">
  <img src="assets/intelligence_per_watt_mood.png" width="500" alt="Intelligence Per Watt">
</p>

<p align="center">
  <b>Benchmarking intelligence efficiency for LLM inference systems.</b>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2511.07885"><img src="https://img.shields.io/badge/arXiv-2511.07885-b31b1b.svg" alt="arXiv"></a>
  <a href="https://www.intelligence-per-watt.ai/"><img src="https://img.shields.io/badge/project-intelligence--per--watt.ai-blue" alt="Project"></a>
  <a href="https://hazyresearch.stanford.edu/intelligence-per-watt/"><img src="https://img.shields.io/badge/docs-mkdocs-blue" alt="Docs"></a>
  <img src="https://img.shields.io/badge/python-%3E%3D3.12-blue" alt="Python">
  <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License">
</p>

----

OpenJarvis measures **accuracy alongside energy** for any LLM inference system. It profiles single-turn and multi-turn agentic workloads, captures per-query energy telemetry, and computes two efficiency metrics: **Intelligence Per Joule (IPJ)** and **Intelligence Per Watt (IPW)**.

> **[Documentation](https://hazyresearch.stanford.edu/intelligence-per-watt/)**
>
> **[Project Site](https://www.intelligence-per-watt.ai/)**

## Installation

```bash
pip install intelligence-per-watt
```

Or from source:

```bash
git clone https://github.com/HazyResearch/intelligence-per-watt.git
cd intelligence-per-watt
uv venv && source .venv/bin/activate
uv run scripts/build_energy_monitor.py    # Build Rust energy monitor
uv pip install -e intelligence-per-watt
```

Optional extras: `ollama`, `vllm`, `react`, `openhands`, `terminus`, `agents`, `tavily`, `flops`, `all`.

## Quick Start

**Profile an inference server:**

```bash
ipw profile --client ollama --model llama3.2:1b --client-base-url http://localhost:11434
```

**Run an agentic benchmark:**

```bash
ipw run --agent react --model gpt-4o --dataset gaia --max-queries 10
```

**Analyze and plot results:**

```bash
ipw analyze ./runs/profile_*
ipw plot ./runs/profile_*
```

Each query captures: energy (Joules), power (Watts), GPU/CPU memory, temperature, TTFT, throughput, token counts, API cost, and FLOPs.

## What's Included

**Inference clients** -- Ollama, vLLM (offline), OpenAI-compatible servers

**Agent harnesses** -- [ReAct](https://github.com/agno-agi/agno) (Agno), [OpenHands](https://github.com/All-Hands-AI/OpenHands), [Terminus](https://github.com/terminal-bench/terminal-bench)

**Benchmarks** -- MMLU-Pro, SuperGPQA, GAIA, FRAMES, HLE, SimpleQA, SWE-bench, SWEfficiency, TerminalBench, and a built-in 1K mixed set

**Energy telemetry** -- Rust gRPC service (50ms sampling) with NVIDIA NVML, AMD ROCm, Apple Silicon powermetrics, and Linux RAPL collectors

**Evaluation** -- LLM-as-judge, MCQ exact match, and task-specific scorers

## Architecture

```
ipw/
├── cli/          CLI commands (profile, run, analyze, plot, list)
├── clients/      Inference adapters (Ollama, vLLM, OpenAI)
├── agents/       Agent harnesses with per-turn telemetry
├── datasets/     Dataset providers (10+ benchmarks)
├── evaluation/   Scoring handlers
├── analysis/     IPJ/IPW computation, regression fitting
├── execution/    ProfilerRunner, AgenticRunner, TelemetrySession
└── telemetry/    Energy monitor launcher + gRPC collector

energy-monitor/   Rust gRPC service with platform-specific collectors
```

All components self-register via the **registry pattern** (`@ClientRegistry.register("id")`, etc.) and are resolved by string key through the CLI.

## About

OpenJarvis is part of [Intelligence Per Watt](https://www.intelligence-per-watt.ai/), a research initiative studying the efficiency of on-device AI systems. The project is developed at [Hazy Research](https://hazyresearch.stanford.edu/) and the [Scaling Intelligence Lab](https://scalingintelligence.stanford.edu/) at [Stanford SAIL](https://ai.stanford.edu/).

## Sponsors

<p>
  <a href="https://www.laude.org/">Laude Institute</a> &bull;
  <a href="https://datascience.stanford.edu/marlowe">Stanford Marlowe</a> &bull;
  <a href="https://cloud.google.com/">Google Cloud Platform</a> &bull;
  <a href="https://lambda.ai/">Lambda Labs</a>
</p>

## Citation

If you use Intelligence Per Watt in your research, please cite:

```bibtex
@misc{saadfalcon2025intelligencewattmeasuringintelligence,
      title={Intelligence per Watt: Measuring Intelligence Efficiency of Local AI},
      author={Jon Saad-Falcon and Avanika Narayan and Hakki Orhun Akengin and J. Wes Griffin and Herumb Shandilya and Adrian Gamarra Lafuente and Medhya Goel and Rebecca Joseph and Shlok Natarajan and Etash Kumar Guha and Shang Zhu and Ben Athiwaratkun and John Hennessy and Azalia Mirhoseini and Christopher Ré},
      year={2025},
      eprint={2511.07885},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2511.07885},
}
```
