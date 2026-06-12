# Quickstart

## Verify Setup

After [installation](installation.md), confirm telemetry is working:

```bash
ipw --help
uv run scripts/test_energy_monitor.py
```

If the energy monitor reports readings, your platform's collector is working. If it fails, see [Platform Support](../benchmarking/platform-support.md).

## Single-Turn Profiling

Start your inference server, then profile:

```bash
ollama serve && ollama pull llama3.2:1b

ipw profile \
  --client ollama \
  --model llama3.2:1b \
  --client-base-url http://localhost:11434 \
  --max-queries 50
```

Analyze and plot:

```bash
ipw analyze ./runs/profile_*
ipw analyze ./runs/profile_* --analysis regression
ipw plot ./runs/profile_*
```

## Agentic Profiling

Install an agent extra and run a multi-turn workload:

```bash
uv pip install -e 'intelligence-per-watt[react]'
export OPENAI_API_KEY=sk-...

ipw run \
  --agent react \
  --model gpt-4o \
  --dataset gaia \
  --max-queries 10
```

Analyze and plot:

```bash
ipw analyze ./runs/run_*
ipw plot ./runs/run_*
```

## Listing Components

```bash
ipw list all              # Everything
ipw list clients          # ollama, vllm, openai
ipw list datasets         # ipw, mmlu-pro, supergpqa, gaia, frames, ...
ipw list analyses         # accuracy, regression
ipw list visualizations   # regression, output-kde
```

## Choosing a Dataset

```bash
ipw profile --dataset ipw ...            # General knowledge (built-in)
ipw profile --dataset mmlu-pro ...       # Multiple-choice academic
ipw run --dataset frames ...             # Multi-hop reasoning (agentic)
ipw run --agent terminus --dataset terminalbench ...  # Terminal tasks
ipw run --agent openhands --dataset swebench ...      # Software engineering
```

## Next Steps

- [Profiling Guide](../user-guide/profiling.md) -- all `ipw profile` options
- [Datasets Overview](../user-guide/datasets.md) -- benchmark details
- [Cost Tracking](../cost/pricing.md) -- cloud API cost estimation
- [Platform Support](../benchmarking/platform-support.md) -- hardware setup
