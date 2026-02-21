# Quickstart

This guide walks through running your first profiling session and viewing the results.

## Single-Turn Profiling

Single-turn profiling sends prompts to an inference server one at a time, capturing energy telemetry for each query.

### 1. Start your inference server

```bash
# Using Ollama
ollama serve
ollama pull llama3.2:1b
```

### 2. Run a profiling session

```bash
ipw profile \
  --client ollama \
  --model llama3.2:1b \
  --client-base-url http://localhost:11434 \
  --max-queries 50
```

This will:

1. Launch the energy monitor subprocess
2. Connect to the Ollama server
3. Send 50 prompts from the default IPW dataset
4. Capture per-query telemetry (power, energy, memory, temperature, latency)
5. Score responses using an LLM judge
6. Compute IPJ and IPW metrics
7. Save results to `./runs/profile_<hardware>_<model>/`

### 3. Analyze the results

```bash
# Compute accuracy and efficiency metrics
ipw analyze ./runs/profile_*

# Fit regression curves for energy vs. input/output length
ipw analyze ./runs/profile_* --analysis regression
```

### 4. Generate plots

```bash
ipw plot ./runs/profile_*
```

Plots are saved to `./runs/profile_*/plots/`.

## Agentic Profiling

Agentic profiling runs multi-turn agent workloads where the model can use tools, reason over multiple steps, and interact with external systems.

### 1. Install an agent extra

```bash
uv pip install -e 'intelligence-per-watt[react]'
```

### 2. Set your API key

The ReAct agent calls a cloud LLM API:

```bash
export OPENAI_API_KEY=sk-...
```

### 3. Run an agentic profiling session

```bash
ipw run \
  --agent react \
  --model gpt-4o \
  --dataset gaia \
  --max-queries 10
```

This will:

1. Start the energy monitor
2. Initialize the ReAct agent with the specified model
3. Iterate over GAIA benchmark questions
4. For each question: run the agent, record per-turn traces, capture telemetry
5. Export results as JSONL traces and HuggingFace Arrow datasets
6. Run accuracy analysis

### 4. View results

```bash
# Analyze the agentic run
ipw analyze ./runs/run_*

# Plot the results
ipw plot ./runs/run_*
```

## Listing Available Components

Use `ipw list` to discover what is available:

```bash
# List everything
ipw list all

# List specific categories
ipw list clients       # ollama, vllm, openai
ipw list datasets      # ipw, mmlu-pro, supergpqa, gaia, frames, ...
ipw list analyses      # accuracy, regression
ipw list visualizations  # regression, output-kde
```

## Choosing a Dataset

Different datasets test different capabilities:

```bash
# General knowledge (built-in, no download needed)
ipw profile --dataset ipw ...

# Multiple-choice academic knowledge
ipw profile --dataset mmlu-pro ...

# Multi-hop reasoning (agentic)
ipw run --dataset frames ...

# Terminal/CLI tasks (requires Terminus agent)
ipw run --agent terminus --dataset terminalbench ...

# Software engineering (requires agent)
ipw run --agent openhands --dataset swebench ...
```

## Next Steps

- Read the [Profiling Guide](../user-guide/profiling.md) for all `ipw profile` options
- Explore the [Datasets Overview](../datasets/overview.md) for benchmark details
- Set up [Cost Tracking](../cost/pricing.md) for cloud API models
- Check [Platform Support](../telemetry/platform-support.md) for your hardware
