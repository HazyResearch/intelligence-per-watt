# Cost Tracking

IPW tracks API costs for cloud-hosted models automatically. Pricing tables are maintained in `ipw/cost/pricing.py` and cover OpenAI, Anthropic, and Google Gemini.

## How Cost Tracking Works

1. During profiling, token counts (input and output) are captured from each inference response.
2. The `calculate_cost()` function looks up the model in the provider's pricing table.
3. Cost is computed as: `(input_tokens / 1M) * input_price + (output_tokens / 1M) * output_price`.
4. Per-query cost is stored in `CostMetrics` within each `ProfilingRecord`.
5. For agentic runs, per-turn cost is recorded in `TurnTrace.cost_usd`.

## Pricing Tables

All prices are per 1 million tokens in USD.

### [OpenAI](https://openai.com/api/pricing/)

| Model | Input ($/1M) | Output ($/1M) |
|-------|-------------|---------------|
| `gpt-4o` | $2.50 | $10.00 |
| `gpt-4o-mini` | $0.15 | $0.60 |
| `gpt-4` | $30.00 | $60.00 |
| `gpt-4-turbo` | $10.00 | $30.00 |
| `gpt-3.5-turbo` | $0.50 | $1.50 |
| `o1` | $15.00 | $60.00 |
| `o1-mini` | $3.00 | $12.00 |
| `gpt-5.2-2025-12-11` | $30.00 | $120.00 |
| `gpt-5-mini-2025-08-07` | $5.00 | $20.00 |
| `gpt-5-nano-2025-08-07` | $1.00 | $4.00 |

### [Anthropic](https://www.anthropic.com/pricing)

| Model | Input ($/1M) | Output ($/1M) |
|-------|-------------|---------------|
| `claude-opus-4-5-20251101` | $20.00 | $100.00 |
| `claude-sonnet-4-5-20250929` | $4.00 | $20.00 |
| `claude-haiku-4-5-20251001` | $1.00 | $5.00 |
| `claude-opus-4-20250514` | $15.00 | $75.00 |
| `claude-sonnet-4-20250514` | $3.00 | $15.00 |
| `claude-3-5-sonnet-20241022` | $3.00 | $15.00 |
| `claude-3-5-haiku-20241022` | $0.80 | $4.00 |
| `claude-3-opus-20240229` | $15.00 | $75.00 |
| `claude-3-sonnet-20240229` | $3.00 | $15.00 |
| `claude-3-haiku-20240307` | $0.25 | $1.25 |

### [Google Gemini](https://ai.google.dev/pricing)

| Model | Input ($/1M) | Output ($/1M) |
|-------|-------------|---------------|
| `gemini-3-flash-preview` | $0.10 | $0.40 |
| `gemini-2.0-flash` | $0.10 | $0.40 |
| `gemini-2.0-flash-lite` | $0.075 | $0.30 |
| `gemini-1.5-pro` | $1.25 | $5.00 |
| `gemini-1.5-flash` | $0.075 | $0.30 |
| `gemini-1.5-flash-8b` | $0.0375 | $0.15 |

### Tool Costs

| Tool | Cost | Unit |
|------|------|------|
| Tavily web search | $0.01 | per search |

## Usage

### Programmatic

```python
from ipw.cost.pricing import calculate_cost

cost = calculate_cost(
    provider="openai",
    model="gpt-4o",
    input_tokens=1500,
    output_tokens=500,
)
print(f"Cost: ${cost:.6f}")  # Cost: $0.008750
```

### In Profiling Results

Cost metrics are stored in each `ProfilingRecord`:

```python
from datasets import load_from_disk

dataset = load_from_disk("./runs/profile_nvidia_gpt4o_gaia/")
for row in dataset:
    cost = row["model_metrics"]["gpt-4o"]["cost"]
    print(f"Input cost: ${cost['input_cost_usd']:.6f}")
    print(f"Output cost: ${cost['output_cost_usd']:.6f}")
    print(f"Total cost: ${cost['total_cost_usd']:.6f}")
```

### In Agentic Traces

Per-turn cost is available in `TurnTrace`:

```python
from ipw.execution.trace import QueryTrace

traces = QueryTrace.load_jsonl(Path("traces.jsonl"))
for trace in traces:
    for turn in trace.turns:
        if turn.cost_usd:
            print(f"Turn {turn.turn_index}: ${turn.cost_usd:.6f}")
    print(f"Total: ${trace.total_cost_usd:.6f}")
```

## Unknown Models

If a model is not in the pricing tables, `calculate_cost()` returns `0.0`. This is common for:

- Local models (Ollama, vLLM) -- they have no API cost
- Fine-tuned models with custom pricing
- New models not yet added to the tables

## Adding New Models

To add pricing for a new model, update the appropriate dictionary in `ipw/cost/pricing.py`:

```python
OPENAI_PRICING["gpt-6-mini"] = {"input": 2.00, "output": 8.00}
```

## Cost vs Energy

IPW tracks both financial cost (API pricing) and energy cost (joules/watts):

| Metric | What it measures | When relevant |
|--------|------------------|---------------|
| `cost_usd` | Cloud API spending | Cloud-hosted models |
| `energy_joules` | Physical energy consumption | Self-hosted models |
| IPJ/IPW | Accuracy per unit energy | Always (when energy data available) |

For cloud models, you get both cost and energy data. For local models, you get energy data with zero cost.
