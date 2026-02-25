# ReAct Agent

The ReAct agent uses the [Agno](https://github.com/agno-agi/agno) framework to implement Reasoning + Acting (ReAct) style tool-augmented reasoning with energy telemetry.

## Installation

```bash
uv pip install -e 'intelligence-per-watt[react]'
```

This installs the `agno` package as a dependency.

## How It Works

The ReAct agent (`ipw/agents/react.py`) wraps Agno's `Agent` class and instruments tool calls for energy tracking:

1. **Initialization**: Creates an Agno `Agent` with the specified model, tools, and instructions. If an `EventRecorder` is provided, all tools are wrapped with telemetry instrumentation.

2. **Tool instrumentation**: Each tool function is wrapped with `_instrument_tools()`, which emits `tool_call_start` and `tool_call_end` events around every invocation:

    ```python
    def wrapper(*args, __tool=tool, __name=tool_name, **kwargs):
        self._record_event("tool_call_start", tool=__name)
        try:
            return __tool(*args, **kwargs)
        finally:
            self._record_event("tool_call_end", tool=__name)
    ```

3. **Execution**: `agent.run(input)` calls the Agno agent, which performs ReAct-style reasoning loops -- thinking, selecting tools, executing tools, and synthesizing a response.

4. **Result extraction**: Token metrics are extracted from Agno's response metrics (if available) and returned in an `AgentRunResult`.

## Usage

### Via CLI

```bash
ipw run \
  --agent react \
  --model gpt-4o \
  --dataset gaia \
  --max-queries 10
```

### Programmatic Usage

```python
from agno.models.openai import OpenAIChat
from ipw.agents.react import React
from ipw.telemetry.events import EventRecorder

# Create the model
model = OpenAIChat(id="gpt-4o")

# Define tools
def web_search(query: str) -> str:
    """Search the web for information."""
    # Implementation here
    return "search results..."

# Create the agent with telemetry
recorder = EventRecorder()
agent = React(
    model=model,
    tools=[web_search],
    event_recorder=recorder,
)

# Run the agent
result = agent.run("What is the population of Tokyo?")
print(result.content)
print(f"Input tokens: {result.input_tokens}")
print(f"Output tokens: {result.output_tokens}")

# Inspect telemetry events
for event in recorder.get_events():
    print(f"{event.event_type}: {event.metadata}")
```

## Configuration

The `React` constructor accepts:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | Any | required | Agno Model instance (e.g., `OpenAIChat`) |
| `tools` | list[Callable] | required | List of callable tool functions |
| `instructions` | str | built-in | Custom system instructions |
| `event_recorder` | EventRecorder | None | Telemetry recorder |
| `**kwargs` | Any | -- | Passed to Agno `Agent()` constructor |

## Default Instructions

If no custom instructions are provided, the agent uses:

> You are a helpful assistant that can answer questions and use the tools provided to you if necessary.

## Return Value

`React.run()` returns an `AgentRunResult`:

```python
AgentRunResult(
    content="Tokyo's population is approximately 13.96 million...",
    input_tokens=150,
    output_tokens=45,
)
```

Token counts are extracted from Agno's `result.metrics` object when available.

## Error Handling

If the Agno agent raises an exception during execution, the `lm_inference_end` event is still emitted (via the `finally` block), ensuring telemetry windows are properly closed.

If `agno` is not installed, importing or instantiating the `React` class raises an `ImportError` with installation instructions.
