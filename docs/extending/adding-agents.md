# Adding Agents

Agent harnesses wrap existing agent frameworks and add energy telemetry instrumentation. To add a new agent, subclass `BaseAgent` and register it with `AgentRegistry`.

## Step 1: Create the Agent File

Create a new file in `intelligence-per-watt/src/ipw/agents/`:

```python
# ipw/agents/my_agent.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from ipw.agents.base import BaseAgent
from ipw.core.registry import AgentRegistry
from ipw.core.types import AgentRunResult

if TYPE_CHECKING:
    from ipw.telemetry.events import EventRecorder


@AgentRegistry.register("my-agent")
class MyAgent(BaseAgent):
    """Custom agent with energy telemetry."""

    def __init__(
        self,
        model: str,
        event_recorder: Optional["EventRecorder"] = None,
        max_turns: int = 20,
        **kwargs: Any,
    ) -> None:
        super().__init__(event_recorder=event_recorder)

        # Lazy import for optional dependency
        try:
            from my_framework import Agent as FrameworkAgent
        except ImportError:
            raise ImportError(
                "my-framework is required. Install with: pip install my-framework"
            )

        self.model = model
        self.max_turns = max_turns
        self._agent = FrameworkAgent(model=model, **kwargs)
        self._tool_names: list[str] = []
        self._num_turns: int = 0

    def run(self, input: str, **kwargs: Any) -> AgentRunResult:
        """Run the agent with telemetry instrumentation."""
        # Reset per-run tracking
        self._tool_names = []
        self._num_turns = 0

        # Record the start of inference
        self._record_event("lm_inference_start", model=self.model)
        try:
            # Run your agent framework
            for step in self._agent.iterate(input, max_turns=self.max_turns):
                self._num_turns += 1

                if step.is_tool_call:
                    tool_name = step.tool_name
                    self._tool_names.append(tool_name)

                    # Record tool call boundaries
                    self._record_event("tool_call_start", tool=tool_name)
                    step.execute()
                    self._record_event("tool_call_end", tool=tool_name)

            # Extract the final response
            content = self._agent.get_final_response()

            return AgentRunResult(
                content=content,
                tool_calls_attempted=len(self._tool_names),
                tool_calls_succeeded=len(self._tool_names),
                tool_names_used=list(self._tool_names),
                num_turns=self._num_turns,
            )
        finally:
            # Always record inference end (even on error)
            self._record_event("lm_inference_end", model=self.model)
```

## Step 2: Register the Import

Add your agent module to `ipw/agents/__init__.py`:

```python
def ensure_registered() -> None:
    from . import (
        react,
        openhands,
        terminus,
        my_agent,  # Add this
    )
```

## Step 3: Add Optional Dependency

Add the framework to `pyproject.toml`:

```toml
[project.optional-dependencies]
my-agent = ["my-framework>=1.0"]
agents = ["agno", "terminal-bench", "docker", "openhands-sdk", "my-framework"]
```

## Step 4: Test

```bash
# Install the extra
uv pip install -e 'intelligence-per-watt[my-agent]'

# Run with ipw
ipw run --agent my-agent --model gpt-4o --dataset gaia --max-queries 5
```

## Telemetry Instrumentation

The key to energy attribution is recording events at the right boundaries:

### LLM Calls

Wrap LLM inference calls with start/end events:

```python
self._record_event("lm_inference_start", model=self.model)
try:
    result = self._call_llm(prompt)
finally:
    self._record_event("lm_inference_end", model=self.model)
```

### Tool Calls

Wrap each tool invocation:

```python
self._record_event("tool_call_start", tool=tool_name)
try:
    tool_result = tool.execute(args)
finally:
    self._record_event("tool_call_end", tool=tool_name)
```

### Token Metadata

Include token counts in the end event when available:

```python
self._record_event(
    "lm_inference_end",
    model=self.model,
    prompt_tokens=usage.input_tokens,
    completion_tokens=usage.output_tokens,
    total_tokens=usage.total_tokens,
)
```

## MCP Tool Support

To support MCP tools, accept a `mcp_tools` parameter and pass it to `BaseAgent.__init__()`:

```python
def __init__(
    self,
    model: str,
    mcp_tools: dict[str, BaseMCPServer] | None = None,
    event_recorder: EventRecorder | None = None,
    **kwargs,
):
    super().__init__(mcp_tools=mcp_tools, event_recorder=event_recorder)

    # Convert MCP tools to your framework's tool format
    framework_tools = []
    for name, server in self.mcp_tools.items():
        framework_tools.append(self._wrap_mcp_tool(name, server))
    self._agent = FrameworkAgent(model=model, tools=framework_tools)
```

## AgentRunResult

Return an `AgentRunResult` with as much information as available:

```python
AgentRunResult(
    content="The answer is...",              # Required: final response text
    tool_calls_attempted=5,                  # Number of tool calls made
    tool_calls_succeeded=4,                  # Number that succeeded
    tool_names_used=["search", "calculate"], # Names of tools used
    num_turns=3,                             # Number of agent turns
    input_tokens=500,                        # Total input tokens
    output_tokens=200,                       # Total output tokens
    cost_usd=0.05,                          # API cost
)
```

## Error Handling

- Always emit `lm_inference_end` in a `finally` block to ensure telemetry windows are closed.
- Use lazy imports so missing optional dependencies produce clear error messages.
- If the agent fails mid-execution, return whatever partial content is available.

## Per-Task Metadata

For agents that need access to task-level environment information (e.g., a tmux session for TerminalBench), override `set_task_metadata()`:

```python
class MyAgent(BaseAgent):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self._task_metadata = None

    def set_task_metadata(self, metadata):
        self._task_metadata = metadata

    def run(self, input, **kwargs):
        session = self._task_metadata.get("session")
        # Use the session to interact with the environment
        ...
```

The runner calls `set_task_metadata()` inside the task environment context (after Docker is ready), so metadata fields like `session`, `terminal`, and `container` are populated.

See `ipw/agents/terminus_tb.py` for a complete example.

## Existing Agents for Reference

- `ipw/agents/react.py` -- Simple tool wrapping with Agno
- `ipw/agents/openhands.py` -- Callback-based instrumentation with context condensing
- `ipw/agents/terminus.py` -- Docker container management with tmux sessions
- `ipw/agents/terminus_tb.py` -- Lightweight Terminus2 wrapper using runner-managed Docker
