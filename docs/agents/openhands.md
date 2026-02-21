# OpenHands Agent

The OpenHands agent uses the [OpenHands SDK](https://github.com/All-Hands-AI/OpenHands) for autonomous task execution with per-tool energy tracking. It is designed for complex, multi-step tasks such as software engineering, research, and document analysis.

## Installation

```bash
uv pip install -e 'intelligence-per-watt[openhands]'
```

This installs the `openhands-sdk` package.

## How It Works

The OpenHands agent (`ipw/agents/openhands.py`) creates an OpenHands `Agent` with an `LLMSummarizingCondenser` for context management and instruments the agent loop via callbacks.

### Initialization

1. Create an OpenHands `Agent` with the specified LLM model.
2. Register any MCP tools as OpenHands `ToolDefinition` instances using `_register_mcp_tools()`.
3. Set up an `LLMSummarizingCondenser` (max 24k tokens, keep first 2 messages) for context window management.
4. Create a `Conversation` with the instrumented callback.

### Instrumented Callback

The `_instrumented_callback` method is called for every event in the agent loop:

```python
def _instrumented_callback(self, event):
    if isinstance(event, ActionEvent):
        # Tool call starting
        self._record_event("tool_call_start", tool=event.tool_name)
    elif isinstance(event, ObservationEvent):
        # Tool call completed
        self._record_event("tool_call_end", tool=tool_name)
```

This enables per-tool energy attribution by creating telemetry windows around each action.

### Turn-Cap Handling

If the agent exhausts its turn budget without calling `FinishTool`, IPW sends a synthesis nudge:

> You have run out of turns. Please provide your final answer now. Use the finish tool to submit your answer.

This gives the agent 2 additional turns to produce a final response. If that also fails, the last observed output from the callback is used as the response.

### Think Block Stripping

Extended thinking output (`<think>...</think>` blocks) is automatically stripped from the response text, as these blocks contain internal reasoning that should not appear in the final answer.

## Usage

### Via CLI

```bash
ipw run \
  --agent openhands \
  --model gpt-4o \
  --dataset swebench \
  --max-turns 30
```

### Programmatic Usage

```python
from ipw.agents.openhands import OpenHands
from ipw.telemetry.events import EventRecorder

recorder = EventRecorder()
agent = OpenHands(
    model="gpt-4o",
    max_turns=20,
    event_recorder=recorder,
)

result = agent.run("Fix the authentication bug in the login module")
print(result.content)
print(f"Turns: {result.num_turns}")
print(f"Tools used: {result.tool_names_used}")
```

### With MCP Tools

```python
from ipw.agents.mcp.openai_server import OpenAIServer
from ipw.agents.openhands import OpenHands

mcp_tools = {
    "gpt4": OpenAIServer(model="gpt-4o"),
}

agent = OpenHands(
    model="gpt-4o",
    mcp_tools=mcp_tools,
    max_turns=20,
)

result = agent.run("Research and summarize the latest advances in...")
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | Any | required | LLM model instance |
| `tools` | list | None | OpenHands Tool specs |
| `mcp_tools` | dict[str, BaseMCPServer] | None | MCP server instances |
| `event_recorder` | EventRecorder | None | Telemetry recorder |
| `max_turns` | int | 20 | Maximum iterations per run |
| `**kwargs` | Any | -- | Passed to OpenHands `Agent()` |

## MCP Tool Registration

When MCP tools are provided, each is registered as an OpenHands `ToolDefinition`:

1. A `MCPToolExecutor` wraps the `BaseMCPServer.execute()` method.
2. An `IPWMCPAction` (containing a query string) and `IPWMCPObservation` are created as the action/observation types.
3. The tool is registered in the OpenHands tool registry with the prefix `mcp_`.

## Return Value

`OpenHands.run()` returns an `AgentRunResult` with:

```python
AgentRunResult(
    content="I've fixed the authentication bug by...",
    tool_calls_attempted=15,
    tool_calls_succeeded=15,
    tool_names_used=["bash", "file_editor", "mcp_gpt4"],
    num_turns=12,
)
```

## Default Tools

If no tools or MCP tools are provided, the OpenHands agent uses its default tool set (`get_default_agent(cli_mode=True)`), which includes file operations, bash execution, and other standard development tools.
