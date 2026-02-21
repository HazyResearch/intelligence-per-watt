# Traces

IPW provides two trace data models for recording per-step telemetry during agent execution: `TurnTrace` for individual turns and `QueryTrace` for complete tasks.

## Data Model

Both types are defined in `ipw/execution/trace.py`.

### TurnTrace

A `TurnTrace` captures telemetry for a single agent turn (one LLM call plus any tool calls):

```python
@dataclass
class TurnTrace:
    turn_index: int                              # Sequential turn number
    input_tokens: int = 0                        # Tokens consumed
    output_tokens: int = 0                       # Tokens generated
    tool_result_tokens: int = 0                  # Tokens from tool results
    tools_called: list[str] = []                 # Tool names invoked
    tool_latencies_s: dict[str, float] = {}      # Per-tool wall-clock time
    wall_clock_s: float = 0.0                    # Total wall-clock time
    error: str | None = None                     # Error message (if failed)
    gpu_energy_joules: float | None = None       # GPU energy
    cpu_energy_joules: float | None = None       # CPU energy
    gpu_power_avg_watts: float | None = None     # Average GPU power
    cpu_power_avg_watts: float | None = None     # Average CPU power
    cost_usd: float | None = None                # API cost
```

### QueryTrace

A `QueryTrace` aggregates all turns for a single task/query:

```python
@dataclass
class QueryTrace:
    query_id: str                                # Unique query identifier
    workload_type: str                           # Dataset type
    query_text: str = ""                         # Input prompt
    response_text: str = ""                      # Final response
    turns: list[TurnTrace] = []                  # Per-turn traces
    total_wall_clock_s: float = 0.0              # Total execution time
    completed: bool = False                      # Whether task completed
```

### Computed Properties

`QueryTrace` provides computed properties that aggregate across all turns:

| Property | Description |
|----------|-------------|
| `num_turns` | Number of turns (`len(turns)`) |
| `total_input_tokens` | Sum of all turns' `input_tokens` |
| `total_output_tokens` | Sum of all turns' `output_tokens` |
| `tool_call_count` | Total number of tool calls across all turns |
| `total_gpu_energy_joules` | Sum of all turns' GPU energy (None if no readings) |
| `total_cost_usd` | Sum of all turns' API cost (None if no cost data) |

## Per-Step vs Per-Query Aggregation

### Per-Step (TurnTrace)

Each turn captures the telemetry window for a single LLM call:

```
Turn 0: [lm_start ... lm_end] -> 150 input, 45 output, 12.5J GPU, $0.0015
Turn 1: [tool_start ... tool_end, lm_start ... lm_end] -> 200 input, 30 output, 18.3J GPU, $0.0020
Turn 2: [lm_start ... lm_end] -> 180 input, 60 output, 15.1J GPU, $0.0018
```

This granularity enables:

- Identifying which turns are most energy-expensive
- Correlating tool use with energy spikes
- Understanding token efficiency per reasoning step

### Per-Query (QueryTrace)

The query-level aggregation provides a summary view:

```
Query: "What is the population of Tokyo?"
  Total turns: 3
  Total tokens: 530 input + 135 output
  Total energy: 45.9 J
  Total cost: $0.0053
  Total time: 6.5s
  Completed: true
```

## Serialization

### JSONL Format

Traces are serialized as one JSON object per line:

```python
# Save a single trace
trace.save_jsonl(Path("traces.jsonl"))  # Appends to file

# Load all traces from a file
traces = QueryTrace.load_jsonl(Path("traces.jsonl"))
```

Each line is a complete `QueryTrace` with nested `TurnTrace` objects:

```json
{"query_id": "gaia_001", "workload_type": "gaia", "query_text": "...", "response_text": "...", "turns": [{"turn_index": 0, "input_tokens": 150, ...}], "total_wall_clock_s": 6.5, "completed": true}
```

### Dict Conversion

Both types support `to_dict()` and `from_dict()`:

```python
# Serialize to dict
data = trace.to_dict()

# Reconstruct from dict
trace = QueryTrace.from_dict(data)
```

### HuggingFace Dataset Conversion

Convert a list of traces to a HuggingFace Dataset for analysis:

```python
traces = QueryTrace.load_jsonl(Path("traces.jsonl"))
dataset = QueryTrace.to_hf_dataset(traces)

# Dataset columns:
# query_id, workload_type, query_text, response_text,
# num_turns, total_input_tokens, total_output_tokens,
# total_tool_calls, total_wall_clock_s,
# total_gpu_energy_joules, total_cost_usd,
# completed, trace_json
```

The `trace_json` column contains the full serialized trace for each query, preserving per-turn detail while keeping the dataset flat.

## Correlation with Energy Telemetry

Traces are designed to be correlated with the energy monitor's telemetry stream:

1. The `EventRecorder` timestamps each event (tool calls, LLM inference) using `time.time()`.
2. The `TelemetrySession` records readings with nanosecond timestamps from the energy monitor.
3. During analysis, time windows from events are matched to telemetry windows to attribute energy to specific actions.

This enables questions like:

- How much energy did the web search tool consume?
- What fraction of total energy went to LLM inference vs. tool execution?
- Which turn in the agent loop was most energy-intensive?
