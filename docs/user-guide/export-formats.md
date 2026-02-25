# Export Formats

IPW produces results in multiple formats depending on the profiling mode.

## HuggingFace Arrow Dataset

The primary output format for both single-turn and agentic profiling. Data is stored as Apache Arrow files compatible with the HuggingFace `datasets` library.

### Location

```
runs/<run_dir>/data-00000-of-00001.arrow
```

### Schema (Single-Turn)

Each row represents one query:

| Column | Type | Description |
|--------|------|-------------|
| `problem` | string | Input prompt sent to the model |
| `answer` | string | Reference/ground-truth answer |
| `subject` | string | Subject category from the dataset |
| `dataset_metadata` | dict | Dataset-specific metadata |
| `model_answers` | dict[str, str] | Map of model name to generated response |
| `model_metrics` | dict[str, ModelMetrics] | Map of model name to per-query metrics |

### ModelMetrics Structure

The `model_metrics` value for each model contains nested dataclasses:

```
ModelMetrics
    compute_metrics:
        flops_per_request: float
        total_flops: int
        flops_per_token: float
    energy_metrics:
        per_query_joules: float       # GPU energy
        total_joules: float
        cpu_per_query_joules: float   # CPU energy
        cpu_total_joules: float
        ane_per_query_joules: float   # ANE energy (macOS)
        ane_total_joules: float
    latency_metrics:
        per_token_ms: float
        throughput_tokens_per_sec: float
        time_to_first_token_seconds: float
        total_query_seconds: float
    memory_metrics:
        cpu_mb: {avg, max, median, min}
        gpu_mb: {avg, max, median, min}
    power_metrics:
        gpu: {per_query_watts, total_watts}
        cpu: {per_query_watts, total_watts}
    temperature_metrics: {avg, max, median, min}
    token_metrics:
        input: int
        output: int
        total: int
    phase_metrics:
        prefill_energy_j: float
        decode_energy_j: float
        prefill_duration_ms: float
        decode_duration_ms: float
    hardware_utilization:
        gpu:
            compute_utilization_pct: float
            memory_bandwidth_utilization_pct: float
            tensor_core_utilization_pct: float
            memory_used_gb: float
            memory_total_gb: float
        derived:
            mfu: float
            mbu: float
            arithmetic_intensity: float
    cost:
        input_cost_usd: float
        output_cost_usd: float
        tool_cost_usd: float
        total_cost_usd: float
    gpu_info: {name, vendor, device_id, device_type, backend}
    system_info: {os_name, os_version, kernel_version, host_name, cpu_count, cpu_brand}
    lm_correctness: bool
    lm_response: str
```

### Loading Arrow Datasets

```python
from datasets import load_from_disk

dataset = load_from_disk("./runs/profile_nvidia_llama3.2_1b_ipw/")

for row in dataset:
    model_metrics = row["model_metrics"]["llama3.2:1b"]
    energy = model_metrics["energy_metrics"]["per_query_joules"]
    print(f"Energy per query: {energy:.2f} J")
```

## JSONL Traces (Agentic Runs)

Agentic profiling produces JSONL files with one `QueryTrace` per line. This format preserves the full per-turn trace structure.

### Location

```
runs/<run_dir>/traces.jsonl
```

### Schema

Each line is a JSON object representing a `QueryTrace`:

```json
{
  "query_id": "gaia_001",
  "workload_type": "gaia",
  "query_text": "What is the population of...",
  "response_text": "The population is...",
  "turns": [
    {
      "turn_index": 0,
      "input_tokens": 150,
      "output_tokens": 45,
      "tool_result_tokens": 0,
      "tools_called": [],
      "tool_latencies_s": {},
      "wall_clock_s": 1.23,
      "error": null,
      "gpu_energy_joules": 12.5,
      "cpu_energy_joules": 3.2,
      "gpu_power_avg_watts": 180.0,
      "cpu_power_avg_watts": 45.0,
      "cost_usd": 0.0015
    },
    {
      "turn_index": 1,
      "input_tokens": 200,
      "output_tokens": 30,
      "tool_result_tokens": 500,
      "tools_called": ["web_search"],
      "tool_latencies_s": {"web_search": 0.85},
      "wall_clock_s": 2.10,
      "gpu_energy_joules": 18.3,
      "cost_usd": 0.0020
    }
  ],
  "total_wall_clock_s": 3.33,
  "completed": true
}
```

### Loading JSONL Traces

```python
from ipw.execution.trace import QueryTrace

# Load all traces from a file
traces = QueryTrace.load_jsonl(Path("./runs/run_react_gpt4o_gaia/traces.jsonl"))

for trace in traces:
    print(f"Query: {trace.query_id}")
    print(f"  Turns: {trace.num_turns}")
    print(f"  Total tokens: {trace.total_input_tokens + trace.total_output_tokens}")
    print(f"  Total energy: {trace.total_gpu_energy_joules} J")
    print(f"  Total cost: ${trace.total_cost_usd:.4f}")
```

### Converting JSONL to HuggingFace Dataset

```python
from ipw.execution.trace import QueryTrace

traces = QueryTrace.load_jsonl(Path("traces.jsonl"))
dataset = QueryTrace.to_hf_dataset(traces)
dataset.save_to_disk("./converted_dataset")
```

## summary.json

Every run produces a `summary.json` with run metadata:

```json
{
  "dataset": "ipw",
  "model": "llama3.2:1b",
  "client": "ollama",
  "start_time": "2025-01-15T10:30:00Z",
  "end_time": "2025-01-15T10:45:00Z",
  "total_queries": 100,
  "profiler_config": {
    "dataset_id": "ipw",
    "client_id": "ollama",
    "client_base_url": "http://localhost:11434",
    "model": "llama3.2:1b",
    "max_queries": 100
  },
  "hardware": {
    "gpu_name": "NVIDIA RTX 4090",
    "cpu_brand": "AMD Ryzen 9 7950X",
    "platform": "nvidia"
  }
}
```

## Analysis Reports

Analysis outputs are JSON files in the `analysis/` subdirectory:

```
analysis/
    accuracy.json      # IPJ/IPW metrics and per-record scores
    regression.json    # Energy/latency regression coefficients
```

See [Analysis](analysis.md) for details on the report contents.
