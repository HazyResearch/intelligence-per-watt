# Extending IPW

All components use a decorator-based registry pattern. Implement the abstract base class, register with the decorator, and it's available in the CLI.

## Adding an Inference Client

1. Create a file in `src/ipw/clients/`, subclass `InferenceClient`
2. Register with `@ClientRegistry.register("my-client")`
3. Import in `src/ipw/clients/__init__.py`

Key methods: `stream_chat_completion(model, prompt)` returns a `Response` with content, `ChatUsage`, and timing; `list_models()` returns model IDs; `health()` returns `True` if reachable.

```python
from ipw.core.registry import ClientRegistry
from ipw.core.types import ChatUsage, Response
from ipw.clients.base import InferenceClient
import time

@ClientRegistry.register("my-service")
class MyServiceClient(InferenceClient):
    client_id = "my-service"
    client_name = "MyService"

    def __init__(self, base_url, **config):
        super().__init__(base_url, **config)
        import my_service_sdk
        self._client = my_service_sdk.Client(base_url=base_url)

    def stream_chat_completion(self, model, prompt, **params):
        t0, first_t, parts = time.time(), None, []
        for chunk in self._client.stream(model=model, prompt=prompt):
            if first_t is None: first_t = time.time()
            parts.append(chunk.text)
        t1 = time.time()
        return Response(
            content="".join(parts),
            usage=ChatUsage(prompt_tokens=chunk.usage.input_tokens,
                            completion_tokens=chunk.usage.output_tokens,
                            total_tokens=chunk.usage.total_tokens),
            time_to_first_token_ms=((first_t - t0) * 1000) if first_t else 0.0,
            first_token_time=first_t, request_start_time=t0, request_end_time=t1,
        )

    def list_models(self):
        return [m.id for m in self._client.list_models()]

    def health(self):
        try: self._client.health(); return True
        except Exception: return False
```

Reference implementations: `ipw/clients/ollama.py` (HTTP),
`ipw/clients/openai.py`, `ipw/clients/vllm.py` (in-process, async engine),
`ipw/clients/mlx.py` and `ipw/clients/afm.py` (in-process, no server).

### Notes for non-HTTP backends

- **`base_url` is optional.** In-process clients pass a placeholder to
  `super().__init__` (`"in-process"` for `mlx`/`afm`, `"offline://vllm"` for
  `vllm`) and ignore it.
- **Async-only SDKs**: `stream_chat_completion` is synchronous. Use
  `ipw.clients._async_loop.AsyncLoopRunner`, which owns an event loop on a
  background thread, rather than `asyncio.run` per call.
- **Merge `self._config` in `stream_chat_completion`.** `ProfilerRunner` passes
  `--client-param` values to the **constructor**, and does not forward per-call
  params, so a client that only reads `**params` silently ignores them. Do
  `effective = {**self._config, **params}`, letting per-call values win.
- **Warm up in `prepare()`** so model load and JIT land outside the first energy
  window.
- **Set `request_start_time` / `request_end_time`** (from `time.time()`) so the
  runner can split prefill from decode energy accurately instead of falling back
  to the first telemetry sample.
- **Don't let one bad query kill the run.** `ProfilerRunner` does not catch
  exceptions from `stream_chat_completion`; anything that escapes ends the run
  and discards records since the last flush. Convert expected per-query failures
  (context overflow, refusals) into an empty `Response`.
- **Optional dependencies**: add an entry to `_CLIENT_CLASS_MAP` in
  `ipw/clients/__init__.py` with the extra name and, when it differs from the
  extra, the SDK's import root (`("afm", ..., "afm", "apple_fm_sdk")`). Import
  the SDK at module top level so a missing extra becomes a `MISSING_CLIENTS`
  hint instead of a crash.
- **Optional `describe()`** returning a dict adds backend metadata to
  `summary.json` under `client_info` — useful for facts not recoverable from the
  records (SDK version, context size, host chip).

## Adding a Dataset

1. Create a file in `src/ipw/datasets/`, subclass `DatasetProvider`
2. Register with `@DatasetRegistry.register("my-dataset")`
3. Import in `src/ipw/datasets/__init__.py`

Key methods: `iter_records()` yields `DatasetRecord(problem, answer, subject, dataset_metadata)`; `size()` returns the total count; `score(record, response, eval_client)` returns `(bool | None, metadata)`.

```python
from datasets import load_dataset
from ipw.core.registry import DatasetRegistry
from ipw.core.types import DatasetRecord
from ipw.datasets.base import DatasetProvider

@DatasetRegistry.register("my-benchmark")
class MyBenchmarkDataset(DatasetProvider):
    dataset_id = "my-benchmark"
    dataset_name = "My Benchmark"
    evaluation_method = "my-benchmark"

    def __init__(self, *, split="test", max_samples=None):
        rows = list(load_dataset("my-org/my-benchmark", split=split))[:max_samples]
        self._records = tuple(
            DatasetRecord(problem=r["question"], answer=r["answer"],
                          subject=r.get("category", "general"),
                          dataset_metadata={"id": r.get("id")})
            for r in rows if r.get("question") and r.get("answer")
        )

    def iter_records(self): return iter(self._records)
    def size(self): return len(self._records)

    def score(self, record, response, *, eval_client=None):
        correct = record.answer.lower().strip() == response.lower().strip()
        return correct, {"method": "exact_match"}
```

Reference implementations: `ipw/datasets/mmlu_pro.py`, `ipw/datasets/gaia.py`, `ipw/datasets/simpleqa.py`.

## Adding an Agent

1. Create a file in `src/ipw/agents/`, subclass `BaseAgent`
2. Register with `@AgentRegistry.register("my-agent")`
3. Import in `src/ipw/agents/__init__.py`

Key methods: `run(input, **kwargs)` returns `AgentRunResult`. Emit `lm_inference_start/end` and `tool_call_start/end` events for energy attribution.

```python
from ipw.agents.base import BaseAgent
from ipw.core.registry import AgentRegistry
from ipw.core.types import AgentRunResult

@AgentRegistry.register("my-agent")
class MyAgent(BaseAgent):
    def __init__(self, model, event_recorder=None, **kwargs):
        super().__init__(event_recorder=event_recorder)
        from my_framework import Agent as FrameworkAgent
        self.model, self._agent = model, FrameworkAgent(model=model, **kwargs)

    def run(self, input, **kwargs):
        tools = []
        self._record_event("lm_inference_start", model=self.model)
        try:
            for step in self._agent.iterate(input):
                if step.is_tool_call:
                    tools.append(step.tool_name)
                    self._record_event("tool_call_start", tool=step.tool_name)
                    step.execute()
                    self._record_event("tool_call_end", tool=step.tool_name)
            return AgentRunResult(
                content=self._agent.get_final_response(),
                tool_calls_attempted=len(tools), tool_calls_succeeded=len(tools),
                tool_names_used=tools, num_turns=len(tools))
        finally:
            self._record_event("lm_inference_end", model=self.model)
```

Reference implementations: `ipw/agents/react.py`, `ipw/agents/openhands.py`, `ipw/agents/terminus.py`.

## Adding a Collector (Rust)

1. Create a file in `energy-monitor/src/collectors/`
2. Implement the `TelemetryCollector` trait
3. Register in `energy-monitor/src/collectors/mod.rs`
4. Build: `uv run scripts/build_energy_monitor.py`

Key methods: `new()` initializes the platform library and records baseline energy; `collect()` returns a `Reading` with power/energy/temperature (use `-1.0` for unavailable metrics); `platform()` returns a platform identifier.

```rust
// energy-monitor/src/collectors/my_platform.rs
use super::{Reading, TelemetryCollector};

pub struct MyPlatformCollector { handle: MyLibraryHandle, baseline_energy: f64 }

impl MyPlatformCollector {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let handle = my_library::init()?;
        let baseline = handle.get_total_energy()?;
        Ok(Self { handle, baseline_energy: baseline })
    }
}

impl TelemetryCollector for MyPlatformCollector {
    fn collect(&mut self) -> Reading {
        let energy = self.handle.get_total_energy().unwrap_or(-1.0);
        Reading {
            power_watts: self.handle.get_power_watts().unwrap_or(-1.0),
            energy_joules: if energy >= 0.0 { energy - self.baseline_energy } else { -1.0 },
            temperature_celsius: self.handle.get_temperature().unwrap_or(-1.0),
            platform: "my-platform".to_string(),
            ..Default::default()
        }
    }

    fn platform(&self) -> &str { "my-platform" }
}
```

Register in `mod.rs` by adding `mod my_platform;` and inserting a `if let Ok(c) = my_platform::MyPlatformCollector::new() { return Box::new(c); }` block inside `create_collector()`.

Reference implementations: `nvidia.rs`, `amd.rs`, `macos.rs`, `linux_rapl.rs`.
