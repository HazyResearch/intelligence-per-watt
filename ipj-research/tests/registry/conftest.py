from __future__ import annotations

import pytest

from ipj.registry.record import (
    BenchmarkSpec,
    HardwareSpec,
    HarnessSpec,
    JudgeSpec,
    Metrics,
    ModelSpec,
    RunRecord,
    SamplingSpec,
    ServingSpec,
)


def _build_rec() -> RunRecord:
    return RunRecord(
        model=ModelSpec(id="openai/Qwen3.6-27B", hf_revision="abc123", quant="FP8"),
        benchmark=BenchmarkSpec(name="terminal-bench/terminal-bench-2-1", version=None,
                                taskset_hash="sha256:7d7b", n_tasks=89),
        harness=HarnessSpec(id="terminus-2", git_sha="deadbeef"),
        sampling=SamplingSpec(temperature=0.7),
        judge=JudgeSpec(model="gpt-5-nano", version="2025-08-07"),
        hardware=HardwareSpec(gpu="H100", count=1, tp=1),
        serving=ServingSpec(concurrency=32),
        n_attempts=1,
        orchestration={},
        metrics=Metrics(accuracy=0.76, n_total=89),
    )


@pytest.fixture
def make_rec():
    """Factory returning a fresh, independent RunRecord each call."""
    return _build_rec
