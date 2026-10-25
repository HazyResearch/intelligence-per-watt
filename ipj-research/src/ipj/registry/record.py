from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ModelSpec:
    id: str
    hf_revision: str | None = None
    quant: str | None = None


@dataclass(slots=True)
class BenchmarkSpec:
    name: str
    version: str | None = None
    taskset_hash: str | None = None
    n_tasks: int | None = None


@dataclass(slots=True)
class HarnessSpec:
    id: str
    git_sha: str | None = None


@dataclass(slots=True)
class SamplingSpec:
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    ctx_len: int | None = None
    reasoning_parser: str | None = None
    tool_parser: str | None = None
    template: str | None = None


@dataclass(slots=True)
class JudgeSpec:
    model: str | None = None
    version: str | None = None


@dataclass(slots=True)
class HardwareSpec:
    gpu: str | None = None
    count: int | None = None
    tp: int | None = None


@dataclass(slots=True)
class ServingSpec:
    batch: int | None = None
    max_num_seqs: int | None = None
    concurrency: int | None = None


@dataclass(slots=True)
class Metrics:
    accuracy: float | None = None
    joules: float | None = None
    dollars: float | None = None
    flops: float | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    wall_clock_s: float | None = None
    turns: int | None = None
    tool_calls: int | None = None
    n_solved: int | None = None
    n_total: int | None = None


@dataclass(slots=True)
class RunRecord:
    model: ModelSpec
    benchmark: BenchmarkSpec
    harness: HarnessSpec
    sampling: SamplingSpec
    judge: JudgeSpec
    hardware: HardwareSpec
    serving: ServingSpec
    n_attempts: int
    orchestration: dict[str, Any]
    metrics: Metrics
    ipj_git_sha: str | None = None
    git_dirty: bool = False
    harness_version: str | None = None
    created_at: str | None = None
    host: str | None = None
    run_dir: str | None = None
    status: str = "ok"  # ok | partial | backfilled
    accuracy_sig: str | None = None
    energy_sig: str | None = None
    trace_uri: str | None = None
    raw_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunRecord":
        d = dict(d)
        return cls(
            model=ModelSpec(**d.pop("model")),
            benchmark=BenchmarkSpec(**d.pop("benchmark")),
            harness=HarnessSpec(**d.pop("harness")),
            sampling=SamplingSpec(**d.pop("sampling")),
            judge=JudgeSpec(**d.pop("judge")),
            hardware=HardwareSpec(**d.pop("hardware")),
            serving=ServingSpec(**d.pop("serving")),
            metrics=Metrics(**d.pop("metrics")),
            **d,
        )
