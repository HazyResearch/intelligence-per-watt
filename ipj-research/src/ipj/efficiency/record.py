from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .failure_modes import classify
from .flops import estimate_flops
from .tb_metadata import TBMeta, bare_task_name


@dataclass(slots=True)
class PerQueryRecord:
    task_id: str
    approach: str
    category: str          # closed | open | oracle | cascade | orchestration
    hardware: str
    precision: str
    # TB metadata (for stratification)
    difficulty: str | None
    tb_category: str | None
    tags: list[str]
    expert_time_min: float | None
    # outcome
    reward: float
    solved: bool
    failure_mode: str
    # the 6 efficiency axes
    input_tokens: int
    output_tokens: int
    dollars: float
    latency_s: float | None
    flops: float | None
    joules: float | None


def iso_to_epoch(s: str) -> float:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp()


def build_record(
    task_result: dict,
    *,
    approach: str,
    category: str,
    hardware: str,
    precision: str,
    tb_meta: TBMeta,
    model_params_b: float | None,
    joules: float | None = None,
) -> PerQueryRecord:
    """Assemble one PerQueryRecord from a Harbor per-task result.json dict."""
    reward = ((task_result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    reward = float(reward) if reward is not None else 0.0
    exc = (task_result.get("exception_info") or {}).get("exception_type")
    ar = task_result.get("agent_result") or {}
    in_tok = int(ar.get("n_input_tokens") or 0)
    out_tok = int(ar.get("n_output_tokens") or 0)
    # local/owned compute is free; cloud uses the harness-reported cost.
    dollars = float(ar.get("cost_usd") or 0.0)
    try:
        latency = iso_to_epoch(task_result["finished_at"]) - iso_to_epoch(task_result["started_at"])
    except (KeyError, ValueError, TypeError):
        latency = None
    flops = estimate_flops(model_params_b, in_tok + out_tok) if model_params_b is not None else None
    return PerQueryRecord(
        task_id=task_result.get("task_name", bare_task_name(approach)),
        approach=approach, category=category, hardware=hardware, precision=precision,
        difficulty=tb_meta.difficulty, tb_category=tb_meta.category, tags=tb_meta.tags,
        expert_time_min=tb_meta.expert_time_min,
        reward=reward, solved=reward >= 0.999, failure_mode=classify(reward, exc),
        input_tokens=in_tok, output_tokens=out_tok, dollars=dollars,
        latency_s=latency, flops=flops, joules=joules,
    )
