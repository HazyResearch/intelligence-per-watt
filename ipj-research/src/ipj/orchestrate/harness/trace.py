"""Trace schema: Step/Outcome/Trajectory plus SFT linearization, shared by harness/generate/filter."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from numbers import Real
from typing import Any, Callable, TypeGuard

from ipj.metrics import (
    MEASURED_DEDICATED_JOULES_PROVENANCE,
    UNKNOWN_JOULES_PROVENANCE,
    accelerator_joules_are_complete,
)


@dataclass(slots=True)
class Step:
    """One recorded action turn: the orchestrator's text plus per-turn cost/energy/latency and any worker/tool call."""

    index: int
    action_type: str
    orch_text: str
    call_target: str | None = None
    call_request: dict[str, Any] | None = None
    call_response: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    dollars: float | None = None
    latency_s: float | None = None
    gpu_energy_j: float | None = None
    correlation_id: str = ""
    gpu_energy_provenance: str = UNKNOWN_JOULES_PROVENANCE
    gpu_energy_complete: bool = False
    accelerator_device_coverage_complete: bool = False


@dataclass(slots=True)
class TaskMeta:
    """Task identity and provenance (task_id, prompt, domain, adp_source, verifiability)."""

    adp_source: str
    task_id: str
    domain: str
    prompt: str
    verifiable: bool
    gold: str | None = None


@dataclass(slots=True)
class Outcome:
    """Verified result of a trajectory: success verdict plus rolled-up dollars/tokens/energy."""

    success: bool
    source: str
    total_dollars: float | None
    total_gpu_energy_j: float | None
    total_latency_s: float | None
    total_tokens_in: int | None
    total_tokens_out: int | None
    gpu_energy_provenance: str = UNKNOWN_JOULES_PROVENANCE
    gpu_energy_complete: bool = False
    accelerator_device_coverage_complete: bool = False


@dataclass(slots=True)
class Trajectory:
    """A full orchestration episode: TaskMeta + steps + Outcome, with to_dict/from_dict and the SFT linearizer."""

    task: TaskMeta
    brain: str
    worker_pool_snapshot: dict[str, Any]
    budget_card: str
    steps: list[Step] = field(default_factory=list)
    outcome: Outcome | None = None
    strategy_tags: list[str] = field(default_factory=list)
    final_answer: str | None = None
    system_prompt: str = ""

    def to_sft_messages(self) -> list[dict[str, Any]]:
        # Loss-mask convention: only the orchestrator's own assistant turns get loss=True;
        # system/user/worker/tool turns are context-only (loss=False).
        msgs: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt or f"Budget: {self.budget_card}", "loss": False},
            {"role": "user", "content": self.task.prompt, "loss": False},
        ]
        for s in self.steps:
            msgs.append({"role": "assistant", "content": s.orch_text, "loss": True})
            if s.call_response is not None:
                msgs.append({"role": "tool", "content": s.call_response, "loss": False})
        return msgs

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Trajectory":
        task = TaskMeta(**d["task"])
        steps = [Step(**s) for s in d["steps"]]
        outcome = Outcome(**d["outcome"]) if d.get("outcome") is not None else None
        return cls(
            task=task,
            brain=d["brain"],
            worker_pool_snapshot=d["worker_pool_snapshot"],
            budget_card=d["budget_card"],
            steps=steps,
            outcome=outcome,
            strategy_tags=list(d.get("strategy_tags", [])),
            final_answer=d.get("final_answer"),
            system_prompt=d.get("system_prompt", ""),
        )


def derive_strategy_tags(steps: list[Step]) -> list[str]:
    tags: set[str] = set()
    for s in steps:
        if s.action_type in ("think", "bash"):
            tags.add("self")
        elif s.action_type == "call_tool":
            tags.add(f"tool:{s.call_target}")
        elif s.action_type == "call_worker":
            tags.add(f"worker:{s.call_target}")
        elif s.action_type == "delegate_task":
            tags.add(f"delegate:{s.call_target}")
    return sorted(tags)


def is_valid_nonnegative_number(value: object) -> TypeGuard[int | float]:
    """Return whether a value is a finite non-negative real, excluding booleans."""
    if (
        value is None
        or isinstance(value, bool)
        or not isinstance(value, Real)
        or value < 0
    ):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def is_valid_token_count(value: object) -> TypeGuard[int]:
    """Return whether a value is a non-negative integer token count."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _complete_real_sum(values: list[float | None]) -> float | None:
    if not all(is_valid_nonnegative_number(value) for value in values):
        return None
    return sum(float(value) for value in values if value is not None)


def _complete_token_sum(values: list[int | None]) -> int | None:
    if not all(is_valid_token_count(value) for value in values):
        return None
    return sum(value for value in values if value is not None)


def summarize_outcome(steps: list[Step], success: bool, source: str) -> Outcome:
    total_gpu_energy_j = _complete_real_sum([step.gpu_energy_j for step in steps])
    gpu_energy_complete = bool(steps) and all(
        step.gpu_energy_complete is True
        and step.accelerator_device_coverage_complete is True
        and is_valid_nonnegative_number(step.gpu_energy_j)
        and accelerator_joules_are_complete(
            step.gpu_energy_j,
            step.gpu_energy_provenance,
            step.accelerator_device_coverage_complete,
        )
        for step in steps
    )
    if not gpu_energy_complete:
        total_gpu_energy_j = None
    return Outcome(
        success=success,
        source=source,
        total_dollars=_complete_real_sum([step.dollars for step in steps]),
        total_gpu_energy_j=total_gpu_energy_j,
        total_latency_s=_complete_real_sum([step.latency_s for step in steps]),
        total_tokens_in=_complete_token_sum([step.tokens_in for step in steps]),
        total_tokens_out=_complete_token_sum([step.tokens_out for step in steps]),
        gpu_energy_provenance=(
            MEASURED_DEDICATED_JOULES_PROVENANCE
            if gpu_energy_complete
            else UNKNOWN_JOULES_PROVENANCE
        ),
        gpu_energy_complete=gpu_energy_complete,
        accelerator_device_coverage_complete=gpu_energy_complete,
    )


def score_trajectory(traj: Trajectory, verify: Callable[[str], bool]) -> Trajectory:
    success = bool(verify(traj.final_answer or ""))
    traj.outcome = summarize_outcome(traj.steps, success, "verifier")
    return traj
