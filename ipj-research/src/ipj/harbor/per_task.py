"""Per-(model, task) solve outcomes from a Harbor run.

This is the data Approach-A's A0 and the P1 sweep analysis need: which model
solves which task. Derived from the job result.json's per-eval reward_stats
(resolved/unresolved) and exception_stats (errored — e.g. timeouts).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskOutcome:
    task: str             # bare task name, e.g. "write-compressor"
    status: str           # "resolved" | "unresolved" | "errored"
    reward: float | None  # 1.0 resolved · 0.0 unresolved · None errored


def _task_of(trial_name: str) -> str:
    # Trial names are "<task>__<trialid>"; strip the trial id.
    return trial_name.rsplit("__", 1)[0]


def per_task_outcomes(result_json_path) -> list[TaskOutcome]:
    """Parse one Harbor job result.json into a per-trial outcome list."""
    data = json.loads(Path(result_json_path).read_text())
    evals = (data.get("stats") or {}).get("evals") or {}
    out: list[TaskOutcome] = []
    if not evals:
        return out
    ev = next(iter(evals.values()))  # one entry per (agent, dataset)
    for reward_str, trials in ((ev.get("reward_stats") or {}).get("reward") or {}).items():
        reward = float(reward_str)
        status = "resolved" if reward >= 1.0 else "unresolved"
        for t in trials:
            out.append(TaskOutcome(_task_of(t), status, reward))
    for trials in ((ev.get("exception_stats") or {}).values()):
        for t in trials:
            out.append(TaskOutcome(_task_of(t), "errored", None))
    return out


def solve_matrix(model_to_result: dict[str, str]) -> dict[str, dict[str, str]]:
    """{task: {model: status}} across runs — the cascade/sweep solve matrix.

    model_to_result maps a model label to its Harbor job result.json path.
    """
    matrix: dict[str, dict[str, str]] = {}
    for model, path in model_to_result.items():
        for o in per_task_outcomes(path):
            matrix.setdefault(o.task, {})[model] = o.status
    return matrix
