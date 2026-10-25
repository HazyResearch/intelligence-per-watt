"""Linearize a kept Trajectory into an SFT example (messages + metadata) and write JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ipj.orchestrate.harness.trace import Trajectory


def to_sft_example(traj: Trajectory) -> dict[str, Any]:
    o = traj.outcome
    return {
        "messages": traj.to_sft_messages(),
        "metadata": {
            "task_id": traj.task.task_id,
            "adp_source": traj.task.adp_source,
            "domain": traj.task.domain,
            "success": o.success if o else None,
            "dollars": o.total_dollars if o else None,
            "gpu_energy_j": o.total_gpu_energy_j if o else None,
            "gpu_energy_provenance": o.gpu_energy_provenance if o else "unknown",
            "gpu_energy_complete": o.gpu_energy_complete if o else False,
            "strategy_tags": list(traj.strategy_tags),
            "budget_card": traj.budget_card,
            "brain": traj.brain,
            "n_turns": len(traj.steps),
        },
    }


def write_jsonl(examples: list[dict[str, Any]], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
