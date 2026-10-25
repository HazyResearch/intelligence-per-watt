"""Router training-data writer: the §7.2 row schema and its two emitters.

A harvested task becomes one :class:`RouterRow`. From a list of rows we emit:

* :func:`write_classifier_table` — a JSONL table (one row-dict per line) the embedding
  classifier trains on (:func:`to_training_matrix` unpacks it into ``X, y, tasks``).
* :func:`write_sft_jsonl` — the LoRA-SFT head's ``messages`` JSONL. Each line is
  ``{"messages": [system, user, assistant], "metadata": {...}}`` where **only** the
  assistant turn carries ``loss: true`` (the ``Trajectory.to_sft_messages`` convention).
  The assistant turn is a decision JSON encoding both the escalate-prediction and the
  cost-awareness signal (§7.5).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Cost-awareness threshold: the 20x budget line ($4.50 Fable / 20 = $0.225/task). A cloud
# solve at or under this is "cheaply solvable" — the cost-aware half of the SFT decision.
CHEAP_CLOUD_COST = 0.225

ROUTER_INSTRUCTION = (
    "You are a routing controller for a local->cloud model cascade. Given a task, decide "
    "whether to keep it on the cheap local model or escalate it to the expensive cloud model. "
    "Escalate when the local model is unlikely to solve the task. Reply with ONLY a JSON object "
    '{"escalate": bool, "p_local_solve": float, "cheap_cloud_solvable": bool} where p_local_solve '
    "is your probability in [0,1] that the local model solves it and cheap_cloud_solvable is "
    "whether the cloud model would solve it within the escalation budget."
)


@dataclass(slots=True)
class RouterRow:
    """One harvested unique task (§7.2 schema)."""

    task_id: str
    task_text: str
    embedding: list[float]
    local_solved: bool
    cloud_solved: bool
    cloud_cost: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_text": self.task_text,
            "embedding": list(self.embedding),
            "local_solved": bool(self.local_solved),
            "cloud_solved": bool(self.cloud_solved),
            "cloud_cost": float(self.cloud_cost),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RouterRow:
        validate_row(d)
        return cls(
            task_id=d["task_id"],
            task_text=d["task_text"],
            embedding=[float(x) for x in d["embedding"]],
            local_solved=bool(d["local_solved"]),
            cloud_solved=bool(d["cloud_solved"]),
            cloud_cost=float(d["cloud_cost"]),
        )


_REQUIRED: dict[str, type | tuple[type, ...]] = {
    "task_id": str,
    "task_text": str,
    "embedding": (list, tuple),
    "local_solved": (bool, int),
    "cloud_solved": (bool, int),
    "cloud_cost": (int, float),
}


def validate_row(d: dict[str, Any]) -> None:
    """Raise ``ValueError`` unless ``d`` carries every §7.2 field with the right type."""
    for key, typ in _REQUIRED.items():
        if key not in d:
            raise ValueError(f"router row missing required field {key!r}")
        if not isinstance(d[key], typ):
            raise ValueError(f"router row field {key!r} must be {typ}, got {type(d[key])}")
    if isinstance(d["local_solved"], bool) is False and d["local_solved"] not in (0, 1):
        raise ValueError("local_solved must be boolean-valued")
    if not all(isinstance(x, (int, float)) for x in d["embedding"]):
        raise ValueError("embedding must be a sequence of numbers")


def decision_for(row: RouterRow, *, cheap_cloud_cost: float = CHEAP_CLOUD_COST) -> dict[str, Any]:
    """The supervised routing decision for ``row`` (the SFT assistant target).

    * ``escalate`` — route to cloud iff the local model did **not** solve the task; this is
      consistent with the score-cache semantics (escalate <=> predicted local fail).
    * ``p_local_solve`` — the solvability target (1.0 solved / 0.0 failed).
    * ``cheap_cloud_solvable`` — cost-awareness: cloud solved it within the escalation budget.
    """
    return {
        "escalate": not bool(row.local_solved),
        "p_local_solve": 1.0 if row.local_solved else 0.0,
        "cheap_cloud_solvable": bool(row.cloud_solved) and float(row.cloud_cost) <= cheap_cloud_cost,
    }


def _task_summary(row: RouterRow, *, max_chars: int = 2000) -> str:
    text = row.task_text.strip()
    if len(text) > max_chars:
        text = text[:max_chars]
    return f"Task {row.task_id}:\n{text}"


def to_sft_example(row: RouterRow, *, cheap_cloud_cost: float = CHEAP_CLOUD_COST) -> dict[str, Any]:
    """One row → ``{"messages": [...], "metadata": {...}}`` (assistant-only loss)."""
    decision = decision_for(row, cheap_cloud_cost=cheap_cloud_cost)
    messages = [
        {"role": "system", "content": ROUTER_INSTRUCTION, "loss": False},
        {"role": "user", "content": _task_summary(row), "loss": False},
        {"role": "assistant", "content": json.dumps(decision), "loss": True},
    ]
    return {
        "messages": messages,
        "metadata": {
            "task_id": row.task_id,
            "local_solved": bool(row.local_solved),
            "cloud_solved": bool(row.cloud_solved),
            "cloud_cost": float(row.cloud_cost),
            "cheap_cloud_cost": cheap_cloud_cost,
        },
    }


def to_training_matrix(
    rows: Sequence[RouterRow],
) -> tuple[list[list[float]], list[bool], list[str]]:
    """Unpack rows into ``(X embeddings, y local_solved, tasks)`` for the classifier."""
    X = [list(r.embedding) for r in rows]
    y = [bool(r.local_solved) for r in rows]
    tasks = [r.task_id for r in rows]
    return X, y, tasks


def write_classifier_table(rows: Sequence[RouterRow], path: str | Path) -> None:
    """Write the §7.2 rows as JSONL (one row-dict per line)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict()) + "\n")


def read_classifier_table(path: str | Path) -> list[RouterRow]:
    """Read a classifier table back into rows (round-trip of :func:`write_classifier_table`)."""
    rows: list[RouterRow] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(RouterRow.from_dict(json.loads(line)))
    return rows


def write_sft_jsonl(
    rows: Sequence[RouterRow],
    path: str | Path,
    *,
    cheap_cloud_cost: float = CHEAP_CLOUD_COST,
) -> None:
    """Write the LoRA-SFT ``messages`` JSONL (assistant-only loss, §7.5)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in rows:
            example = to_sft_example(row, cheap_cloud_cost=cheap_cloud_cost)
            f.write(json.dumps(example) + "\n")
