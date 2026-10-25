"""ADPTask record and load_smoke_tasks (the built-in smoke task set)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .trace import TaskMeta


@dataclass(slots=True)
class ADPTask:
    meta: TaskMeta
    verify: Callable[[str], bool]


def _contains(gold: str) -> Callable[[str], bool]:
    def _v(answer: str) -> bool:
        return gold.lower() in (answer or "").lower()

    return _v


def load_smoke_tasks() -> list[ADPTask]:
    """A few self-contained verifiable tasks. Placeholder for the ADP verifiable subset (sub-project #2)."""
    specs = [
        ("arithmetic", "sum-1-10", "What is the sum of the integers from 1 to 10? Answer with the number only.", "55"),
        ("string", "reverse", "Reverse the string 'level'. Answer with the resulting string only.", "level"),
        (
            "bash",
            "count-words",
            "How many words are in the sentence 'the quick brown fox'? Answer with the number only.",
            "4",
        ),
    ]
    tasks: list[ADPTask] = []
    for domain, tid, prompt, gold in specs:
        meta = TaskMeta(
            adp_source="smoke", task_id=tid, domain=domain, prompt=prompt, verifiable=True, gold=gold
        )
        tasks.append(ADPTask(meta=meta, verify=_contains(gold)))
    return tasks
