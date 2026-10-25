from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TBMeta:
    difficulty: str | None
    category: str | None
    tags: list[str]
    expert_time_min: float | None
    junior_time_min: float | None


def bare_task_name(task_id: str) -> str:
    """`terminal-bench/protein-assembly` -> `protein-assembly`."""
    return task_id.rsplit("/", 1)[-1]


def load_tb_metadata(task_id: str, repo_dir: Path) -> TBMeta:
    """Read a TerminalBench task's [metadata] from <repo_dir>/<bare-name>/task.toml.

    Returns an all-None TBMeta if the toml is absent (so callers never crash on a missing task).
    """
    toml_path = Path(repo_dir) / bare_task_name(task_id) / "task.toml"
    if not toml_path.exists():
        return TBMeta(None, None, [], None, None)
    md = tomllib.loads(toml_path.read_text()).get("metadata", {})
    return TBMeta(
        difficulty=md.get("difficulty"),
        category=md.get("category"),
        tags=list(md.get("tags", [])),
        expert_time_min=md.get("expert_time_estimate_min"),
        junior_time_min=md.get("junior_time_estimate_min"),
    )
