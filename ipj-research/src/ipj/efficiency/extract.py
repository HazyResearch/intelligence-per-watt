from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .flops import params_for
from .record import PerQueryRecord, build_record
from .tb_metadata import load_tb_metadata


def extract_run(
    run_dir: Path,
    *,
    approach: str,
    category: str,
    hardware: str,
    precision: str,
    model_id: str,
    tb_repo: Path,
    joules_by_task: dict[str, float] | None = None,
) -> list[PerQueryRecord]:
    """One PerQueryRecord per task subdir (`*__*/result.json`) under run_dir.

    Dedupe by task_id keeping the best reward (a task may have retry subdirs). joules_by_task maps
    a bare task name -> measured Joules for that task (local runs); None leaves joules unset.
    """
    params = params_for(model_id)
    best: dict[str, PerQueryRecord] = {}
    for pt in Path(run_dir).glob("*__*/result.json"):
        try:
            tr = json.loads(pt.read_text())
        except (ValueError, OSError):
            continue
        task_id = tr.get("task_name", pt.parent.name)
        meta = load_tb_metadata(task_id, tb_repo)
        joules = (joules_by_task or {}).get(task_id.rsplit("/", 1)[-1])
        rec = build_record(tr, approach=approach, category=category, hardware=hardware,
                           precision=precision, tb_meta=meta, model_params_b=params, joules=joules)
        prev = best.get(rec.task_id)
        if prev is None or rec.reward > prev.reward:
            best[rec.task_id] = rec
    return list(best.values())


def write_jsonl(records: list[PerQueryRecord], out_path: Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")
