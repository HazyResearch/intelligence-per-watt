from __future__ import annotations

import shutil
from pathlib import Path

from ipj.registry.backfill import backfill_records, find_run_dirs

FIXTURE = Path(__file__).parent / "fixtures" / "run"


def test_find_run_dirs(tmp_path):
    dst = tmp_path / "campaign" / "2026-01-01"
    dst.mkdir(parents=True)
    for f in ("config.json", "result.json"):
        shutil.copy(FIXTURE / f, dst / f)
    (tmp_path / "not_a_run").mkdir()
    found = find_run_dirs(tmp_path)
    assert dst in found
    assert len(found) == 1


def test_backfill_records_marked_backfilled(tmp_path):
    dst = tmp_path / "c" / "r"
    dst.mkdir(parents=True)
    for f in ("config.json", "result.json"):
        shutil.copy(FIXTURE / f, dst / f)
    recs = backfill_records(tmp_path)
    assert len(recs) == 1
    assert recs[0].status == "backfilled"
    assert recs[0].model.id == "openai/Qwen3.6-27B"
