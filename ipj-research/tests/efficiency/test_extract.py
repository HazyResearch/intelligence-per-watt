from __future__ import annotations

import json
from pathlib import Path

from ipj.efficiency.extract import extract_run, write_jsonl

FIX = Path(__file__).parent / "fixtures"

def test_extract_run_builds_one_record_per_task():
    recs = extract_run(FIX / "run", approach="Qwen/Qwen3.6-27B-FP8", category="open",
                       hardware="H100", precision="FP8", model_id="Qwen/Qwen3.6-27B-FP8", tb_repo=FIX)
    by_id = {r.task_id: r for r in recs}
    assert len(recs) == 2
    assert by_id["terminal-bench/sample-task"].solved is True
    assert by_id["terminal-bench/sample-task"].difficulty == "hard"   # joined from fixtures/sample-task/task.toml
    assert by_id["terminal-bench/other-task"].failure_mode == "timeout"
    assert by_id["terminal-bench/other-task"].difficulty is None      # no toml -> empty meta

def test_write_jsonl_roundtrip(tmp_path):
    recs = extract_run(FIX / "run", approach="m", category="open", hardware="hw",
                       precision="p", model_id="Qwen/Qwen3.6-27B-FP8", tb_repo=FIX)
    out = tmp_path / "rec.jsonl"
    write_jsonl(recs, out)
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(lines) == 2
    assert {line["task_id"] for line in lines} == {"terminal-bench/sample-task", "terminal-bench/other-task"}
    assert "flops" in lines[0] and "joules" in lines[0]
