from __future__ import annotations

from pathlib import Path

from ipj.efficiency.tb_metadata import bare_task_name, load_tb_metadata

FIX = Path(__file__).parent / "fixtures"

def test_bare_task_name():
    assert bare_task_name("terminal-bench/protein-assembly") == "protein-assembly"
    assert bare_task_name("protein-assembly") == "protein-assembly"

def test_load_tb_metadata():
    m = load_tb_metadata("terminal-bench/sample-task", FIX)
    assert m.difficulty == "hard"
    assert m.category == "scientific-computing"
    assert m.tags == ["biology", "cloning"]
    assert m.expert_time_min == 60.0
    assert m.junior_time_min == 300.0

def test_load_tb_metadata_missing_returns_empty():
    m = load_tb_metadata("terminal-bench/does-not-exist", FIX)
    assert m.difficulty is None and m.tags == []
