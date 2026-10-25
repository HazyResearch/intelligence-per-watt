from __future__ import annotations

import json

from click.testing import CliRunner

from ipj import cell as cellmod
from ipj.cli import cli
from ipj.metrics import Cell


def test_measure_cli_writes_and_appends_cell(tmp_path, monkeypatch):
    # Stub the real run; the CLI's job is arg-wiring + write/append.
    fake = Cell(
        model="openai/gpt-5.5-2026-04-23", harness="terminus-2", hardware="cloud",
        dataset="terminal-bench/terminal-bench-2-1", n_tasks=5, accuracy=1.0,
        tokens_in=238932, tokens_out=31550, dollars=1.405, dollars_basis="api-billed",
        joules=None, wall_clock_s=995.8, energy_source="n/a (cloud)", run_dir=str(tmp_path / "j"),
    )
    captured = {}

    def _fake_measure(**kw):
        captured.update(kw)
        return fake

    monkeypatch.setattr(cellmod, "measure_cell", _fake_measure)

    out = tmp_path / "cellout"
    jsonl = tmp_path / "cells.jsonl"
    res = CliRunner().invoke(cli, [
        "measure", "--model", "openai/gpt-5.5-2026-04-23", "--hardware", "cloud", "--mode", "cloud",
        "--dataset", "terminal-bench/terminal-bench-2-1", "--out", str(out), "--n-tasks", "5",
        "--cells-jsonl", str(jsonl),
    ])
    assert res.exit_code == 0, res.output
    # arg wiring: --mode cloud -> is_local False; --n-tasks threaded through
    assert captured["is_local"] is False
    assert captured["n_tasks"] == 5
    assert captured["energy_provenance"] == "unknown"
    assert captured["price_in_per_m"] is None
    assert captured["price_out_per_m"] is None
    # outputs written
    data = json.loads((out / "cell.json").read_text())
    assert data["accuracy"] == 1.0
    assert data["dollars_basis"] == "api-billed"
    assert json.loads(jsonl.read_text().strip())["dollars"] == 1.405


def test_measure_cli_mode_required(tmp_path):
    # --mode is required (no silent default between local/cloud).
    res = CliRunner().invoke(cli, ["measure", "--model", "m", "--hardware", "cloud", "--out", str(tmp_path)])
    assert res.exit_code != 0
    assert "mode" in res.output.lower()


def test_orchestrate_v2_has_router_flags():
    from ipj.cli import cli
    res = CliRunner().invoke(cli, ["orchestrate-v2", "--help"])
    assert res.exit_code == 0
    for flag in (
        "--pre-route",
        "--router-model",
        "--router-margin",
        "--router-key-file",
        "--router-base-url",
        "--energy-provenance",
    ):
        assert flag in res.output
