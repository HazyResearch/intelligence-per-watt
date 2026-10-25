# ipj/tests/test_cli_orchestrate.py
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from ipj.cli import cli
from ipj.metrics import Cell


def _fake_cell(model):
    return Cell(model=model, harness="terminus-2-cascade-v1", hardware="h100-local+cloud",
                dataset="ds", n_tasks=4, accuracy=0.75, tokens_in=0, tokens_out=0,
                dollars=1.0, dollars_basis="gpu-hours", joules=4000.0,
                wall_clock_s=10.0, energy_source="measured-local+est-cloud", run_dir="x")


def test_orchestrate_v1_appends_two_cells(tmp_path: Path):
    baseline = tmp_path / "cloud_base.json"
    baseline.write_text("{}")
    cells = tmp_path / "cells.jsonl"

    class _Res:
        realized_cell = _fake_cell("cascade-v1")
        oracle_cell = _fake_cell("cascade-v1-oracle")
        escalated = ["t4"]

    # run_v1 is imported lazily inside the command (keeps `ipj.cli` import ipw-free),
    # so patch it at the source module, not on ipj.cli.
    with patch("ipj.orchestrate.routing.v1_query.run_v1", return_value=_Res()) as m:
        r = CliRunner().invoke(cli, [
            "orchestrate-v1", "--local-model", "Qwen/Qwen3.6-27B-FP8",
            "--local-base-url", "http://localhost:8816/v1",
            "--cloud-baseline-result", str(baseline),
            "--output-dir", str(tmp_path / "run"), "--cells", str(cells), "--n-tasks", "4",
        ])
    assert r.exit_code == 0, r.output
    assert cells.read_text().count("\n") == 2   # realized + oracle
    assert m.call_args.kwargs["local_model"] == "Qwen/Qwen3.6-27B-FP8"
