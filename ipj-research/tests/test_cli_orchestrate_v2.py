from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from ipj.cli import cli
from ipj.metrics import Cell


def _fake_cell():
    return Cell(model="cascade-v2", harness="terminus-2-cascade-v2", hardware="h100-local+cloud",
                dataset="ds", n_tasks=3, accuracy=0.66, tokens_in=1200, tokens_out=600,
                dollars=1.0, dollars_basis="gpu-hours", joules=5000.0,
                wall_clock_s=10.0, energy_source="measured-local+est-cloud", run_dir="x")


def test_orchestrate_v2_appends_cell(tmp_path: Path):
    keyfile = tmp_path / "key"
    keyfile.write_text("sk-test\n")
    cells = tmp_path / "cells.jsonl"

    class _Res:
        cell = _fake_cell()
        n_escalated = 1
        cloud_tokens = {"in": 200, "out": 100}
        n_local = 2
        n_cloud = 1
        n_deferred = 0

    # run_v2 is imported lazily inside the command -> patch it at the source module.
    with patch("ipj.orchestrate.routing.v2_turn.run_v2", return_value=_Res()) as m:
        r = CliRunner().invoke(cli, [
            "orchestrate-v2", "--local-model", "Qwen/Qwen3.6-27B-FP8", "--cloud-model", "gpt-5.5",
            "--local-base-url", "http://127.0.0.1:8817/v1", "--cloud-base-url", "https://api.openai.com/v1",
            "--cloud-key-file", str(keyfile), "--output-dir", str(tmp_path / "run"),
            "--cells", str(cells), "--n-tasks", "3", "--local-turn-budget", "15",
        ])
    assert r.exit_code == 0, r.output
    assert cells.read_text().count("\n") == 1
    assert m.call_args.kwargs["local_turn_budget"] == 15
    assert m.call_args.kwargs["cloud_api_key"] == "sk-test"
    assert m.call_args.kwargs["cloud_price_in"] is None
    assert m.call_args.kwargs["cloud_price_out"] is None
