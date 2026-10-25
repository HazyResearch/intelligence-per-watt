from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipj.cell import find_latest_result_json, measure_cell
from ipj.harbor.run import RunWindow


def _write_harbor_result(jobs_dir: Path, *, accuracy, n_tasks, tokens_in, tokens_out, cost_usd=None):
    job = jobs_dir / "2026-06-17__01-00-00"
    job.mkdir(parents=True, exist_ok=True)
    (job / "result.json").write_text(json.dumps({
        "n_total_trials": n_tasks,
        "stats": {
            "evals": {"terminus-2__terminal-bench/terminal-bench-2-1": {"metrics": [{"mean": accuracy}]}},
            "n_input_tokens": tokens_in,
            "n_output_tokens": tokens_out,
            "cost_usd": cost_usd,
        },
    }))


def _fake_run(**summary):
    def _run(cmd, **kw):
        out = Path(cmd[cmd.index("--jobs-dir") + 1])
        _write_harbor_result(out, **summary)
        return RunWindow(start=1000.0, end=1060.0, returncode=0)
    return _run


class _FakeCapture:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return None

    def joules(self, start, end):
        return 5000.0

    def accelerator_device_coverage_complete(self, start, end):
        return True


def test_local_cell_uses_gpu_hours_and_measured_joules(tmp_path):
    cell = measure_cell(
        model="openai/Qwen/Qwen3.5-27B-FP8", harness="terminus-2", hardware="H100",
        dataset="terminal-bench/terminal-bench-2-1", output_dir=tmp_path,
        is_local=True, k=1, n_tasks=5, gpu_hourly_rate=2.49, n_gpus=1,
        energy_provenance="measured_dedicated",
        run_fn=_fake_run(accuracy=0.6, n_tasks=5, tokens_in=100, tokens_out=20),
        capture_factory=lambda: _FakeCapture(),
    )
    assert cell.accuracy == 0.6
    assert cell.joules == 5000.0
    assert cell.dollars_basis == "gpu-hours"
    assert abs(cell.dollars - (60.0 / 3600.0) * 2.49) < 1e-9
    assert cell.n_tasks == 5
    assert cell.joules_provenance == "measured_dedicated"
    assert cell.joules_scope == "accelerator"
    assert cell.joules_complete is True
    assert cell.accelerator_device_coverage_complete is True


def test_local_cell_requires_explicit_isolation_provenance_for_complete_energy(tmp_path):
    cell = measure_cell(
        model="openai/Qwen/Qwen3.5-27B-FP8", harness="terminus-2", hardware="H100",
        dataset="terminal-bench/terminal-bench-2-1", output_dir=tmp_path,
        is_local=True, n_tasks=1,
        run_fn=_fake_run(accuracy=1.0, n_tasks=1, tokens_in=10, tokens_out=2),
        capture_factory=lambda: _FakeCapture(),
    )

    assert cell.joules == 5000.0
    assert cell.joules_provenance == "unknown"
    assert cell.joules_complete is False
    assert "isolation unverified" in cell.energy_source


def test_local_cell_marks_missing_telemetry_as_unknown(tmp_path):
    class _UnavailableCapture(_FakeCapture):
        def joules(self, start, end):
            return None

    cell = measure_cell(
        model="openai/Qwen/Qwen3.5-27B-FP8", harness="terminus-2", hardware="H100",
        dataset="terminal-bench/terminal-bench-2-1", output_dir=tmp_path,
        is_local=True, n_tasks=1,
        energy_provenance="measured_dedicated",
        run_fn=_fake_run(accuracy=1.0, n_tasks=1, tokens_in=10, tokens_out=2),
        capture_factory=lambda: _UnavailableCapture(),
    )

    assert cell.joules is None
    assert cell.joules_provenance == "unknown"
    assert cell.joules_complete is False
    assert "unknown" in cell.energy_source


def test_cloud_cell_joules_none_and_token_priced(tmp_path):
    cell = measure_cell(
        model="openai/gpt-5.5", harness="terminus-2", hardware="cloud",
        dataset="terminal-bench/terminal-bench-2-1", output_dir=tmp_path,
        is_local=False, k=1, n_tasks=5, price_in_per_m=1.25, price_out_per_m=10.0,
        run_fn=_fake_run(accuracy=0.8, n_tasks=5, tokens_in=1_000_000, tokens_out=1_000_000),
    )
    assert cell.joules is None
    assert cell.dollars_basis == "api-tokens"
    assert cell.dollars == 11.25
    assert cell.accuracy == 0.8
    assert cell.joules_provenance == "unknown"
    assert cell.joules_complete is False


def test_cloud_cell_prefers_harbor_billed_cost(tmp_path):
    # When Harbor reports cost_usd, the cloud cell uses it (real billed $) over the token estimate.
    cell = measure_cell(
        model="openai/gpt-5.5-2026-04-23", harness="terminus-2", hardware="cloud",
        dataset="terminal-bench/terminal-bench-2-1", output_dir=tmp_path,
        is_local=False, n_tasks=1, price_in_per_m=1.25, price_out_per_m=10.0,
        run_fn=_fake_run(accuracy=1.0, n_tasks=1, tokens_in=30976, tokens_out=7976, cost_usd=0.33656),
    )
    assert cell.dollars == 0.33656
    assert cell.dollars_basis == "api-billed"


def test_cloud_cell_rejects_unknown_cost_instead_of_using_zero(tmp_path):
    with pytest.raises(ValueError, match="cloud cost is unknown"):
        measure_cell(
            model="openai/unknown-model",
            harness="terminus-2",
            hardware="cloud",
            dataset="terminal-bench/terminal-bench-2-1",
            output_dir=tmp_path,
            is_local=False,
            n_tasks=1,
            run_fn=_fake_run(accuracy=1.0, n_tasks=1, tokens_in=100, tokens_out=20),
        )


def test_find_latest_result_json_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_latest_result_json(tmp_path)


def test_nonzero_harbor_exit_raises(tmp_path):
    # A failed harbor run (nonzero exit, no result.json written) fails loudly.
    def _failing_run(cmd, **kw):
        return RunWindow(start=1000.0, end=1001.0, returncode=1)

    with pytest.raises(RuntimeError):
        measure_cell(
            model="openai/gpt-5.5", harness="terminus-2", hardware="cloud",
            dataset="terminal-bench/terminal-bench-2-1", output_dir=tmp_path,
            is_local=False, run_fn=_failing_run,
        )


def test_env_type_flows_to_harbor_command(tmp_path):
    # env_type (e.g. "singularity" for Slurm) must reach `harbor run --env ...`.
    captured = {}

    def _capturing_run(cmd, **kw):
        captured["cmd"] = list(cmd)
        out = Path(cmd[cmd.index("--jobs-dir") + 1])
        _write_harbor_result(out, accuracy=0.5, n_tasks=2, tokens_in=10, tokens_out=2, cost_usd=0.001)
        return RunWindow(start=1.0, end=2.0, returncode=0)

    measure_cell(
        model="openai/Qwen/Qwen3.5-27B-FP8", harness="terminus-2", hardware="H100-slurm",
        dataset="terminal-bench/terminal-bench-2-1", output_dir=tmp_path,
        is_local=False, n_tasks=2, env_type="singularity", run_fn=_capturing_run,
    )
    cmd = captured["cmd"]
    assert cmd[cmd.index("--env") + 1] == "singularity"
