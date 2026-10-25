from __future__ import annotations

from pathlib import Path

from ipj.registry.capture import capture_run
from ipj.registry.record import HardwareSpec

FIXTURE = Path(__file__).parent / "fixtures" / "run"


def test_capture_reads_model_benchmark_metrics():
    rec = capture_run(FIXTURE, hardware=HardwareSpec(gpu="H100", count=1, tp=1))
    assert rec.model.id == "openai/Qwen3.6-27B"
    assert rec.benchmark.name == "terminal-bench/terminal-bench-2-1"
    assert rec.benchmark.taskset_hash == "sha256:7d7bdc1c"
    assert rec.benchmark.n_tasks == 89
    assert rec.metrics.accuracy == 0.76
    assert rec.metrics.tokens_in == 1000
    assert rec.serving.concurrency == 32
    assert rec.created_at == "2026-07-08T05:29:15"


def test_capture_populates_signatures():
    rec = capture_run(FIXTURE, hardware=HardwareSpec(gpu="H100"))
    assert rec.accuracy_sig and len(rec.accuracy_sig) == 16
    assert rec.energy_sig and rec.energy_sig != rec.accuracy_sig


def test_capture_stamps_git_dirty_flag_type(tmp_path):
    # tmp_path is outside any git repo -> sha None, dirty False (never crashes)
    rec = capture_run(FIXTURE, repo_root=tmp_path)
    assert rec.ipj_git_sha is None
    assert rec.git_dirty is False
