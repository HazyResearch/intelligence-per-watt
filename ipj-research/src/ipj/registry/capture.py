from __future__ import annotations

import json
import socket
import subprocess
from pathlib import Path

from ipj.harbor.results import parse_harbor_summary

from .record import (
    BenchmarkSpec,
    HardwareSpec,
    HarnessSpec,
    JudgeSpec,
    Metrics,
    ModelSpec,
    RunRecord,
    SamplingSpec,
    ServingSpec,
)
from .signature import accuracy_signature, energy_signature


def _git_sha_and_dirty(repo: Path) -> tuple[str | None, bool]:
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        porcelain = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return sha, porcelain != ""
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, False


def capture_run(
    run_dir: str | Path,
    *,
    hardware: HardwareSpec | None = None,
    judge: JudgeSpec | None = None,
    repo_root: str | Path | None = None,
    status: str = "ok",
) -> RunRecord:
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    result = json.loads((run_dir / "result.json").read_text())
    summary = parse_harbor_summary(run_dir / "result.json")

    agent = (config.get("agents") or [{}])[0]
    ds = (config.get("datasets") or [{}])[0]
    sha, dirty = _git_sha_and_dirty(Path(repo_root) if repo_root else Path.cwd())

    rec = RunRecord(
        model=ModelSpec(id=agent.get("model_name", "unknown")),
        benchmark=BenchmarkSpec(
            name=ds.get("name", "unknown"),
            version=ds.get("version"),
            taskset_hash=ds.get("ref"),
            n_tasks=summary.n_tasks,
        ),
        harness=HarnessSpec(id=agent.get("name", "unknown")),
        sampling=SamplingSpec(),  # not recorded in Harbor config.json; left null
        judge=judge or JudgeSpec(),
        hardware=hardware or HardwareSpec(),
        serving=ServingSpec(concurrency=config.get("n_concurrent_trials")),
        n_attempts=int(config.get("n_attempts") or 1),
        orchestration={},
        metrics=Metrics(
            accuracy=summary.accuracy,
            dollars=summary.cost_usd,
            tokens_in=summary.tokens_in,
            tokens_out=summary.tokens_out,
            n_total=summary.n_tasks,
        ),
        ipj_git_sha=sha,
        git_dirty=dirty,
        created_at=result.get("finished_at"),
        host=socket.gethostname(),
        run_dir=str(run_dir),
        status=status,
        raw_config=config,
    )
    rec.accuracy_sig = accuracy_signature(rec)
    rec.energy_sig = energy_signature(rec)
    return rec
