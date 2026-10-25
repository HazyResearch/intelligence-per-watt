"""Run an external `harbor` evaluation and report the wall-clock window."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


@dataclass
class RunWindow:
    start: float        # epoch seconds, just before launch
    end: float          # epoch seconds, just after exit
    returncode: int


def run_command(cmd: Sequence[str], *, cwd: Optional[Path] = None, env=None) -> RunWindow:
    start = time.time()
    proc = subprocess.run(list(cmd), cwd=cwd, env=env)
    end = time.time()
    return RunWindow(start=start, end=end, returncode=proc.returncode)


def harbor_command(
    dataset: str,
    agent: str,
    model: Optional[str],
    *,
    output_path,
    k: int = 1,
    env_type: str = "docker",
    n_concurrent: int = 1,    # sequential by default → unambiguous per-run energy
    n_tasks: Optional[int] = None,
    agent_timeout_multiplier: Optional[float] = None,
) -> list[str]:
    # Flags verified against Harbor 0.13.2 `harbor run --help`:
    #   -d dataset · -a agent · -m model · --jobs-dir output · -k attempts ·
    #   --env env · --n-concurrent N · -l N-tasks · -y auto-confirm host-access prompts ·
    #   --agent-timeout-multiplier X (scale each task's agent timeout; <1 = fail-fast).
    cmd = [
        "harbor", "run", "-d", dataset, "-a", agent,
        "--jobs-dir", str(output_path),
        "-k", str(k), "--env", env_type, "--n-concurrent", str(n_concurrent), "-y",
    ]
    if agent_timeout_multiplier is not None:
        cmd += ["--agent-timeout-multiplier", str(agent_timeout_multiplier)]
    if n_tasks is not None:
        cmd += ["-l", str(n_tasks)]
    if model:
        cmd += ["-m", model]
    return cmd
