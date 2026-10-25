"""Load verified Trajectory JSONs from a traces dir (skips unreadable files with a warning)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ipj.orchestrate.harness.trace import Trajectory

_log = logging.getLogger(__name__)


def load_trajectories(traces_dir: str | Path) -> list[Trajectory]:
    out: list[Trajectory] = []
    for path in sorted(Path(traces_dir).glob("*.json")):
        try:
            out.append(Trajectory.from_dict(json.loads(path.read_text())))
        except Exception:
            _log.warning("skipping unreadable/invalid trace: %s", path)
    return out
