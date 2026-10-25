from __future__ import annotations

import logging
from pathlib import Path

from .capture import capture_run
from .record import HardwareSpec, RunRecord

logger = logging.getLogger(__name__)


def find_run_dirs(root) -> list[Path]:
    root = Path(root)
    found: list[Path] = []
    for cfg in root.rglob("config.json"):
        if (cfg.parent / "result.json").exists():
            found.append(cfg.parent)
    return sorted(found)


def backfill_records(root, *, hardware: HardwareSpec | None = None) -> list[RunRecord]:
    recs: list[RunRecord] = []
    for d in find_run_dirs(root):
        try:
            recs.append(capture_run(d, hardware=hardware, status="backfilled"))
        except Exception as e:
            logger.warning("backfill: skipping %s: %s", d, e)
            continue  # skip malformed/partial dirs; never guess missing fields
    return recs
