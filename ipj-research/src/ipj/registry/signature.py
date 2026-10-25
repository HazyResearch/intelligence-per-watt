from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from .record import RunRecord


def _canonical(obj: object) -> str:
    # sorted keys + compact separators => stable across dict ordering; json normalizes floats
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _hash(payload: dict) -> str:
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:16]


def accuracy_signature(rec: RunRecord) -> str:
    """Everything that determines the correctness distribution; NOT hardware/serving/host/metrics."""
    payload = {
        "model": asdict(rec.model),
        "benchmark": asdict(rec.benchmark),
        "harness": asdict(rec.harness),
        "sampling": asdict(rec.sampling),
        "judge": asdict(rec.judge),
        "n_attempts": rec.n_attempts,
        "orchestration": rec.orchestration,
    }
    return _hash(payload)


def energy_signature(rec: RunRecord) -> str:
    """accuracy_sig plus the physical execution config that determines J/$ per task."""
    payload = {
        "accuracy_sig": accuracy_signature(rec),
        "hardware": asdict(rec.hardware),
        "serving": asdict(rec.serving),
    }
    return _hash(payload)
