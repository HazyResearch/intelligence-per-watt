from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from ipj.registry.capture import capture_run
from ipj.registry.record import HardwareSpec
from ipj.registry.store import ensure_repo, pull_index, push_record

FIXTURE = Path(__file__).parent / "fixtures" / "run"

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not os.environ.get("HF_TOKEN"), reason="needs HF_TOKEN")
def test_push_pull_roundtrip_throwaway_repo():
    repo = f"ScalingIntelligence/ipj-results-test-{uuid.uuid4().hex[:8]}"
    ensure_repo(repo_id=repo)
    rec = capture_run(FIXTURE, hardware=HardwareSpec(gpu="H100"))
    assert push_record(rec, repo_id=repo) == "pushed"
    rows = pull_index(repo_id=repo)
    assert any(r["energy_sig"] == rec.energy_sig for r in rows)
