from __future__ import annotations

import json
from pathlib import Path

from ipj.registry.public_record import sanitize_public_run_record
from ipj.registry.store import (
    collapse_public_rows,
    collapse_rows,
    index_to_jsonl,
    queue_pending,
    queue_pending_public,
)


def test_collapse_keeps_last_per_energy_sig_created_at():
    rows = [
        {"energy_sig": "e1", "created_at": "t1", "metrics": {"accuracy": 0.5}},
        {"energy_sig": "e1", "created_at": "t1", "metrics": {"accuracy": 0.6}},  # re-push, wins
        {"energy_sig": "e2", "created_at": "t1", "metrics": {"accuracy": 0.7}},
    ]
    out = collapse_rows(rows)
    assert len(out) == 2
    by_sig = {r["energy_sig"]: r for r in out}
    assert by_sig["e1"]["metrics"]["accuracy"] == 0.6


def test_index_to_jsonl_roundtrip():
    rows = [{"a": 1}, {"b": 2}]
    text = index_to_jsonl(rows)
    parsed = [json.loads(line) for line in text.splitlines() if line.strip()]
    assert parsed == rows


def test_queue_pending_writes_record_json(tmp_path, make_rec):
    rec = make_rec()
    rec.energy_sig = "deadbeef"
    path = queue_pending(rec, tmp_path)
    assert Path(path).exists()
    data = json.loads(Path(path).read_text())
    assert data["energy_sig"] == "deadbeef"


def test_push_record_queues_on_hf_error(tmp_path, monkeypatch, make_rec):
    """push_record must queue locally and return 'queued' when HF is unreachable."""
    import ipj.registry.store as store

    def boom(**kwargs):
        raise RuntimeError("simulated HF outage")

    monkeypatch.setattr(store, "pull_index", boom)
    rec = make_rec()
    rec.energy_sig = "deadbeef"
    result = store.push_record(rec, pending_dir=tmp_path)
    assert result == "queued"
    # The record must be persisted to pending_dir — never lost
    assert any(p.suffix == ".json" for p in tmp_path.iterdir())


def test_collapse_public_rows_keys_by_trace_uri():
    rows = [
        {"energy_sig": "e1", "trace_uri": "uri1", "metrics": {"accuracy": 0.5}},
        {"energy_sig": "e1", "trace_uri": "uri1", "metrics": {"accuracy": 0.6}},
        {"energy_sig": "e1", "trace_uri": "uri2", "metrics": {"accuracy": 0.7}},
    ]

    collapsed = collapse_public_rows(rows)

    assert len(collapsed) == 2
    assert next(row for row in collapsed if row["trace_uri"] == "uri1")["metrics"]["accuracy"] == 0.6


def test_queue_pending_public_never_serializes_raw_record_fields(tmp_path, make_rec):
    record = make_rec()
    record.ipj_git_sha = "deadbeef"
    record.git_dirty = False
    record.accuracy_sig = "a" * 16
    record.energy_sig = "b" * 16
    record.trace_uri = f"hf://example/public/traces/{record.energy_sig}@revision123"
    record.host = "compute-01.internal"
    record.run_dir = "/home/alice/private"
    record.raw_config = {"api_key": "sk-proj-abcdefghijklmnop"}
    public = sanitize_public_run_record(record)

    path = queue_pending_public(public, tmp_path)
    serialized = path.read_text()

    assert "compute-01.internal" not in serialized
    assert "/home/alice/private" not in serialized
    assert "sk-proj-abcdefghijklmnop" not in serialized
