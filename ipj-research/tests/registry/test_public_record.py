"""Allowlist and integrity tests for public registry index rows."""

from __future__ import annotations

import json

import pytest

from ipj.registry.public_record import (
    PublicRecordError,
    public_index_to_jsonl,
    sanitize_public_run_record,
    validate_public_run_record,
)


def _make_public_ready(make_rec):
    record = make_rec()
    record.ipj_git_sha = "deadbeef"
    record.git_dirty = False
    record.accuracy_sig = "a" * 16
    record.energy_sig = "b" * 16
    record.trace_uri = f"hf://example/public/traces/{record.energy_sig}@revision123"
    return record


def test_public_record_omits_raw_and_infrastructure_fields(make_rec) -> None:
    record = _make_public_ready(make_rec)
    record.host = "compute-01.internal"
    record.run_dir = "/home/alice/private/run"
    record.raw_config = {
        "api_key": "sk-proj-abcdefghijklmnop",
        "prompt": "private prompt",
    }
    record.orchestration = {"endpoint": "10.42.0.9", "tool_output": "private"}
    record.sampling.template = "private system prompt"
    record.sampling.reasoning_parser = "private-parser"

    public = sanitize_public_run_record(record)
    serialized = public_index_to_jsonl([public]).decode()

    assert set(public) == {
        "accuracy_sig",
        "accounting",
        "artifact_kind",
        "benchmark",
        "code_dirty",
        "code_revision",
        "created_at",
        "energy_sig",
        "hardware",
        "harness",
        "judge",
        "metrics",
        "model",
        "n_attempts",
        "record_sha256",
        "sampling",
        "schema_version",
        "serving",
        "status",
        "trace_uri",
    }
    assert "host" not in public
    assert "run_dir" not in public
    assert "raw_config" not in public
    assert "orchestration" not in public
    assert "template" not in public["sampling"]
    assert "reasoning_parser" not in public["sampling"]
    assert public["accounting"] == {
        "status": "legacy_run_record_unverified",
        "rigorous_utility_eligible": False,
        "rigorous_energy_eligible": False,
        "complete_cost_eligible": False,
    }
    assert public["metrics"]["accuracy"] is None
    assert public["metrics"]["joules"] is None
    assert public["metrics"]["dollars"] is None
    assert public["metrics"]["n_solved"] is None
    assert public["metrics"]["n_total"] is None
    for private_value in (
        "compute-01.internal",
        "/home/alice/private/run",
        "sk-proj-abcdefghijklmnop",
        "private prompt",
        "10.42.0.9",
        "private system prompt",
        "private-parser",
    ):
        assert private_value not in serialized
    validate_public_run_record(public)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("git_dirty", True, "clean source revision"),
        ("ipj_git_sha", None, "code revision"),
        ("trace_uri", None, "trace URI"),
    ],
)
def test_public_record_requires_clean_complete_provenance(make_rec, field: str, value: object, match: str) -> None:
    record = _make_public_ready(make_rec)
    setattr(record, field, value)

    with pytest.raises(PublicRecordError, match=match):
        sanitize_public_run_record(record)


def test_public_record_requires_trace_uri_to_match_energy_signature(make_rec) -> None:
    record = _make_public_ready(make_rec)
    record.trace_uri = f"hf://example/public/traces/{'c' * 16}@revision123"

    with pytest.raises(PublicRecordError, match="energy signature does not match"):
        sanitize_public_run_record(record)


@pytest.mark.parametrize(
    ("accelerator", "category"),
    [
        ("GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "hardware_unique_identifier"),
        ("serial=ABCDEF123456", "hardware_unique_identifier"),
        ("compute-01", "private_hostname"),
        ("192.222.48.102", "ip_address"),
    ],
)
def test_public_record_scans_allowlisted_provenance(make_rec, accelerator: str, category: str) -> None:
    record = _make_public_ready(make_rec)
    record.hardware.gpu = accelerator

    with pytest.raises(PublicRecordError, match=category):
        sanitize_public_run_record(record)


def test_public_record_hash_detects_tampering(make_rec) -> None:
    public = sanitize_public_run_record(_make_public_ready(make_rec))
    public["metrics"]["tokens_in"] = 1

    with pytest.raises(PublicRecordError, match="hash mismatch"):
        validate_public_run_record(public)


@pytest.mark.parametrize(
    "metric",
    ["accuracy", "joules", "dollars", "n_solved", "n_total"],
)
def test_public_record_rejects_unverified_headline_metrics(make_rec, metric: str) -> None:
    public = sanitize_public_run_record(_make_public_ready(make_rec))
    public["metrics"][metric] = 1

    with pytest.raises(PublicRecordError, match="requires strict provenance"):
        validate_public_run_record(public)


def test_public_index_serialization_is_deterministic(make_rec) -> None:
    public = sanitize_public_run_record(_make_public_ready(make_rec))

    first = public_index_to_jsonl([public])
    second = public_index_to_jsonl([json.loads(first)])

    assert first == second
