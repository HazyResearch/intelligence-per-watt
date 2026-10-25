"""Allowlisted public registry records for sanitizer-backed trace uploads."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from typing import Any, Mapping, Sequence

from .artifact import PublicArtifactError, scan_sensitive_data
from .record import RunRecord

PUBLIC_RECORD_KIND = "ipj.public-run-record"
PUBLIC_RECORD_SCHEMA_VERSION = "1"

_HEX_SIGNATURE = re.compile(r"^[0-9a-f]{8,64}$")
_CODE_REVISION = re.compile(r"^[0-9a-fA-F]{7,64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TRACE_URI = re.compile(
    r"^hf://[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/traces/"
    r"(?P<energy_sig>[0-9a-f]{8,64})@[A-Za-z0-9._-]+$"
)
_STATUS_VALUES = frozenset({"ok", "partial", "backfilled"})

_TOP_LEVEL_FIELDS = frozenset(
    {
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
)
_MODEL_FIELDS = frozenset({"hf_revision", "id", "quant"})
_BENCHMARK_FIELDS = frozenset({"n_tasks", "name", "taskset_hash", "version"})
_HARNESS_FIELDS = frozenset({"git_sha", "id", "version"})
_SAMPLING_FIELDS = frozenset({"ctx_len", "max_tokens", "temperature", "top_p"})
_JUDGE_FIELDS = frozenset({"model", "version"})
_HARDWARE_FIELDS = frozenset({"accelerator_class", "count", "tensor_parallel"})
_SERVING_FIELDS = frozenset({"batch", "concurrency", "max_num_seqs"})
_METRIC_FIELDS = frozenset(
    {
        "accuracy",
        "dollars",
        "flops",
        "joules",
        "n_solved",
        "n_total",
        "tokens_in",
        "tokens_out",
        "tool_calls",
        "turns",
        "wall_clock_s",
    }
)
_ACCOUNTING_FIELDS = frozenset(
    {
        "complete_cost_eligible",
        "rigorous_energy_eligible",
        "rigorous_utility_eligible",
        "status",
    }
)
_LEGACY_ACCOUNTING_STATUS = "legacy_run_record_unverified"


class PublicRecordError(PublicArtifactError):
    """Raised when a run record is not eligible for a public index."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise PublicRecordError("public run record contains a non-JSON or non-finite value") from exc
    return (encoded + "\n").encode("utf-8")


def _check_fields(value: Mapping[str, Any], expected: frozenset[str], location: str) -> None:
    actual = frozenset(value)
    if actual == expected:
        return
    unknown = sorted(actual - expected)
    missing = sorted(expected - actual)
    raise PublicRecordError(
        f"public record field allowlist mismatch at {location}: unknown={unknown}; missing={missing}"
    )


def _require_optional_string(value: Any, location: str, *, max_length: int = 256) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value or len(value) > max_length:
        raise PublicRecordError(f"{location} must be a non-empty public string or null")


def _require_optional_number(value: Any, location: str, *, maximum: float | None = None) -> None:
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (maximum is not None and value > maximum)
    ):
        raise PublicRecordError(f"{location} must be a finite non-negative number or null")


def _require_optional_integer(value: Any, location: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PublicRecordError(f"{location} must be a non-negative integer or null")


def _raise_for_sensitive_data(value: Any) -> None:
    findings = scan_sensitive_data(value, location="public_record")
    if not findings:
        return
    summary = ", ".join(f"{finding.category} at {finding.location}" for finding in findings[:8])
    suffix = "" if len(findings) <= 8 else f" (+{len(findings) - 8} more)"
    raise PublicRecordError(f"public run record privacy scan failed: {summary}{suffix}")


def _validate_source_provenance(record: RunRecord) -> None:
    if record.git_dirty:
        raise PublicRecordError("public run records require a clean source revision")
    if not isinstance(record.ipj_git_sha, str) or not _CODE_REVISION.fullmatch(record.ipj_git_sha):
        raise PublicRecordError("public run records require a 7-64 character hexadecimal code revision")
    if not isinstance(record.accuracy_sig, str) or not _HEX_SIGNATURE.fullmatch(record.accuracy_sig):
        raise PublicRecordError("public run records require a valid accuracy_sig")
    if not isinstance(record.energy_sig, str) or not _HEX_SIGNATURE.fullmatch(record.energy_sig):
        raise PublicRecordError("public run records require a valid energy_sig")
    if record.status not in _STATUS_VALUES:
        raise PublicRecordError(f"unsupported public run status: {record.status!r}")


def sanitize_public_run_record(record: RunRecord) -> dict[str, Any]:
    """Return an allowlisted index row linked to a sanitized trace artifact.

    ``host``, ``run_dir``, ``raw_config``, arbitrary orchestration dictionaries,
    prompt templates, and parser/endpoint details are absent rather than
    redacted or hashed.
    """

    _validate_source_provenance(record)
    trace_match = _TRACE_URI.fullmatch(record.trace_uri) if isinstance(record.trace_uri, str) else None
    if trace_match is None:
        raise PublicRecordError("public run records require a sanitizer-backed HF trace URI")
    if trace_match.group("energy_sig") != record.energy_sig:
        raise PublicRecordError("public trace URI energy signature does not match the run record")

    payload: dict[str, Any] = {
        "artifact_kind": PUBLIC_RECORD_KIND,
        "schema_version": PUBLIC_RECORD_SCHEMA_VERSION,
        "model": {
            "id": record.model.id,
            "hf_revision": record.model.hf_revision,
            "quant": record.model.quant,
        },
        "benchmark": {
            "name": record.benchmark.name,
            "version": record.benchmark.version,
            "taskset_hash": record.benchmark.taskset_hash,
            "n_tasks": record.benchmark.n_tasks,
        },
        "harness": {
            "id": record.harness.id,
            "git_sha": record.harness.git_sha,
            "version": record.harness_version,
        },
        "sampling": {
            "temperature": record.sampling.temperature,
            "top_p": record.sampling.top_p,
            "max_tokens": record.sampling.max_tokens,
            "ctx_len": record.sampling.ctx_len,
        },
        "judge": {
            "model": record.judge.model,
            "version": record.judge.version,
        },
        "hardware": {
            "accelerator_class": record.hardware.gpu,
            "count": record.hardware.count,
            "tensor_parallel": record.hardware.tp,
        },
        "serving": {
            "batch": record.serving.batch,
            "max_num_seqs": record.serving.max_num_seqs,
            "concurrency": record.serving.concurrency,
        },
        "n_attempts": record.n_attempts,
        "accounting": {
            "status": _LEGACY_ACCOUNTING_STATUS,
            "rigorous_utility_eligible": False,
            "rigorous_energy_eligible": False,
            "complete_cost_eligible": False,
        },
        "metrics": {
            # Legacy RunRecord has no verifier, energy-coverage/provenance, or
            # complete-cost basis. Fail closed in the public index; strict
            # values remain available in the sanitizer-backed trace artifact.
            "accuracy": None,
            "joules": None,
            "dollars": None,
            "flops": record.metrics.flops,
            "tokens_in": record.metrics.tokens_in,
            "tokens_out": record.metrics.tokens_out,
            "wall_clock_s": record.metrics.wall_clock_s,
            "turns": record.metrics.turns,
            "tool_calls": record.metrics.tool_calls,
            "n_solved": None,
            "n_total": None,
        },
        "code_revision": record.ipj_git_sha,
        "code_dirty": False,
        "created_at": record.created_at,
        "status": record.status,
        "accuracy_sig": record.accuracy_sig,
        "energy_sig": record.energy_sig,
        "trace_uri": record.trace_uri,
    }
    _raise_for_sensitive_data(payload)
    _validate_public_run_record_payload(payload, require_hash=False)
    payload["record_sha256"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return payload


def _validate_public_run_record_payload(value: Mapping[str, Any], *, require_hash: bool) -> None:
    expected_top_level = _TOP_LEVEL_FIELDS if require_hash else _TOP_LEVEL_FIELDS - {"record_sha256"}
    _check_fields(value, expected_top_level, "$")
    if value["artifact_kind"] != PUBLIC_RECORD_KIND:
        raise PublicRecordError("unexpected public record kind")
    if value["schema_version"] != PUBLIC_RECORD_SCHEMA_VERSION:
        raise PublicRecordError("unsupported public record schema version")
    if value["code_dirty"] is not False:
        raise PublicRecordError("public record code_dirty must be false")
    if not isinstance(value["code_revision"], str) or not _CODE_REVISION.fullmatch(value["code_revision"]):
        raise PublicRecordError("invalid public record code revision")
    for name in ("accuracy_sig", "energy_sig"):
        if not isinstance(value[name], str) or not _HEX_SIGNATURE.fullmatch(value[name]):
            raise PublicRecordError(f"invalid public record {name}")
    trace_match = _TRACE_URI.fullmatch(value["trace_uri"]) if isinstance(value["trace_uri"], str) else None
    if trace_match is None:
        raise PublicRecordError("invalid public record trace_uri")
    if trace_match.group("energy_sig") != value["energy_sig"]:
        raise PublicRecordError("public trace URI energy signature mismatch")
    if value["status"] not in _STATUS_VALUES:
        raise PublicRecordError("invalid public record status")
    _require_optional_string(value["created_at"], "created_at")
    _require_optional_integer(value["n_attempts"], "n_attempts")

    nested_fields = {
        "model": _MODEL_FIELDS,
        "benchmark": _BENCHMARK_FIELDS,
        "harness": _HARNESS_FIELDS,
        "sampling": _SAMPLING_FIELDS,
        "judge": _JUDGE_FIELDS,
        "hardware": _HARDWARE_FIELDS,
        "serving": _SERVING_FIELDS,
        "metrics": _METRIC_FIELDS,
        "accounting": _ACCOUNTING_FIELDS,
    }
    for name, expected in nested_fields.items():
        nested = value[name]
        if not isinstance(nested, Mapping):
            raise PublicRecordError(f"{name} must be an object")
        _check_fields(nested, expected, name)

    accounting = value["accounting"]
    if accounting != {
        "status": _LEGACY_ACCOUNTING_STATUS,
        "rigorous_utility_eligible": False,
        "rigorous_energy_eligible": False,
        "complete_cost_eligible": False,
    }:
        raise PublicRecordError("legacy public record accounting must fail closed")

    for location, string_value in (
        ("model.id", value["model"]["id"]),
        ("model.hf_revision", value["model"]["hf_revision"]),
        ("model.quant", value["model"]["quant"]),
        ("benchmark.name", value["benchmark"]["name"]),
        ("benchmark.version", value["benchmark"]["version"]),
        ("benchmark.taskset_hash", value["benchmark"]["taskset_hash"]),
        ("harness.id", value["harness"]["id"]),
        ("harness.git_sha", value["harness"]["git_sha"]),
        ("harness.version", value["harness"]["version"]),
        ("judge.model", value["judge"]["model"]),
        ("judge.version", value["judge"]["version"]),
        ("hardware.accelerator_class", value["hardware"]["accelerator_class"]),
    ):
        _require_optional_string(string_value, location)
    if value["model"]["id"] is None or value["benchmark"]["name"] is None or value["harness"]["id"] is None:
        raise PublicRecordError("model, benchmark, and harness IDs are required")

    for name in ("n_tasks",):
        _require_optional_integer(value["benchmark"][name], f"benchmark.{name}")
    for name in ("max_tokens", "ctx_len"):
        _require_optional_integer(value["sampling"][name], f"sampling.{name}")
    for name in ("temperature", "top_p"):
        _require_optional_number(value["sampling"][name], f"sampling.{name}")
    for name in ("count", "tensor_parallel"):
        _require_optional_integer(value["hardware"][name], f"hardware.{name}")
    for name in ("batch", "max_num_seqs", "concurrency"):
        _require_optional_integer(value["serving"][name], f"serving.{name}")
    for name, metric in value["metrics"].items():
        if name in {"tokens_in", "tokens_out", "turns", "tool_calls", "n_solved", "n_total"}:
            _require_optional_integer(metric, f"metrics.{name}")
        else:
            _require_optional_number(
                metric,
                f"metrics.{name}",
                maximum=1.0 if name == "accuracy" else None,
            )
    for name in ("accuracy", "joules", "dollars", "n_solved", "n_total"):
        if value["metrics"][name] is not None:
            raise PublicRecordError(f"metrics.{name} requires strict provenance unavailable in legacy RunRecord")
    _canonical_json_bytes(value)


def validate_public_run_record(value: Mapping[str, Any]) -> None:
    """Validate a public index row and its sanitized-content hash."""

    _validate_public_run_record_payload(value, require_hash=True)
    supplied_hash = value["record_sha256"]
    if not isinstance(supplied_hash, str) or not _SHA256.fullmatch(supplied_hash):
        raise PublicRecordError("invalid public record hash")
    unhashed = dict(value)
    del unhashed["record_sha256"]
    expected_hash = hashlib.sha256(_canonical_json_bytes(unhashed)).hexdigest()
    if not hmac.compare_digest(supplied_hash, expected_hash):
        raise PublicRecordError("public record hash mismatch")
    _raise_for_sensitive_data(value)


def public_index_to_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Serialize only validated public rows with deterministic JSON."""

    for row in rows:
        validate_public_run_record(row)
    return b"".join(_canonical_json_bytes(row) for row in rows)


__all__ = [
    "PUBLIC_RECORD_KIND",
    "PUBLIC_RECORD_SCHEMA_VERSION",
    "PublicRecordError",
    "public_index_to_jsonl",
    "sanitize_public_run_record",
    "validate_public_run_record",
]
