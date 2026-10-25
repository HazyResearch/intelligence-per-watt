"""Fail-closed construction and validation of public trace artifacts.

Raw traces are intentionally handled by the existing local export APIs.  This
module creates the narrower artifact that is safe to hand to a remote sink: it
uses explicit field and file allowlists, removes content by default, remaps raw
trace identifiers with a secret HMAC key, and verifies every sanitized byte
before returning.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import os
import re
import secrets
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import parse_qsl, urlsplit

from .trace import QueryTrace

ARTIFACT_KIND = "ipj.public-trace-artifact"
ARTIFACT_SCHEMA_VERSION = "1"
PUBLIC_TRACE_SCHEMA_VERSION = "ipj.public-query-trace.v1"
SANITIZER_NAME = "ipw-public-trace-sanitizer"
SANITIZER_VERSION = "1.0.0"
SCANNER_VERSION = "1.0.0"

MANIFEST_FILENAME = "artifact_manifest.json"
PRIVACY_REPORT_FILENAME = "privacy_report.json"
TRACES_FILENAME = "public_traces.jsonl"
SANITIZER_MARKER = "sanitized-and-verified-by-ipw-public-trace-sanitizer-v1"

_ALLOWED_ARTIFACT_FILES = frozenset({MANIFEST_FILENAME, PRIVACY_REPORT_FILENAME, TRACES_FILENAME})
_ALLOWED_CONTENT_FIELDS = frozenset({"query_text", "response_text"})
_MAX_CONTENT_FIELD_BYTES = 1_048_576
_MAX_ARTIFACT_FILE_BYTES = 64 * 1_048_576

# This is deliberately the exact current QueryTrace serialization contract.
# A newly added source field must receive an explicit privacy classification
# before the sanitizer accepts it.
_EXPECTED_QUERY_SOURCE_FIELDS = frozenset(
    {
        "accuracy_metadata",
        "accuracy_score",
        "accelerator_device_coverage_complete",
        "complete_total_cost_usd",
        "complete_total_cpu_energy_joules",
        "complete_total_gpu_energy_joules",
        "complete_total_task_cpu_energy_joules",
        "complete_total_task_gpu_energy_joules",
        "complete_energy_per_output_token_joules",
        "completed",
        "cost_coverage",
        "cost_known_turns",
        "cost_unknown_turns",
        "energy_per_output_token_joules",
        "gpu_energy_complete",
        "gpu_energy_provenance",
        "is_resolved",
        "judge_cpu_energy_joules",
        "judge_cpu_power_avg_watts",
        "judge_gpu_energy_joules",
        "judge_gpu_power_avg_watts",
        "judge_wall_clock_s",
        "legacy_lm_only_cost_usd",
        "lm_energy_measurable",
        "n_retries",
        "num_turns",
        "query_cpu_energy_joules",
        "query_cpu_power_avg_watts",
        "query_gpu_energy_joules",
        "query_gpu_power_avg_watts",
        "query_id",
        "query_mbu_avg_pct",
        "query_mbu_max_pct",
        "query_text",
        "response_text",
        "score_metadata",
        "shared_device_warning",
        "timed_out",
        "tool_call_count",
        "throughput_tokens_per_sec",
        "total_cost_usd",
        "total_cpu_energy_joules",
        "total_gpu_energy_joules",
        "total_input_tokens",
        "total_output_tokens",
        "total_task_cpu_energy_joules",
        "total_task_gpu_energy_joules",
        "total_tokens",
        "total_tool_calls",
        "total_wall_clock_s",
        "turns",
        "unscorable_reason",
        "utility_calibration_id",
        "utility_complete",
        "utility_provenance",
        "utility_task_manifest_id",
        "utility_task_manifest_size",
        "utility_task_membership_id",
        "utility_verifier_id",
        "utility_verifier_version",
        "workload_type",
    }
)
_EXPECTED_TURN_SOURCE_FIELDS = frozenset(
    {
        "cached_tokens",
        "complete_total_cost_usd",
        "cost_usd",
        "cpu_energy_joules",
        "cpu_power_avg_watts",
        "dram_energy_joules",
        "error",
        "gpu_energy_joules",
        "gpu_power_avg_watts",
        "has_parallel_tools",
        "input_tokens",
        "legacy_lm_only_cost_usd",
        "output_tokens",
        "peak_watts",
        "reasoning_tokens",
        "shared_device_warning",
        "tool_cost_complete",
        "tool_cost_usd",
        "tool_latencies_s",
        "tool_result_tokens",
        "tools_called",
        "total_cost_usd",
        "turn_index",
        "wall_clock_s",
    }
)

_PUBLIC_QUERY_FIELDS = frozenset(
    {
        "accuracy_score",
        "accelerator_device_coverage_complete",
        "complete_total_cost_usd",
        "complete_total_cpu_energy_joules",
        "complete_total_gpu_energy_joules",
        "complete_total_task_cpu_energy_joules",
        "complete_total_task_gpu_energy_joules",
        "complete_energy_per_output_token_joules",
        "completed",
        "cost_coverage",
        "cost_known_turns",
        "cost_unknown_turns",
        "energy_per_output_token_joules",
        "gpu_energy_complete",
        "gpu_energy_provenance",
        "is_resolved",
        "judge_cpu_energy_joules",
        "judge_cpu_power_avg_watts",
        "judge_gpu_energy_joules",
        "judge_gpu_power_avg_watts",
        "judge_wall_clock_s",
        "legacy_lm_only_cost_usd",
        "lm_energy_measurable",
        "n_retries",
        "num_turns",
        "public_trace_id",
        "query_cpu_energy_joules",
        "query_cpu_power_avg_watts",
        "query_gpu_energy_joules",
        "query_gpu_power_avg_watts",
        "query_mbu_avg_pct",
        "query_mbu_max_pct",
        "record_sha256",
        "schema_version",
        "shared_device_warning",
        "timed_out",
        "tool_call_count",
        "throughput_tokens_per_sec",
        "total_cost_usd",
        "total_cpu_energy_joules",
        "total_gpu_energy_joules",
        "total_input_tokens",
        "total_output_tokens",
        "total_task_cpu_energy_joules",
        "total_task_gpu_energy_joules",
        "total_tokens",
        "total_wall_clock_s",
        "turns",
        "utility_calibration_id",
        "utility_complete",
        "utility_provenance",
        "utility_task_manifest_id",
        "utility_task_manifest_size",
        "public_utility_task_membership_id",
        "utility_verifier_id",
        "utility_verifier_version",
        "workload_type",
    }
)
_PUBLIC_TURN_FIELDS = frozenset(
    {
        "cached_tokens",
        "complete_total_cost_usd",
        "cost_usd",
        "cpu_energy_joules",
        "cpu_power_avg_watts",
        "dram_energy_joules",
        "gpu_energy_joules",
        "gpu_power_avg_watts",
        "has_parallel_tools",
        "input_tokens",
        "legacy_lm_only_cost_usd",
        "output_tokens",
        "peak_watts",
        "reasoning_tokens",
        "shared_device_warning",
        "tool_call_count",
        "tool_cost_complete",
        "tool_cost_usd",
        "tool_result_tokens",
        "total_cost_usd",
        "turn_index",
        "wall_clock_s",
    }
)

_PROVENANCE_FIELDS = frozenset(
    {
        "benchmark_id",
        "benchmark_version",
        "code_dirty",
        "code_revision",
        "data_license",
        "driver_version",
        "hardware_class",
        "harness_id",
        "harness_revision",
        "model_id",
        "model_revision",
        "source_schema_version",
        "verifier_id",
        "verifier_version",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "artifact_kind",
        "artifact_schema_version",
        "content_retention_mode",
        "files",
        "policy_sha256",
        "provenance",
        "sanitizer",
        "sanitizer_marker",
        "trace_count",
    }
)
_FILE_MANIFEST_FIELDS = frozenset({"bytes", "path", "sha256"})
_PRIVACY_REPORT_FIELDS = frozenset({"checks", "finding_count", "scanner_version", "status"})

_SENSITIVE_FIELD_NAMES = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "authorization",
        "base_url",
        "credential",
        "credentials",
        "cwd",
        "endpoint",
        "environment",
        "headers",
        "home",
        "host",
        "hostname",
        "ip_address",
        "password",
        "private_key",
        "run_dir",
        "secret",
        "secret_key",
        "ssh_key",
        "token",
        "user",
        "username",
    }
)
_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private_key_material",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----", re.I),
    ),
    ("bearer_credential", re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}", re.I)),
    ("openai_credential", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{12,}\b")),
    ("anthropic_credential", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{12,}\b")),
    ("huggingface_credential", re.compile(r"\bhf_[A-Za-z0-9]{16,}\b")),
    ("github_credential", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("google_credential", re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    (
        "credential_assignment",
        re.compile(
            r"\b(?:api[_-]?key|access[_-]?token|auth(?:orization)?|password|passwd|"
            r"secret(?:[_-]?key)?|private[_-]?key)\s*[:=]\s*[^\s,;]{8,}",
            re.I,
        ),
    ),
    (
        "signed_url_credential",
        re.compile(r"(?:X-Amz-(?:Credential|Signature)|[?&](?:sig|signature|token)=)", re.I),
    ),
)
_EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_IPV4_PATTERN = re.compile(r"(?<![0-9.])(?:\d{1,3}\.){3}\d{1,3}(?![0-9.])")
_IPV6_CANDIDATE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9:])(?:\[[0-9A-Fa-f:.%]+\]|[0-9A-Fa-f:.]*:[0-9A-Fa-f:.]+)(?![A-Za-z0-9:])"
)
_UNIX_PATH_PATTERN = re.compile(r"(?<![A-Za-z0-9:/])/(?:[^\s/]+/)+[^\s]*")
_WINDOWS_PATH_PATTERN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:\\(?:[^\s\\]+\\)*[^\s]*")
_INTERNAL_HOST_PATTERN = re.compile(
    r"\b(?:localhost|[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.(?:internal|local|lan|corp))\b",
    re.I,
)
_BARE_PRIVATE_HOST_PATTERN = re.compile(
    r"\b(?:compute|node|host|worker|server|login|head|cn)[-_]?\d+\b",
    re.I,
)
_HARDWARE_UNIQUE_ID_PATTERN = re.compile(
    r"\b(?:GPU-)?[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.I,
)
_SERIAL_LABEL_PATTERN = re.compile(r"\b(?:serial(?:[_ -]?number)?|s/?n)\s*[:=]\s*[A-Za-z0-9-]{6,}\b", re.I)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_CODE_REVISION_PATTERN = re.compile(r"^[0-9a-fA-F]{7,64}$")
_PROVENANCE_STRING_MAX_BYTES = {
    "benchmark_id": 256,
    "benchmark_version": 128,
    "code_revision": 64,
    "data_license": 256,
    "driver_version": 128,
    "hardware_class": 128,
    "harness_id": 256,
    "harness_revision": 256,
    "model_id": 512,
    "model_revision": 256,
    "source_schema_version": 128,
    "verifier_id": 256,
    "verifier_version": 128,
}
_REQUIRED_PROVENANCE_STRING_FIELDS = frozenset(
    {"code_revision", "data_license", "source_schema_version"}
)


class PublicArtifactError(ValueError):
    """Raised when an artifact cannot satisfy the public data policy."""


@dataclass(frozen=True, slots=True)
class SensitiveDataFinding:
    """A non-content-bearing scanner result safe to show in diagnostics."""

    category: str
    location: str


def _validate_provenance_string(field_name: str, value: Any) -> None:
    """Require a bounded, non-blank string or an allowed ``None`` value."""

    required = field_name in _REQUIRED_PROVENANCE_STRING_FIELDS
    if value is None:
        if required:
            raise PublicArtifactError(f"{field_name} must be a string")
        return
    if type(value) is not str:
        raise PublicArtifactError(f"{field_name} must be a string or null")
    if not value.strip():
        raise PublicArtifactError(f"{field_name} must not be blank")
    if len(value.encode("utf-8")) > _PROVENANCE_STRING_MAX_BYTES[field_name]:
        raise PublicArtifactError(f"{field_name} exceeds the public provenance size limit")


@dataclass(frozen=True, slots=True)
class ArtifactProvenance:
    """Strict public provenance carried by every sanitized trace artifact."""

    code_revision: str
    code_dirty: bool
    data_license: str
    source_schema_version: str = "ipw.query-trace.v1"
    model_id: str | None = None
    model_revision: str | None = None
    benchmark_id: str | None = None
    benchmark_version: str | None = None
    harness_id: str | None = None
    harness_revision: str | None = None
    verifier_id: str | None = None
    verifier_version: str | None = None
    hardware_class: str | None = None
    driver_version: str | None = None

    def __post_init__(self) -> None:
        if type(self.code_dirty) is not bool:
            raise PublicArtifactError("code_dirty must be a boolean")
        for field_name in _PROVENANCE_STRING_MAX_BYTES:
            _validate_provenance_string(field_name, getattr(self, field_name))
        if not _CODE_REVISION_PATTERN.fullmatch(self.code_revision):
            raise PublicArtifactError("code_revision must be a 7-64 character hexadecimal revision")
        if self.code_dirty:
            raise PublicArtifactError("public artifacts require a clean source revision")


@dataclass(frozen=True, slots=True)
class PublicArtifactPolicy:
    """Explicit content-retention policy for public trace export."""

    retained_content_fields: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        unknown = self.retained_content_fields - _ALLOWED_CONTENT_FIELDS
        if unknown:
            raise PublicArtifactError(f"unsupported public content fields: {', '.join(sorted(unknown))}")

    @property
    def content_retention_mode(self) -> str:
        if not self.retained_content_fields:
            return "metadata_only"
        return "explicit:" + ",".join(sorted(self.retained_content_fields))


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise PublicArtifactError("artifact contains a non-JSON or non-finite value") from exc
    return (text + "\n").encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _object_no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PublicArtifactError(f"duplicate JSON key at {key!r}")
        result[key] = value
    return result


def _load_json_bytes(data: bytes, *, location: str) -> Any:
    try:
        return json.loads(data, object_pairs_hook=_object_no_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicArtifactError(f"invalid JSON in {location}") from exc


def _check_exact_fields(value: Mapping[str, Any], expected: frozenset[str], *, location: str) -> None:
    actual = frozenset(value)
    if actual == expected:
        return
    unknown = sorted(actual - expected)
    missing = sorted(expected - actual)
    details = []
    if unknown:
        details.append(f"unknown={unknown}")
    if missing:
        details.append(f"missing={missing}")
    raise PublicArtifactError(f"field allowlist mismatch at {location}: {'; '.join(details)}")


def _string_findings(value: str, location: str) -> list[SensitiveDataFinding]:
    findings = [
        SensitiveDataFinding(category, location) for category, pattern in _SECRET_PATTERNS if pattern.search(value)
    ]
    if _EMAIL_PATTERN.search(value):
        findings.append(SensitiveDataFinding("email_or_username", location))
    if _UNIX_PATH_PATTERN.search(value) or _WINDOWS_PATH_PATTERN.search(value):
        findings.append(SensitiveDataFinding("absolute_path", location))
    if value.startswith(("/", "\\\\", "file://")):
        findings.append(SensitiveDataFinding("absolute_path", location))
    if _INTERNAL_HOST_PATTERN.search(value):
        findings.append(SensitiveDataFinding("private_hostname", location))
    if _BARE_PRIVATE_HOST_PATTERN.search(value):
        findings.append(SensitiveDataFinding("private_hostname", location))
    if _HARDWARE_UNIQUE_ID_PATTERN.search(value) or _SERIAL_LABEL_PATTERN.search(value):
        findings.append(SensitiveDataFinding("hardware_unique_identifier", location))

    for match in _IPV4_PATTERN.finditer(value):
        try:
            ipaddress.ip_address(match.group(0))
        except ValueError:
            continue
        findings.append(SensitiveDataFinding("ip_address", location))
        break
    for match in _IPV6_CANDIDATE_PATTERN.finditer(value):
        candidate = match.group(0).strip("[]")
        candidate = candidate.split("%", maxsplit=1)[0]
        try:
            ipaddress.ip_address(candidate)
        except ValueError:
            continue
        findings.append(SensitiveDataFinding("ip_address", location))
        break

    parsed = urlsplit(value)
    if parsed.scheme and parsed.netloc:
        if parsed.username is not None or parsed.password is not None:
            findings.append(SensitiveDataFinding("url_embedded_credential", location))
        hostname = parsed.hostname
        if hostname:
            try:
                ipaddress.ip_address(hostname)
            except ValueError:
                if hostname.lower() == "localhost" or hostname.lower().endswith(
                    (".internal", ".local", ".lan", ".corp")
                ):
                    findings.append(SensitiveDataFinding("private_hostname", location))
            else:
                findings.append(SensitiveDataFinding("ip_address", location))
        sensitive_query_names = {
            "access_token",
            "api_key",
            "key",
            "secret",
            "sig",
            "signature",
            "token",
            "x-amz-credential",
            "x-amz-signature",
        }
        if any(name.lower() in sensitive_query_names for name, _ in parse_qsl(parsed.query)):
            findings.append(SensitiveDataFinding("url_query_credential", location))
    return findings


def scan_sensitive_data(value: Any, *, location: str = "$") -> tuple[SensitiveDataFinding, ...]:
    """Scan a JSON-like value without returning sensitive matched text."""

    findings: list[SensitiveDataFinding] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            child_location = f"{location}.{key_text}"
            if key_text.lower() in _SENSITIVE_FIELD_NAMES:
                findings.append(SensitiveDataFinding("sensitive_field_name", child_location))
            findings.extend(scan_sensitive_data(item, location=child_location))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            findings.extend(scan_sensitive_data(item, location=f"{location}[{index}]"))
    elif isinstance(value, str):
        findings.extend(_string_findings(value, location))
    return tuple(dict.fromkeys(findings))


def _raise_for_findings(findings: Iterable[SensitiveDataFinding]) -> None:
    unique = tuple(dict.fromkeys(findings))
    if not unique:
        return
    summary = ", ".join(f"{finding.category} at {finding.location}" for finding in unique[:8])
    suffix = "" if len(unique) <= 8 else f" (+{len(unique) - 8} more)"
    raise PublicArtifactError(f"public artifact privacy scan failed: {summary}{suffix}")


def _public_trace_id(id_key: bytes, raw_query_id: str, ordinal: int) -> str:
    message = ordinal.to_bytes(8, "big") + b"\0" + raw_query_id.encode("utf-8")
    digest = hmac.new(id_key, message, hashlib.sha256).hexdigest()
    return f"pt_{digest[:32]}"


def _public_membership_id(id_key: bytes, raw_membership_id: str) -> str:
    """Irreversibly remap a task membership ID while preserving equality."""
    digest = hmac.new(
        id_key,
        f"utility-membership\x00{raw_membership_id}".encode(),
        hashlib.sha256,
    ).hexdigest()
    return f"pm_{digest[:32]}"


def _sanitize_turn(source: Mapping[str, Any], *, location: str) -> dict[str, Any]:
    _check_exact_fields(source, _EXPECTED_TURN_SOURCE_FIELDS, location=location)
    tools_called = source["tools_called"]
    if not isinstance(tools_called, list) or any(not isinstance(item, str) for item in tools_called):
        raise PublicArtifactError(f"invalid tools_called at {location}")
    return {
        "turn_index": source["turn_index"],
        "input_tokens": source["input_tokens"],
        "output_tokens": source["output_tokens"],
        "tool_result_tokens": source["tool_result_tokens"],
        "tool_call_count": len(tools_called),
        "wall_clock_s": source["wall_clock_s"],
        "gpu_energy_joules": source["gpu_energy_joules"],
        "cpu_energy_joules": source["cpu_energy_joules"],
        "gpu_power_avg_watts": source["gpu_power_avg_watts"],
        "cpu_power_avg_watts": source["cpu_power_avg_watts"],
        "cost_usd": source["cost_usd"],
        "tool_cost_usd": source["tool_cost_usd"],
        "tool_cost_complete": source["tool_cost_complete"],
        "total_cost_usd": source["total_cost_usd"],
        "legacy_lm_only_cost_usd": source["legacy_lm_only_cost_usd"],
        "complete_total_cost_usd": source["complete_total_cost_usd"],
        "dram_energy_joules": source["dram_energy_joules"],
        "peak_watts": source["peak_watts"],
        "reasoning_tokens": source["reasoning_tokens"],
        "cached_tokens": source["cached_tokens"],
        "has_parallel_tools": source["has_parallel_tools"],
        "shared_device_warning": source["shared_device_warning"],
    }


def _sanitize_trace(
    trace: QueryTrace,
    *,
    ordinal: int,
    id_key: bytes,
    policy: PublicArtifactPolicy,
) -> dict[str, Any]:
    source = trace.to_dict()
    if not isinstance(source, Mapping):
        raise PublicArtifactError(f"trace {ordinal} did not serialize to an object")
    _check_exact_fields(source, _EXPECTED_QUERY_SOURCE_FIELDS, location=f"trace[{ordinal}]")
    raw_query_id = source["query_id"]
    if not isinstance(raw_query_id, str):
        raise PublicArtifactError(f"query_id must be a string at trace[{ordinal}]")
    source_turns = source["turns"]
    if not isinstance(source_turns, list):
        raise PublicArtifactError(f"turns must be a list at trace[{ordinal}]")
    raw_membership_id = source["utility_task_membership_id"]
    if raw_membership_id is not None and not isinstance(raw_membership_id, str):
        raise PublicArtifactError(f"utility_task_membership_id must be a string or null at trace[{ordinal}]")

    public: dict[str, Any] = {
        "schema_version": PUBLIC_TRACE_SCHEMA_VERSION,
        "public_trace_id": _public_trace_id(id_key, raw_query_id, ordinal),
        "workload_type": source["workload_type"],
        "accelerator_device_coverage_complete": source["accelerator_device_coverage_complete"],
        "turns": [
            _sanitize_turn(turn, location=f"trace[{ordinal}].turns[{index}]") for index, turn in enumerate(source_turns)
        ],
        "num_turns": source["num_turns"],
        "total_input_tokens": source["total_input_tokens"],
        "total_output_tokens": source["total_output_tokens"],
        "total_tokens": source["total_tokens"],
        "tool_call_count": source["tool_call_count"],
        "total_wall_clock_s": source["total_wall_clock_s"],
        "completed": source["completed"],
        "timed_out": source["timed_out"],
        "query_gpu_energy_joules": source["query_gpu_energy_joules"],
        "query_cpu_energy_joules": source["query_cpu_energy_joules"],
        "query_gpu_power_avg_watts": source["query_gpu_power_avg_watts"],
        "query_cpu_power_avg_watts": source["query_cpu_power_avg_watts"],
        "judge_gpu_energy_joules": source["judge_gpu_energy_joules"],
        "judge_cpu_energy_joules": source["judge_cpu_energy_joules"],
        "judge_gpu_power_avg_watts": source["judge_gpu_power_avg_watts"],
        "judge_cpu_power_avg_watts": source["judge_cpu_power_avg_watts"],
        "judge_wall_clock_s": source["judge_wall_clock_s"],
        "total_gpu_energy_joules": source["total_gpu_energy_joules"],
        "complete_total_gpu_energy_joules": source["complete_total_gpu_energy_joules"],
        "total_cpu_energy_joules": source["total_cpu_energy_joules"],
        "complete_total_cpu_energy_joules": source["complete_total_cpu_energy_joules"],
        "total_task_gpu_energy_joules": source["total_task_gpu_energy_joules"],
        "total_task_cpu_energy_joules": source["total_task_cpu_energy_joules"],
        "complete_total_task_gpu_energy_joules": source["complete_total_task_gpu_energy_joules"],
        "complete_total_task_cpu_energy_joules": source["complete_total_task_cpu_energy_joules"],
        "query_mbu_avg_pct": source["query_mbu_avg_pct"],
        "query_mbu_max_pct": source["query_mbu_max_pct"],
        "throughput_tokens_per_sec": source["throughput_tokens_per_sec"],
        "energy_per_output_token_joules": source["energy_per_output_token_joules"],
        "complete_energy_per_output_token_joules": source["complete_energy_per_output_token_joules"],
        "total_cost_usd": source["total_cost_usd"],
        "legacy_lm_only_cost_usd": source["legacy_lm_only_cost_usd"],
        "complete_total_cost_usd": source["complete_total_cost_usd"],
        "cost_known_turns": source["cost_known_turns"],
        "cost_unknown_turns": source["cost_unknown_turns"],
        "cost_coverage": source["cost_coverage"],
        "is_resolved": source["is_resolved"],
        "utility_complete": source["utility_complete"],
        "utility_provenance": source["utility_provenance"],
        "utility_calibration_id": source["utility_calibration_id"],
        "utility_verifier_id": source["utility_verifier_id"],
        "utility_verifier_version": source["utility_verifier_version"],
        "utility_task_manifest_id": source["utility_task_manifest_id"],
        "utility_task_manifest_size": source["utility_task_manifest_size"],
        "public_utility_task_membership_id": (
            _public_membership_id(id_key, raw_membership_id) if raw_membership_id is not None else None
        ),
        "lm_energy_measurable": source["lm_energy_measurable"],
        "shared_device_warning": source["shared_device_warning"],
        "gpu_energy_complete": source["gpu_energy_complete"],
        "gpu_energy_provenance": source["gpu_energy_provenance"],
        "accuracy_score": source["accuracy_score"],
        "n_retries": source["n_retries"],
    }
    for content_field in policy.retained_content_fields:
        content = source[content_field]
        if not isinstance(content, str):
            raise PublicArtifactError(f"{content_field} must be a string")
        if len(content.encode("utf-8")) > _MAX_CONTENT_FIELD_BYTES:
            raise PublicArtifactError(f"{content_field} exceeds the public artifact size limit")
        public[content_field] = content

    _raise_for_findings(scan_sensitive_data(public, location=f"trace[{ordinal}]"))
    public["record_sha256"] = _sha256(_canonical_json_bytes(public))
    return public


def _policy_sha256(policy: PublicArtifactPolicy) -> str:
    return _sha256(
        _canonical_json_bytes(
            {
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "id_remap": "hmac-sha256-secret-key-ordinal-v1",
                "retained_content_fields": sorted(policy.retained_content_fields),
                "sanitizer_version": SANITIZER_VERSION,
            }
        )
    )


def _provenance_dict(provenance: ArtifactProvenance) -> dict[str, Any]:
    # Frozen dataclasses can still be mutated through low-level reflection.
    # Reconstruct the value at the serialization boundary so export and later
    # manifest validation enforce the same exact runtime schema.
    value = asdict(ArtifactProvenance(**asdict(provenance)))
    _check_exact_fields(value, _PROVENANCE_FIELDS, location="manifest.provenance")
    _raise_for_findings(scan_sensitive_data(value, location="manifest.provenance"))
    return value


def _write_new_file(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def create_public_trace_artifact(
    traces: Iterable[QueryTrace],
    output_dir: str | Path,
    *,
    provenance: ArtifactProvenance,
    policy: PublicArtifactPolicy | None = None,
    id_key: bytes | None = None,
) -> Path:
    """Create and validate a public, metadata-only trace artifact by default.

    ``id_key`` is intentionally never written to the artifact.  Supplying it
    supports controlled reproducibility tests; production callers should let
    the function generate a fresh 256-bit key and retain any raw-to-public ID
    mapping only in an authorized private system.
    """

    resolved_policy = policy or PublicArtifactPolicy()
    resolved_key = id_key or secrets.token_bytes(32)
    if not isinstance(resolved_key, bytes) or len(resolved_key) < 32:
        raise PublicArtifactError("id_key must contain at least 32 bytes")

    destination = Path(output_dir)
    if destination.exists():
        raise PublicArtifactError(f"artifact destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    trace_records = [
        _sanitize_trace(
            trace,
            ordinal=ordinal,
            id_key=resolved_key,
            policy=resolved_policy,
        )
        for ordinal, trace in enumerate(traces)
    ]
    trace_bytes = b"".join(_canonical_json_bytes(record) for record in trace_records)
    privacy_report = {
        "scanner_version": SCANNER_VERSION,
        "status": "passed",
        "finding_count": 0,
        "checks": [
            "exact-file-allowlist",
            "exact-field-allowlist",
            "credential-patterns",
            "private-addresses-and-hostnames",
            "absolute-paths",
            "content-hashes",
        ],
    }
    privacy_bytes = _canonical_json_bytes(privacy_report)
    files = [
        {
            "path": PRIVACY_REPORT_FILENAME,
            "bytes": len(privacy_bytes),
            "sha256": _sha256(privacy_bytes),
        },
        {
            "path": TRACES_FILENAME,
            "bytes": len(trace_bytes),
            "sha256": _sha256(trace_bytes),
        },
    ]
    manifest = {
        "artifact_kind": ARTIFACT_KIND,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "sanitizer": {"name": SANITIZER_NAME, "version": SANITIZER_VERSION},
        "sanitizer_marker": SANITIZER_MARKER,
        "content_retention_mode": resolved_policy.content_retention_mode,
        "policy_sha256": _policy_sha256(resolved_policy),
        "trace_count": len(trace_records),
        "provenance": _provenance_dict(provenance),
        "files": files,
    }
    manifest_bytes = _canonical_json_bytes(manifest)
    _raise_for_findings(scan_sensitive_data(manifest, location="manifest"))

    staging = Path(tempfile.mkdtemp(prefix=".ipj-public-artifact-", dir=destination.parent))
    try:
        _write_new_file(staging / TRACES_FILENAME, trace_bytes)
        _write_new_file(staging / PRIVACY_REPORT_FILENAME, privacy_bytes)
        _write_new_file(staging / MANIFEST_FILENAME, manifest_bytes)
        validate_public_trace_artifact(staging)
        try:
            staging.rename(destination)
        except FileExistsError as exc:
            raise PublicArtifactError(f"artifact destination appeared during creation: {destination}") from exc
    finally:
        if staging.exists():
            for child in staging.iterdir():
                child.unlink(missing_ok=True)
            staging.rmdir()
    return destination


def _parse_retention_mode(mode: Any) -> PublicArtifactPolicy:
    if mode == "metadata_only":
        return PublicArtifactPolicy()
    if isinstance(mode, str) and mode.startswith("explicit:"):
        names = frozenset(name for name in mode.removeprefix("explicit:").split(",") if name)
        return PublicArtifactPolicy(retained_content_fields=names)
    raise PublicArtifactError("invalid content_retention_mode")


def _validate_public_turn(turn: Any, *, location: str) -> None:
    if not isinstance(turn, Mapping):
        raise PublicArtifactError(f"turn must be an object at {location}")
    _check_exact_fields(turn, _PUBLIC_TURN_FIELDS, location=location)
    _canonical_json_bytes(turn)


def _validate_public_record(record: Any, *, policy: PublicArtifactPolicy, location: str) -> str:
    if not isinstance(record, Mapping):
        raise PublicArtifactError(f"trace must be an object at {location}")
    expected = _PUBLIC_QUERY_FIELDS | policy.retained_content_fields
    _check_exact_fields(record, expected, location=location)
    if record["schema_version"] != PUBLIC_TRACE_SCHEMA_VERSION:
        raise PublicArtifactError(f"unsupported public trace schema at {location}")
    public_trace_id = record["public_trace_id"]
    if not isinstance(public_trace_id, str) or not re.fullmatch(r"pt_[0-9a-f]{32}", public_trace_id):
        raise PublicArtifactError(f"invalid public_trace_id at {location}")
    public_membership_id = record["public_utility_task_membership_id"]
    if public_membership_id is not None and (
        not isinstance(public_membership_id, str) or not re.fullmatch(r"pm_[0-9a-f]{32}", public_membership_id)
    ):
        raise PublicArtifactError(f"invalid public utility membership ID at {location}")
    turns = record["turns"]
    if not isinstance(turns, list):
        raise PublicArtifactError(f"turns must be a list at {location}")
    for index, turn in enumerate(turns):
        _validate_public_turn(turn, location=f"{location}.turns[{index}]")
    if record["num_turns"] != len(turns):
        raise PublicArtifactError(f"num_turns does not match turns at {location}")
    for content_field in policy.retained_content_fields:
        content = record[content_field]
        if not isinstance(content, str) or len(content.encode("utf-8")) > _MAX_CONTENT_FIELD_BYTES:
            raise PublicArtifactError(f"invalid retained content at {location}.{content_field}")

    supplied_hash = record["record_sha256"]
    if not isinstance(supplied_hash, str) or not _SHA256_PATTERN.fullmatch(supplied_hash):
        raise PublicArtifactError(f"invalid record hash at {location}")
    unhashed = dict(record)
    del unhashed["record_sha256"]
    if not hmac.compare_digest(supplied_hash, _sha256(_canonical_json_bytes(unhashed))):
        raise PublicArtifactError(f"record hash mismatch at {location}")
    _raise_for_findings(scan_sensitive_data(record, location=location))
    _canonical_json_bytes(record)
    return public_trace_id


def _read_artifact_file(root: Path, name: str) -> bytes:
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise PublicArtifactError(f"artifact member must be a regular file: {name}")
    size = path.stat().st_size
    if size > _MAX_ARTIFACT_FILE_BYTES:
        raise PublicArtifactError(f"artifact member exceeds size limit: {name}")
    return path.read_bytes()


def validate_public_trace_artifact(artifact_dir: str | Path) -> Path:
    """Validate a sanitizer-marked artifact before any remote upload."""

    root = Path(artifact_dir)
    if root.is_symlink() or not root.is_dir():
        raise PublicArtifactError("public trace artifact must be a regular directory")
    children = tuple(root.iterdir())
    names = frozenset(child.name for child in children)
    if names != _ALLOWED_ARTIFACT_FILES:
        unknown = sorted(names - _ALLOWED_ARTIFACT_FILES)
        missing = sorted(_ALLOWED_ARTIFACT_FILES - names)
        raise PublicArtifactError(f"artifact file allowlist mismatch: unknown={unknown}; missing={missing}")

    manifest_bytes = _read_artifact_file(root, MANIFEST_FILENAME)
    manifest = _load_json_bytes(manifest_bytes, location=MANIFEST_FILENAME)
    if not isinstance(manifest, Mapping):
        raise PublicArtifactError("artifact manifest must be an object")
    _check_exact_fields(manifest, _MANIFEST_FIELDS, location="manifest")
    if manifest["artifact_kind"] != ARTIFACT_KIND:
        raise PublicArtifactError("unexpected artifact kind")
    if manifest["artifact_schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise PublicArtifactError("unsupported artifact schema version")
    if manifest["sanitizer_marker"] != SANITIZER_MARKER:
        raise PublicArtifactError("artifact is not marked by the public trace sanitizer")
    if manifest["sanitizer"] != {"name": SANITIZER_NAME, "version": SANITIZER_VERSION}:
        raise PublicArtifactError("unsupported sanitizer identity or version")
    policy = _parse_retention_mode(manifest["content_retention_mode"])
    expected_policy_hash = _policy_sha256(policy)
    if not isinstance(manifest["policy_sha256"], str) or not hmac.compare_digest(
        manifest["policy_sha256"], expected_policy_hash
    ):
        raise PublicArtifactError("sanitizer policy hash mismatch")

    provenance = manifest["provenance"]
    if not isinstance(provenance, Mapping):
        raise PublicArtifactError("manifest provenance must be an object")
    _check_exact_fields(provenance, _PROVENANCE_FIELDS, location="manifest.provenance")
    ArtifactProvenance(**provenance)

    file_rows = manifest["files"]
    if not isinstance(file_rows, list) or len(file_rows) != 2:
        raise PublicArtifactError("manifest must describe exactly two payload files")
    file_map: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(file_rows):
        if not isinstance(row, Mapping):
            raise PublicArtifactError(f"invalid file manifest row at index {index}")
        _check_exact_fields(row, _FILE_MANIFEST_FIELDS, location=f"manifest.files[{index}]")
        name = row["path"]
        if name not in {PRIVACY_REPORT_FILENAME, TRACES_FILENAME} or name in file_map:
            raise PublicArtifactError("file manifest path is unknown or duplicated")
        if not isinstance(row["bytes"], int) or isinstance(row["bytes"], bool) or row["bytes"] < 0:
            raise PublicArtifactError(f"invalid byte count for {name}")
        if not isinstance(row["sha256"], str) or not _SHA256_PATTERN.fullmatch(row["sha256"]):
            raise PublicArtifactError(f"invalid file hash for {name}")
        file_map[name] = row

    payloads: dict[str, bytes] = {}
    for name in (PRIVACY_REPORT_FILENAME, TRACES_FILENAME):
        data = _read_artifact_file(root, name)
        row = file_map[name]
        if row["bytes"] != len(data) or not hmac.compare_digest(row["sha256"], _sha256(data)):
            raise PublicArtifactError(f"content hash mismatch for {name}")
        payloads[name] = data

    report = _load_json_bytes(payloads[PRIVACY_REPORT_FILENAME], location=PRIVACY_REPORT_FILENAME)
    if not isinstance(report, Mapping):
        raise PublicArtifactError("privacy report must be an object")
    _check_exact_fields(report, _PRIVACY_REPORT_FIELDS, location="privacy_report")
    if report["scanner_version"] != SCANNER_VERSION or report["status"] != "passed" or report["finding_count"] != 0:
        raise PublicArtifactError("privacy report is not a passing current-version report")

    traces_data = payloads[TRACES_FILENAME]
    public_ids: set[str] = set()
    record_count = 0
    for line_number, line in enumerate(traces_data.splitlines(), start=1):
        if not line.strip():
            raise PublicArtifactError(f"blank JSONL record at line {line_number}")
        record = _load_json_bytes(line, location=f"{TRACES_FILENAME}:{line_number}")
        public_id = _validate_public_record(
            record,
            policy=policy,
            location=f"public_traces[{line_number - 1}]",
        )
        if public_id in public_ids:
            raise PublicArtifactError("duplicate public_trace_id")
        public_ids.add(public_id)
        record_count += 1
    if not isinstance(manifest["trace_count"], int) or isinstance(manifest["trace_count"], bool):
        raise PublicArtifactError("trace_count must be an integer")
    if manifest["trace_count"] != record_count:
        raise PublicArtifactError("trace_count does not match public trace payload")

    _raise_for_findings(scan_sensitive_data(manifest, location="manifest"))
    _raise_for_findings(scan_sensitive_data(report, location="privacy_report"))
    return root


def copy_validated_public_trace_artifact(artifact_dir: str | Path, destination: str | Path) -> Path:
    """Copy only validated bytes into a new private upload snapshot.

    The source directory is untrusted and may be modified concurrently.  This
    function enumerates it once, captures only the three allowlisted regular
    files, writes those bytes into a mode-0700 directory, and validates the
    captured copy.  A remote uploader can therefore operate on the snapshot
    without a validate-then-upload race against the original directory.
    """

    source = Path(artifact_dir)
    target = Path(destination)
    if source.is_symlink() or not source.is_dir():
        raise PublicArtifactError("public trace artifact must be a regular directory")
    source_names = frozenset(child.name for child in source.iterdir())
    if source_names != _ALLOWED_ARTIFACT_FILES:
        unknown = sorted(source_names - _ALLOWED_ARTIFACT_FILES)
        missing = sorted(_ALLOWED_ARTIFACT_FILES - source_names)
        raise PublicArtifactError(f"artifact file allowlist mismatch: unknown={unknown}; missing={missing}")
    if target.exists():
        raise PublicArtifactError(f"upload snapshot destination already exists: {target}")

    # Capture fixed names, never a recursive source-directory copy.  Any
    # concurrent mutation yields either a self-consistent safe snapshot or a
    # hash/schema/privacy validation failure below.
    captured = {name: _read_artifact_file(source, name) for name in sorted(_ALLOWED_ARTIFACT_FILES)}
    target.mkdir(parents=True, mode=0o700)
    try:
        for name, data in captured.items():
            _write_new_file(target / name, data)
        validate_public_trace_artifact(target)
    except Exception:
        for child in target.iterdir():
            child.unlink(missing_ok=True)
        target.rmdir()
        raise
    return target


__all__ = [
    "ARTIFACT_KIND",
    "ARTIFACT_SCHEMA_VERSION",
    "ArtifactProvenance",
    "PublicArtifactError",
    "PublicArtifactPolicy",
    "SensitiveDataFinding",
    "copy_validated_public_trace_artifact",
    "create_public_trace_artifact",
    "scan_sensitive_data",
    "validate_public_trace_artifact",
]
