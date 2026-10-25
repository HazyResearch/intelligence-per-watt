"""Security and compatibility tests for public trace artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from ipw.execution.exporters import export_jsonl, export_public_trace_artifact
from ipw.execution.public_artifact import (
    ArtifactProvenance,
    PublicArtifactError,
    PublicArtifactPolicy,
    create_public_trace_artifact,
    scan_sensitive_data,
    validate_public_trace_artifact,
)
from ipw.execution.trace import QueryTrace, TurnTrace

_ID_KEY = b"test-only-public-id-key-32-bytes"
_PROVENANCE_STRING_FIELDS = (
    "benchmark_id",
    "benchmark_version",
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
)


def _provenance() -> ArtifactProvenance:
    return ArtifactProvenance(
        code_revision="deadbeef",
        code_dirty=False,
        data_license="Apache-2.0",
        model_id="example/model",
        benchmark_id="gaia",
        harness_id="react",
        verifier_id="task-native",
    )


def _trace(*, query_text: str = "private prompt", response_text: str = "private response") -> QueryTrace:
    return QueryTrace(
        query_id="raw-query-0001",
        workload_type="gaia",
        query_text=query_text,
        response_text=response_text,
        turns=[
            TurnTrace(
                turn_index=0,
                input_tokens=10,
                output_tokens=4,
                tool_result_tokens=2,
                tools_called=["private_cluster_tool"],
                tool_latencies_s={"private_cluster_tool": 0.25},
                wall_clock_s=1.5,
                error="raw tool error on /home/alice/private.txt",
                gpu_energy_joules=12.0,
                cost_usd=0.01,
                tool_cost_usd=0.02,
                tool_cost_complete=True,
            )
        ],
        total_wall_clock_s=1.5,
        completed=True,
        is_resolved=True,
        utility_complete=True,
        utility_provenance="task_native_verifier",
        utility_verifier_id="gaia-native-verifier",
        utility_verifier_version="1.0.0",
        utility_task_manifest_id="gaia-fixture-v1",
        utility_task_membership_id="private-gaia-task-0001",
        utility_task_manifest_size=1,
        gpu_energy_complete=True,
        gpu_energy_provenance="measured_dedicated",
        score_metadata={"private_host": "compute-01.internal"},
        accuracy_metadata={"raw_output": "do not publish"},
    )


def _read_records(artifact: Path) -> list[dict]:
    return [json.loads(line) for line in (artifact / "public_traces.jsonl").read_text().splitlines()]


def test_default_artifact_is_metadata_only_hashed_and_validated(tmp_path: Path) -> None:
    artifact = export_public_trace_artifact(
        [_trace()],
        tmp_path / "public",
        provenance=_provenance(),
        id_key=_ID_KEY,
    )

    assert validate_public_trace_artifact(artifact) == artifact
    assert {path.name for path in artifact.iterdir()} == {
        "artifact_manifest.json",
        "privacy_report.json",
        "public_traces.jsonl",
    }
    record = _read_records(artifact)[0]
    serialized_bundle = "".join(path.read_text() for path in artifact.iterdir())
    assert record["public_trace_id"].startswith("pt_")
    assert record["tool_call_count"] == 1
    assert record["turns"][0]["tool_call_count"] == 1
    assert record["legacy_lm_only_cost_usd"] == pytest.approx(0.01)
    assert record["total_cost_usd"] == pytest.approx(0.01)
    assert record["complete_total_cost_usd"] == pytest.approx(0.03)
    assert record["public_utility_task_membership_id"].startswith("pm_")
    assert record["turns"][0]["legacy_lm_only_cost_usd"] == pytest.approx(0.01)
    assert "query_text" not in record
    assert "response_text" not in record
    assert "tools_called" not in record["turns"][0]
    assert "tool_latencies_s" not in record["turns"][0]
    assert "error" not in record["turns"][0]
    assert "score_metadata" not in record
    assert "accuracy_metadata" not in record
    assert "raw-query-0001" not in serialized_bundle
    assert "private-gaia-task-0001" not in serialized_bundle
    assert "private prompt" not in serialized_bundle
    assert "private response" not in serialized_bundle
    assert "private_cluster_tool" not in serialized_bundle
    assert "/home/alice" not in serialized_bundle

    manifest = json.loads((artifact / "artifact_manifest.json").read_text())
    assert manifest["sanitizer_marker"]
    assert manifest["provenance"]["code_revision"] == "deadbeef"
    assert manifest["content_retention_mode"] == "metadata_only"
    for row in manifest["files"]:
        payload = (artifact / row["path"]).read_bytes()
        assert row["bytes"] == len(payload)
        assert row["sha256"] == hashlib.sha256(payload).hexdigest()


def test_raw_local_export_remains_available_and_content_preserving(tmp_path: Path) -> None:
    destination = export_jsonl([_trace()], tmp_path / "raw.jsonl")

    raw = json.loads(destination.read_text())
    assert raw["query_id"] == "raw-query-0001"
    assert raw["query_text"] == "private prompt"
    assert raw["response_text"] == "private response"
    assert raw["turns"][0]["error"].startswith("raw tool error")


def test_membership_remapping_preserves_equality_without_exposing_raw_ids(
    tmp_path: Path,
) -> None:
    artifact = export_public_trace_artifact(
        [_trace(), _trace()],
        tmp_path / "public-memberships",
        provenance=_provenance(),
        id_key=_ID_KEY,
    )

    records = _read_records(artifact)
    remapped = [record["public_utility_task_membership_id"] for record in records]
    assert remapped[0] == remapped[1]
    assert all(value.startswith("pm_") for value in remapped)
    assert "private-gaia-task-0001" not in (artifact / "public_traces.jsonl").read_text()


def test_explicit_clean_content_allowlist_is_recorded(tmp_path: Path) -> None:
    policy = PublicArtifactPolicy(retained_content_fields=frozenset({"query_text", "response_text"}))
    artifact = create_public_trace_artifact(
        [_trace(query_text="public benchmark prompt", response_text="verified answer")],
        tmp_path / "public",
        provenance=_provenance(),
        policy=policy,
        id_key=_ID_KEY,
    )

    record = _read_records(artifact)[0]
    manifest = json.loads((artifact / "artifact_manifest.json").read_text())
    assert record["query_text"] == "public benchmark prompt"
    assert record["response_text"] == "verified answer"
    assert manifest["content_retention_mode"] == "explicit:query_text,response_text"
    validate_public_trace_artifact(artifact)


@pytest.mark.parametrize(
    ("content", "category"),
    [
        ("Authorization: Bearer abcdefghijklmnopqrstuvwxyz", "bearer_credential"),
        ("key=sk-proj-abcdefghijklmnop", "openai_credential"),
        ("read /home/alice/private/trace.json", "absolute_path"),
        ("connect to 10.42.0.9", "ip_address"),
        ("connect to 192.222.48.102", "ip_address"),
        ("connect to fd00::1", "ip_address"),
        ("connect to fe80::1", "ip_address"),
        ("connect to ::1", "ip_address"),
        ("compute-01.internal", "private_hostname"),
        ("compute-01", "private_hostname"),
        ("GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "hardware_unique_identifier"),
        ("serial=ABCDEF123456", "hardware_unique_identifier"),
        ("alice@example.com", "email_or_username"),
    ],
)
def test_explicit_content_still_fails_privacy_scan(tmp_path: Path, content: str, category: str) -> None:
    with pytest.raises(PublicArtifactError, match=category) as error:
        create_public_trace_artifact(
            [_trace(query_text=content)],
            tmp_path / "public",
            provenance=_provenance(),
            policy=PublicArtifactPolicy(retained_content_fields=frozenset({"query_text"})),
            id_key=_ID_KEY,
        )
    assert content not in str(error.value)


def test_default_policy_drops_contaminated_content_before_scanning(tmp_path: Path) -> None:
    artifact = create_public_trace_artifact(
        [_trace(query_text="secret=sk-proj-abcdefghijklmnop at /home/alice")],
        tmp_path / "public",
        provenance=_provenance(),
        id_key=_ID_KEY,
    )

    validate_public_trace_artifact(artifact)
    assert "sk-proj" not in "".join(path.read_text() for path in artifact.iterdir())


def test_scanner_reports_locations_but_never_matched_values() -> None:
    secret = "sk-proj-abcdefghijklmnop"
    findings = scan_sensitive_data({"content": secret})

    assert any(finding.category == "openai_credential" for finding in findings)
    assert all(secret not in repr(finding) for finding in findings)


def test_unknown_future_source_field_fails_closed(tmp_path: Path) -> None:
    class FutureTrace(QueryTrace):
        def to_dict(self) -> dict:
            value = super().to_dict()
            value["new_provider_payload"] = "new unclassified value"
            return value

    trace = FutureTrace(query_id="q", workload_type="gaia")
    with pytest.raises(PublicArtifactError, match="unknown=.*new_provider_payload"):
        create_public_trace_artifact(
            [trace],
            tmp_path / "public",
            provenance=_provenance(),
            id_key=_ID_KEY,
        )


def test_artifact_tampering_fails_before_upload(tmp_path: Path) -> None:
    artifact = create_public_trace_artifact(
        [_trace()],
        tmp_path / "public",
        provenance=_provenance(),
        id_key=_ID_KEY,
    )
    trace_file = artifact / "public_traces.jsonl"
    trace_file.write_bytes(trace_file.read_bytes().replace(b'"gaia"', b'"GAIA"'))

    with pytest.raises(PublicArtifactError, match="content hash mismatch"):
        validate_public_trace_artifact(artifact)


def test_extra_file_fails_exact_file_allowlist(tmp_path: Path) -> None:
    artifact = create_public_trace_artifact(
        [_trace()],
        tmp_path / "public",
        provenance=_provenance(),
        id_key=_ID_KEY,
    )
    (artifact / "raw_response.txt").write_text("raw")

    with pytest.raises(PublicArtifactError, match="file allowlist mismatch"):
        validate_public_trace_artifact(artifact)


def test_dirty_or_underspecified_provenance_is_rejected() -> None:
    with pytest.raises(PublicArtifactError, match="clean source revision"):
        ArtifactProvenance(
            code_revision="deadbeef",
            code_dirty=True,
            data_license="Apache-2.0",
        )
    with pytest.raises(PublicArtifactError, match="data_license"):
        ArtifactProvenance(code_revision="deadbeef", code_dirty=False, data_license=" ")
    with pytest.raises(PublicArtifactError, match="code_revision"):
        ArtifactProvenance(code_revision="main", code_dirty=False, data_license="Apache-2.0")


@pytest.mark.parametrize("invalid", [0, 1, None, "false", [], {}])
def test_code_dirty_requires_an_exact_boolean(invalid: object) -> None:
    values = asdict(_provenance())
    values["code_dirty"] = invalid

    with pytest.raises(PublicArtifactError, match="code_dirty must be a boolean"):
        ArtifactProvenance(**values)


@pytest.mark.parametrize("field_name", _PROVENANCE_STRING_FIELDS)
@pytest.mark.parametrize("invalid", [False, 0, [], {}])
def test_provenance_strings_reject_unexpected_shapes(
    field_name: str,
    invalid: object,
) -> None:
    values = asdict(_provenance())
    values[field_name] = invalid

    with pytest.raises(PublicArtifactError, match=field_name):
        ArtifactProvenance(**values)


@pytest.mark.parametrize("field_name", _PROVENANCE_STRING_FIELDS)
@pytest.mark.parametrize("invalid", [" ", "x" * 10_000])
def test_provenance_strings_reject_blank_or_unbounded_values(
    field_name: str,
    invalid: str,
) -> None:
    values = asdict(_provenance())
    values[field_name] = invalid

    with pytest.raises(PublicArtifactError, match=field_name):
        ArtifactProvenance(**values)


def test_export_revalidates_a_reflectively_mutated_provenance(tmp_path: Path) -> None:
    provenance = _provenance()
    object.__setattr__(provenance, "model_id", {"unexpected": "shape"})

    with pytest.raises(PublicArtifactError, match="model_id"):
        create_public_trace_artifact(
            [_trace()],
            tmp_path / "public",
            provenance=provenance,
            id_key=_ID_KEY,
        )


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    [("code_dirty", 0), ("model_id", {"unexpected": "shape"})],
)
def test_validation_rejects_tampered_provenance_shapes(
    tmp_path: Path,
    field_name: str,
    invalid: object,
) -> None:
    artifact = create_public_trace_artifact(
        [_trace()],
        tmp_path / field_name,
        provenance=_provenance(),
        id_key=_ID_KEY,
    )
    manifest_path = artifact / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["provenance"][field_name] = invalid
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")

    with pytest.raises(PublicArtifactError, match=field_name):
        validate_public_trace_artifact(artifact)


def test_destination_and_id_key_fail_closed(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(PublicArtifactError, match="already exists"):
        create_public_trace_artifact([_trace()], existing, provenance=_provenance(), id_key=_ID_KEY)
    with pytest.raises(PublicArtifactError, match="at least 32 bytes"):
        create_public_trace_artifact(
            [_trace()],
            tmp_path / "new",
            provenance=_provenance(),
            id_key=b"short",
        )
