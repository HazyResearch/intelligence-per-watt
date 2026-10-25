from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from ipw.execution.trace import QueryTrace

from ipj.registry.artifact import (
    ArtifactProvenance,
    PublicArtifactError,
    create_public_trace_artifact,
)
from ipj.registry.store import push_traces, trace_repo_path


def test_trace_repo_path():
    assert trace_repo_path("deadbeef") == "traces/deadbeef"


@pytest.mark.parametrize("signature", ["../private", "ABCDEF12", "dead/beef", "short"])
def test_trace_repo_path_rejects_unsafe_signature(signature):
    with pytest.raises(ValueError, match="energy_sig"):
        trace_repo_path(signature)


def test_push_traces_rejects_raw_directory_before_hub_client(tmp_path):
    (tmp_path / "raw_trace.json").write_text('{"prompt":"private"}')

    with pytest.raises(PublicArtifactError, match="file allowlist mismatch"):
        push_traces(tmp_path, "deadbeef")


def test_push_traces_uploads_only_validated_public_artifact(tmp_path, monkeypatch):
    artifact = create_public_trace_artifact(
        [QueryTrace(query_id="private-id", workload_type="gaia")],
        tmp_path / "public",
        provenance=ArtifactProvenance(
            code_revision="deadbeef",
            code_dirty=False,
            data_license="Apache-2.0",
        ),
        id_key=b"test-only-public-id-key-32-bytes",
    )
    calls = []
    uploaded_payloads = {}

    class FakeApi:
        def __init__(self, *, token=None):
            self.token = token

        def upload_folder(self, **kwargs):
            # Simulate an attacker changing the original directory exactly at
            # the validation/upload boundary. The uploader must use a private
            # fixed-file snapshot, not this now-contaminated source.
            (artifact / "late_raw_secret.txt").write_text("sk-proj-abcdefghijklmnop")
            upload_dir = Path(kwargs["folder_path"])
            assert upload_dir != artifact
            assert {path.name for path in upload_dir.iterdir()} == {
                "artifact_manifest.json",
                "privacy_report.json",
                "public_traces.jsonl",
            }
            uploaded_payloads.update({path.name: path.read_bytes() for path in upload_dir.iterdir()})
            calls.append(kwargs)
            return SimpleNamespace(oid="revision123")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    uri = push_traces(artifact, "deadbeef", repo_id="example/public")

    assert uri == "hf://example/public/traces/deadbeef@revision123"
    assert len(calls) == 1
    assert calls[0]["folder_path"] != str(artifact)
    assert calls[0]["path_in_repo"] == "traces/deadbeef"
    assert calls[0]["repo_id"] == "example/public"
    assert calls[0]["repo_type"] == "dataset"
    assert "late_raw_secret.txt" not in uploaded_payloads
    assert uploaded_payloads["public_traces.jsonl"] == (artifact / "public_traces.jsonl").read_bytes()
