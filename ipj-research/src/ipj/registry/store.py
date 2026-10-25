from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

from .record import RunRecord

REPO_ID = "ScalingIntelligence/ipj-results"
INDEX_FILE = "index.jsonl"
PUBLIC_INDEX_FILE = "public-index.jsonl"
_ENERGY_SIGNATURE_PATTERN = re.compile(r"^[0-9a-f]{8,64}$")


def _token(token: str | None) -> str | None:
    return token or os.environ.get("HF_TOKEN")


def index_to_jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(r, default=str) + "\n" for r in rows)


def collapse_rows(rows: list[dict]) -> list[dict]:
    """Keep the last row per (energy_sig, created_at) — collapses exact re-pushes."""
    keyed: dict[tuple, dict] = {}
    for r in rows:
        keyed[(r.get("energy_sig"), r.get("created_at"))] = r
    return list(keyed.values())


def collapse_public_rows(rows: list[dict]) -> list[dict]:
    """Keep the last sanitized row for each immutable trace URI."""

    keyed: dict[tuple[object, object], dict] = {}
    for row in rows:
        keyed[(row.get("energy_sig"), row.get("trace_uri"))] = row
    return list(keyed.values())


def queue_pending(rec: RunRecord, pending_dir) -> Path:
    pending_dir = Path(pending_dir)
    pending_dir.mkdir(parents=True, exist_ok=True)
    name = f"{rec.energy_sig or 'unknown'}__{(rec.created_at or 'na').replace(':', '-')}.json"
    path = pending_dir / name
    path.write_text(json.dumps(rec.to_dict(), default=str))
    return path


def queue_pending_public(row: dict, pending_dir) -> Path:
    """Queue an already-sanitized public index row without raw run metadata."""

    from .public_record import public_index_to_jsonl, validate_public_run_record

    validate_public_run_record(row)
    pending_dir = Path(pending_dir)
    pending_dir.mkdir(parents=True, exist_ok=True)
    name = f"{row['energy_sig']}__{row['record_sha256'][:16]}.jsonl"
    path = pending_dir / name
    path.write_bytes(public_index_to_jsonl([row]))
    return path


def pull_index(*, repo_id: str = REPO_ID, token: str | None = None) -> list[dict]:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError, RepositoryNotFoundError

    try:
        path = hf_hub_download(repo_id, INDEX_FILE, repo_type="dataset", token=_token(token))
    except (EntryNotFoundError, RepositoryNotFoundError, HfHubHTTPError):
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def pull_public_index(*, repo_id: str = REPO_ID, token: str | None = None) -> list[dict]:
    """Read and validate the isolated allowlisted public index."""

    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError, RepositoryNotFoundError

    from .public_record import validate_public_run_record

    try:
        path = hf_hub_download(
            repo_id,
            PUBLIC_INDEX_FILE,
            repo_type="dataset",
            token=_token(token),
        )
    except (EntryNotFoundError, RepositoryNotFoundError, HfHubHTTPError):
        return []
    rows = [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
    for row in rows:
        validate_public_run_record(row)
    return rows


def push_record(
    rec: RunRecord,
    *,
    repo_id: str = REPO_ID,
    token: str | None = None,
    pending_dir="runs/_pending",
) -> str:
    """Append rec to the HF index. On any network/HF error, queue locally and return 'queued'."""
    from huggingface_hub import HfApi

    try:
        rows = pull_index(repo_id=repo_id, token=token)
        rows = collapse_rows([*rows, rec.to_dict()])
        HfApi(token=_token(token)).upload_file(
            path_or_fileobj=index_to_jsonl(rows).encode(),
            path_in_repo=INDEX_FILE,
            repo_id=repo_id,
            repo_type="dataset",
        )
        return "pushed"
    except Exception:
        queue_pending(rec, pending_dir)
        return "queued"


def push_public_record(
    rec: RunRecord,
    *,
    repo_id: str = REPO_ID,
    token: str | None = None,
    pending_dir="runs/_pending_public",
) -> str:
    """Append an allowlisted trace-linked row to the isolated public index.

    Sanitization occurs before the network error boundary. Invalid, dirty, or
    sensitive records raise and are never converted into a raw pending file.
    """

    from huggingface_hub import HfApi

    from .public_record import public_index_to_jsonl, sanitize_public_run_record

    public_row = sanitize_public_run_record(rec)
    try:
        rows = pull_public_index(repo_id=repo_id, token=token)
        rows = collapse_public_rows([*rows, public_row])
        HfApi(token=_token(token)).upload_file(
            path_or_fileobj=public_index_to_jsonl(rows),
            path_in_repo=PUBLIC_INDEX_FILE,
            repo_id=repo_id,
            repo_type="dataset",
        )
        return "pushed"
    except Exception:
        queue_pending_public(public_row, pending_dir)
        return "queued"


def ensure_repo(*, repo_id: str = REPO_ID, token: str | None = None) -> None:
    """Create the private dataset repo on HF Hub if it does not already exist."""
    from huggingface_hub import HfApi

    HfApi(token=_token(token)).create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)


def trace_repo_path(energy_sig: str) -> str:
    if not isinstance(energy_sig, str) or not _ENERGY_SIGNATURE_PATTERN.fullmatch(energy_sig):
        raise ValueError("energy_sig must be an 8-64 character lowercase hexadecimal digest")
    return f"traces/{energy_sig}"


def push_traces(trace_dir, energy_sig: str, *, repo_id: str = REPO_ID, token: str | None = None) -> str:
    """Upload a centrally sanitized public trace artifact.

    Raw local run directories are intentionally rejected.  The validator
    checks the sanitizer marker, exact file/field allowlists, content hashes,
    provenance, and the privacy scan before the Hub client is constructed.
    """

    from .artifact import copy_validated_public_trace_artifact

    repo_path = trace_repo_path(energy_sig)

    from huggingface_hub import HfApi

    # Upload a private, fixed-file snapshot. The source directory may be
    # mutable; no byte is sent from it after validation.
    with tempfile.TemporaryDirectory(prefix="ipj-public-upload-") as staging_root:
        artifact_dir = copy_validated_public_trace_artifact(trace_dir, Path(staging_root) / "artifact")
        commit = HfApi(token=_token(token)).upload_folder(
            folder_path=str(artifact_dir),
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="dataset",
        )
    rev = getattr(commit, "oid", None) or "main"
    return f"hf://{repo_id}/{repo_path}@{rev}"


def pull_traces(energy_sig: str, dest, *, repo_id: str = REPO_ID, token: str | None = None) -> Path:
    from huggingface_hub import snapshot_download

    local = snapshot_download(
        repo_id,
        repo_type="dataset",
        allow_patterns=[f"{trace_repo_path(energy_sig)}/*"],
        local_dir=str(dest),
        token=_token(token),
    )
    return Path(local) / trace_repo_path(energy_sig)
