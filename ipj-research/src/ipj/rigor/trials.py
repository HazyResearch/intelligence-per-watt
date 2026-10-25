"""Trial dataclass + harvest helpers for k>1 serving-run analysis.

Reward semantics (frozen; matches EXP-20260710-05/07 conventions):
  * SOLVED := verifier_result.rewards non-empty AND max(numeric values) >= 1.0.
  * has_exception := bool(exception_info) — the significance-table 'errored' convention.
  * INFRA := "Docker compose command failed" in exception_message OR "docker" in
    exception_type (case-insensitive). Excluded from numerator AND denominator upstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# Bootstrap / MC defaults — match EXP-20260710-07 significance table exactly.
SEED: int = 0
N_RESAMPLES: int = 10_000
MC_SEED: int = 20260711  # separate stream from the bootstrap CIs

# Infra-artifact string (docker compose stderr signature).
INFRA_MSG: str = "Docker compose command failed"

# Timezone-naive timestamp default (Pacific Daylight Time; tests can override).
PDT = timezone(timedelta(hours=-7))

# Cap on how many trial-name items a parse-audit dict retains per warning list.
AUDIT_LIST_CAP: int = 20


@dataclass(frozen=True, slots=True)
class Trial:
    task: str
    trial_name: str
    solved: bool
    has_exception: bool          # bool(exception_info) — the campaign errored := truthy convention
    exception_type: str | None   # breakdown only ("?" if info lacks a type); None = clean
    started: datetime | None
    finished: datetime | None
    n_episodes: float
    model_name: str | None
    jobs_dir: str


def parse_ts(ts: str | None, *, tz_naive: timezone = PDT) -> datetime | None:
    """ISO timestamp -> aware datetime. tz-naive values default to PDT; a 'Z' suffix is UTC."""
    if not ts:
        return None
    dt = datetime.fromisoformat(ts)
    return dt.replace(tzinfo=tz_naive) if dt.tzinfo is None else dt


def is_infra(exc_type: str | None, exc_msg: str | None) -> bool:
    """True iff the exception is a docker/infra artifact (harness failed to start), not a real fail."""
    return (INFRA_MSG in (exc_msg or "")) or ("docker" in (exc_type or "").lower())


def capped_append(audit: dict, count_key: str, list_key: str, item: str) -> None:
    """Increment audit[count_key]; append item to audit[list_key] only until it reaches AUDIT_LIST_CAP."""
    audit[count_key] += 1
    if len(audit[list_key]) < AUDIT_LIST_CAP:
        audit[list_key].append(item)


def group_by_task(trials: list[Trial]) -> dict[str, list[Trial]]:
    """{task: trials sorted chronologically (started_at, trial_name)} — roll i = i-th completion."""
    g: dict[str, list[Trial]] = {}
    for t in trials:
        g.setdefault(t.task, []).append(t)
    far_past = datetime(1970, 1, 1, tzinfo=timezone.utc)
    for task in g:
        g[task].sort(key=lambda tr: (tr.started or far_past, tr.trial_name))
    return g
