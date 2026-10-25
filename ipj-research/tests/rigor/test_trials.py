"""Unit tests for ipj.rigor.trials: Trial harvest helpers + group_by_task ordering."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ipj.rigor.trials import (
    AUDIT_LIST_CAP,
    INFRA_MSG,
    Trial,
    capped_append,
    group_by_task,
    is_infra,
    parse_ts,
)


class TestParseTs:
    def test_none_returns_none(self):
        assert parse_ts(None) is None
        assert parse_ts("") is None

    def test_z_suffix_is_utc(self):
        dt = parse_ts("2026-06-17T12:34:56Z")
        assert dt.tzinfo is timezone.utc

    def test_naive_gets_pdt(self):
        dt = parse_ts("2026-06-17T12:34:56")
        # PDT = UTC-7
        assert dt.utcoffset() == timedelta(hours=-7)

    def test_explicit_offset_preserved(self):
        dt = parse_ts("2026-06-17T12:34:56+00:00")
        assert dt.utcoffset() == timedelta(0)


class TestIsInfra:
    def test_docker_msg_matches(self):
        assert is_infra(None, f"stderr: {INFRA_MSG}: no such file")

    def test_docker_type_matches_case_insensitive(self):
        assert is_infra("DockerComposeError", None)
        assert is_infra("docker_compose_failed", None)

    def test_agent_timeout_not_infra(self):
        assert not is_infra("AgentTimeoutError", "task exceeded 600s")

    def test_empty_inputs_not_infra(self):
        assert not is_infra(None, None)
        assert not is_infra("", "")


class TestCappedAppend:
    def test_increments_count_but_caps_list(self):
        audit = {"n": 0, "items": []}
        for i in range(AUDIT_LIST_CAP + 5):
            capped_append(audit, "n", "items", f"trial{i}")
        assert audit["n"] == AUDIT_LIST_CAP + 5
        assert len(audit["items"]) == AUDIT_LIST_CAP
        # First items retained; later ones dropped
        assert audit["items"][0] == "trial0"


def _make_trial(task: str, name: str, started: datetime | None) -> Trial:
    return Trial(
        task=task, trial_name=name, solved=False,
        has_exception=False, exception_type=None,
        started=started, finished=None,
        n_episodes=0.0, model_name=None, jobs_dir="/tmp",
    )


class TestGroupByTask:
    def test_groups_by_task_key(self):
        t1 = _make_trial("taskA", "taskA__1", datetime(2026, 6, 17, tzinfo=timezone.utc))
        t2 = _make_trial("taskB", "taskB__1", datetime(2026, 6, 17, tzinfo=timezone.utc))
        t3 = _make_trial("taskA", "taskA__2", datetime(2026, 6, 17, 12, tzinfo=timezone.utc))
        g = group_by_task([t1, t2, t3])
        assert set(g) == {"taskA", "taskB"}
        assert len(g["taskA"]) == 2
        assert len(g["taskB"]) == 1

    def test_chronological_ordering_by_started(self):
        early = datetime(2026, 6, 17, tzinfo=timezone.utc)
        later = datetime(2026, 6, 17, 6, tzinfo=timezone.utc)
        t_late = _make_trial("t", "t__late", later)
        t_early = _make_trial("t", "t__early", early)
        g = group_by_task([t_late, t_early])
        assert [tr.trial_name for tr in g["t"]] == ["t__early", "t__late"]

    def test_none_started_sorts_first_via_far_past(self):
        # None start → epoch 1970 (far_past) → sorts before any real timestamp
        t_none = _make_trial("t", "t__none", None)
        t_ts = _make_trial("t", "t__2026", datetime(2026, 6, 17, tzinfo=timezone.utc))
        g = group_by_task([t_ts, t_none])
        assert g["t"][0].trial_name == "t__none"

    def test_trial_name_tiebreaker(self):
        # Same started → sorted by trial_name ascending
        same = datetime(2026, 6, 17, tzinfo=timezone.utc)
        trials = [_make_trial("t", f"t__{i}", same) for i in ("c", "a", "b")]
        g = group_by_task(trials)
        assert [tr.trial_name for tr in g["t"]] == ["t__a", "t__b", "t__c"]


class TestTrialImmutability:
    def test_trial_is_frozen(self):
        tr = _make_trial("t", "t__1", None)
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            tr.task = "t2"  # type: ignore[misc]
