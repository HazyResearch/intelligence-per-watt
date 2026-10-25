from __future__ import annotations

from types import SimpleNamespace

from ipj.energy import capture
from ipj.energy.capture import EnergyCoveragePolicy, RunEnergyCapture, samples_from_session


class _FakeSession:
    def __init__(self, rows):
        self._rows = rows  # list of (timestamp, watts)

    def window(self, start, end):
        for t, w in self._rows:
            if start <= t <= end:
                yield SimpleNamespace(timestamp=t, reading=SimpleNamespace(power_watts=w))


_TEST_POLICY = EnergyCoveragePolicy(
    max_boundary_gap_s=0.5,
    max_sample_gap_s=2.0,
)


def _capture(rows, *, policy=_TEST_POLICY):
    cap = object.__new__(RunEnergyCapture)
    cap._session = _FakeSession(rows)
    cap._coverage_policy = policy
    cap._controller_clock_aligned = True
    return cap


def test_samples_from_session_maps_and_filters_none_and_sentinel():
    # None (unavailable) and -1.0 (Rust negative sentinel) are both dropped.
    sess = _FakeSession([(0.0, 100.0), (1.0, None), (2.0, 120.0), (3.0, -1.0)])
    out = samples_from_session(sess, 0.0, 10.0)
    assert [s.t for s in out] == [0.0, 2.0]
    assert [s.watts for s in out] == [100.0, 120.0]


def test_joules_is_none_when_ipw_telemetry_is_unavailable(monkeypatch):
    monkeypatch.setattr(capture, "_IPW_OK", False)
    cap = RunEnergyCapture()

    assert cap.joules(0.0, 10.0) is None


def test_joules_is_none_for_unaligned_remote_clock():
    cap = _capture([(0.25, 100.0), (9.75, 100.0)])
    cap._controller_clock_aligned = False

    assert cap.joules(0.0, 10.0) is None


def test_joules_is_none_with_fewer_than_two_valid_samples():
    cap = _capture([(0.25, 100.0), (2.0, None), (3.0, -1.0)])

    assert cap.joules(0.0, 10.0) is None


def test_joules_is_none_for_invalid_window_or_duplicate_timestamps():
    cap = _capture([(0.25, 100.0), (0.25, 110.0)])

    assert cap.joules(0.0, 10.0) is None
    assert cap.joules(10.0, 10.0) is None


def test_joules_rejects_early_telemetry_dropout():
    cap = _capture([(0.25, 100.0), (1.0, 100.0), (2.0, 100.0)])

    assert cap.joules(0.0, 10.0) is None


def test_joules_rejects_late_telemetry_start():
    cap = _capture([(8.0, 100.0), (9.0, 100.0), (9.75, 100.0)])

    assert cap.joules(0.0, 10.0) is None


def test_joules_rejects_large_internal_sample_gap():
    cap = _capture([(0.25, 100.0), (2.0, 100.0), (8.0, 100.0), (9.75, 100.0)])

    assert cap.joules(0.0, 10.0) is None


def test_joules_integrates_when_the_complete_window_is_covered():
    cap = _capture(
        [
            (0.25, 100.0),
            (2.0, 100.0),
            (4.0, 100.0),
            (6.0, 100.0),
            (8.0, 100.0),
            (9.75, 100.0),
        ]
    )

    assert cap.joules(0.0, 10.0) == 1000.0
    assert not cap.accelerator_device_coverage_complete(0.0, 10.0)


def test_joules_preserves_a_genuine_measured_zero_with_complete_coverage():
    cap = _capture(
        [
            (0.25, 0.0),
            (2.0, 0.0),
            (4.0, 0.0),
            (6.0, 0.0),
            (8.0, 0.0),
            (9.75, 0.0),
        ]
    )

    assert cap.joules(0.0, 10.0) == 0.0
