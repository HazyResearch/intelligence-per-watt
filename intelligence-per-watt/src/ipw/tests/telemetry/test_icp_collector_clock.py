"""Cross-package clock compatibility for controller-window measurements."""

from __future__ import annotations

import pytest

from ipw.telemetry.collector import EnergyMonitorCollector


@pytest.mark.parametrize(
    ("target", "local"),
    [
        ("127.0.0.1:50053", True),
        ("localhost:50053", True),
        ("[::1]:50053", True),
        ("192.0.2.1:50053", False),
        ("monitor.example.org:50053", False),
    ],
)
def test_capture_can_distinguish_local_and_remote_clocks(target, local):
    collector = EnergyMonitorCollector(target=target)
    assert collector.target == target
    assert collector.shares_controller_clock is local
