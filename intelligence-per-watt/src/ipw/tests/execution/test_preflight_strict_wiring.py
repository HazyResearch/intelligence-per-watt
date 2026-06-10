"""Tests that AgenticRunner.require_dedicated_hardware drives preflight strictness.

Contract:
  - AgenticRunner(require_dedicated_hardware=True)  → run_preflight(strict=True)
  - AgenticRunner(require_dedicated_hardware=False) → run_preflight(strict=False)

The wiring lives in ``AgenticRunner.run()``, which passes
``self._require_dedicated_hardware`` to ``_run_preflight_if_needed``.  These
tests drive a real (empty-dataset) ``run()`` and assert on the recorded
``strict`` value, so they exercise the actual call site rather than the
mapping in isolation.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from ipw.execution.agentic_runner import AgenticRunner
from ipw.execution.preflight import PreflightResult
from ipw.telemetry.events import EventRecorder


class _PreflightRecorder:
    """Records the ``strict`` kwarg passed to run_preflight; returns a clean result."""

    def __init__(self) -> None:
        self.calls: list[bool] = []

    def __call__(self, *, strict: bool = False) -> PreflightResult:
        self.calls.append(strict)
        return PreflightResult(
            gpu_util_pct_avg=0.0,
            cpu_util_pct_avg=0.0,
            foreign_gpu_pids=[],
            shared_device_baseline_dirty=False,
            warnings=[],
        )


def _make_empty_runner(*, require_dedicated_hardware: bool) -> AgenticRunner:
    """Build a real runner over an empty dataset so ``run()`` returns after preflight."""
    dataset = MagicMock()
    dataset.size.return_value = 0
    dataset.__iter__ = MagicMock(return_value=iter([]))
    return AgenticRunner(
        agent=MagicMock(),
        dataset=dataset,
        telemetry_session=None,
        config={"model": "stub-model"},
        event_recorder=EventRecorder(),
        require_dedicated_hardware=require_dedicated_hardware,
    )


def test_require_dedicated_hardware_true_runs_preflight_strict() -> None:
    runner = _make_empty_runner(require_dedicated_hardware=True)
    recorder = _PreflightRecorder()
    with patch("ipw.execution.agentic_runner.run_preflight", recorder):
        asyncio.run(runner.run())
    assert recorder.calls == [True]


def test_require_dedicated_hardware_false_runs_preflight_lenient() -> None:
    runner = _make_empty_runner(require_dedicated_hardware=False)
    recorder = _PreflightRecorder()
    with patch("ipw.execution.agentic_runner.run_preflight", recorder):
        asyncio.run(runner.run())
    assert recorder.calls == [False]
