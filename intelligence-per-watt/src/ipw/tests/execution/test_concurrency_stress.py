"""20-way concurrency stress test for AgenticRunner isolation.

Validates that AgenticRunner with concurrency=20 runs 20 queries in parallel
without cross-talk between agents, preserves index order, and completes without
deadlocks under a generous wall-clock bound.

No API keys, no GPU, no real model required.
"""

from __future__ import annotations

import asyncio
import time
from typing import Iterable, Optional

import pytest

from ipw.agents.base import ToolUsingAgent
from ipw.core.types import DatasetRecord
from ipw.datasets.base import DatasetProvider
from ipw.execution.executor import ExecutorContext, TurnOutput
from ipw.telemetry.events import EventRecorder

# ---------------------------------------------------------------------------
# Stub dataset — 20 distinct records
# ---------------------------------------------------------------------------

N_QUERIES = 20


class _StubDataset(DatasetProvider):
    """Stub dataset that yields N_QUERIES distinct records."""

    dataset_id = "stub_stress"
    dataset_name = "stub_stress"

    def iter_records(self) -> Iterable[DatasetRecord]:
        for i in range(N_QUERIES):
            yield DatasetRecord(
                problem=f"task-{i:02d}",
                answer=f"answer-{i:02d}",
                subject="stress",
                dataset_metadata={"workload_type": "stress", "index": i},
            )

    def size(self) -> int:
        return N_QUERIES


# ---------------------------------------------------------------------------
# Stub agent — subclasses ToolUsingAgent, returns final answer immediately
# ---------------------------------------------------------------------------


class _StubAgent(ToolUsingAgent):
    """Minimal ToolUsingAgent stub for stress testing.

    Echoes its task text as the final answer so cross-talk can be detected.
    """

    tools: list = []

    def __init__(
        self,
        model: str = "stub",
        event_recorder: Optional[EventRecorder] = None,
        **kwargs,
    ) -> None:
        super().__init__(event_recorder=event_recorder)
        self._model = model
        self._task: Optional[str] = None

    def set_task(self, task: str) -> None:
        self._task = task

    async def step(self, context: ExecutorContext) -> TurnOutput:
        # Tiny sleep to allow concurrency to actually overlap
        await asyncio.sleep(0.01)
        return TurnOutput(
            final_answer=self._task,
            tool_calls=[],
        )


# ---------------------------------------------------------------------------
# Agent factory — fresh agent + fresh recorder per query
# ---------------------------------------------------------------------------


def _make_agent_factory():
    """Return a callable that produces a fresh (_StubAgent, EventRecorder) pair.

    AgenticRunner's agent_factory is called with no arguments; it returns a
    BaseAgent instance.  The recorder is not returned by the factory itself —
    the concurrent path creates its own recorder per query.  So the factory
    only needs to return the agent.
    """
    def factory() -> _StubAgent:
        return _StubAgent(model="stub", event_recorder=EventRecorder())

    return factory


# ---------------------------------------------------------------------------
# The stress test
# ---------------------------------------------------------------------------


@pytest.mark.stress
class TestConcurrencyStress:
    """20-way concurrency stress test for AgenticRunner."""

    def test_20_concurrent_queries_no_crosstalk(self) -> None:
        """Run 20 queries concurrently, assert order + isolation + completion."""
        from ipw.execution.agentic_runner import AgenticRunner

        dataset = _StubDataset()
        # A placeholder agent is required by the constructor; the concurrent
        # path uses agent_factory instead, but __init__ still needs an agent.
        placeholder_agent = _StubAgent()
        factory = _make_agent_factory()

        runner = AgenticRunner(
            agent=placeholder_agent,
            dataset=dataset,
            telemetry_session=None,
            config={"model": "stub"},
            concurrency=N_QUERIES,
            agent_factory=factory,
            max_turns=5,
        )

        wall_clock_limit = 60.0  # generous bound to catch deadlocks
        t0 = time.monotonic()

        traces = asyncio.run(runner.run(max_queries=N_QUERIES))

        elapsed = time.monotonic() - t0

        # ------------------------------------------------------------------
        # Assertion 1: exactly 20 traces returned
        # ------------------------------------------------------------------
        assert len(traces) == N_QUERIES, (
            f"Expected {N_QUERIES} traces, got {len(traces)}"
        )

        # ------------------------------------------------------------------
        # Assertion 5: completed within the wall-clock bound (no deadlock)
        # ------------------------------------------------------------------
        assert elapsed < wall_clock_limit, (
            f"Runner took {elapsed:.1f}s, exceeding {wall_clock_limit}s limit — "
            "possible deadlock or severe slowdown"
        )

        # ------------------------------------------------------------------
        # Assertion 2: traces in index order (query_id q0000..q0019)
        # ------------------------------------------------------------------
        for i, trace in enumerate(traces):
            expected_qid = f"q{i:04d}"
            assert trace.query_id == expected_qid, (
                f"Slot {i}: expected query_id={expected_qid!r}, got {trace.query_id!r}"
            )

        # ------------------------------------------------------------------
        # Assertion 3: no cross-talk — each trace's response echoes its task
        # ------------------------------------------------------------------
        for i, trace in enumerate(traces):
            expected_answer = f"task-{i:02d}"
            assert trace.response_text == expected_answer, (
                f"Query {i} cross-talk detected: "
                f"expected response_text={expected_answer!r}, "
                f"got {trace.response_text!r}"
            )

        # ------------------------------------------------------------------
        # Assertion 4: all 20 completed == True
        # ------------------------------------------------------------------
        for i, trace in enumerate(traces):
            assert trace.completed is True, (
                f"Query {i} (query_id={trace.query_id}) has completed=False"
            )
