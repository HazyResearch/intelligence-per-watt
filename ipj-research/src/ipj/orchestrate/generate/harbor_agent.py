"""IPJOrchestrator: a terminal-bench BaseAgent wrapping the harness for in-container generation/eval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from terminal_bench.agents.base_agent import AgentResult, BaseAgent
    from terminal_bench.agents.failure_mode import FailureMode
except ImportError as e:  # pragma: no cover — surfaced only in bare venvs
    raise ImportError(
        "terminal_bench is required for ipj.orchestrate.generate. "
        "Install with:  uv pip install -e '../ipj[orchestrate]'  (or ipw[terminus])."
    ) from e

from ipj.orchestrate.harness.delegation import make_container_delegate_runner
from ipj.orchestrate.harness.loop import OrchestratorHarness
from ipj.orchestrate.harness.trace import TaskMeta, Trajectory
from ipj.orchestrate.harness.workers import WorkerPool

from .session_tools import build_session_tool_surface


class IPJOrchestrator(BaseAgent):
    """Runs the IPJ orchestrator (#1) as a terminal-bench agent: drives the harness with tools that
    act inside the task container, records the Trajectory, returns token totals to the harness."""

    def __init__(
        self,
        brain: Any,
        dispatch: Any,
        pool: WorkerPool,
        trace_dir: Path,
        *,
        brain_name: str = "teacher",
        budget_card: str = "med",
        max_turns: int = 30,
        task_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._brain = brain
        self._dispatch = dispatch
        self._pool = pool
        self._trace_dir = Path(trace_dir)
        self._brain_name = brain_name
        self._budget_card = budget_card
        self._max_turns = max_turns
        self._task_id = task_id
        self.last_trajectory: Trajectory | None = None

    @staticmethod
    def name() -> str:
        return "ipj-orchestrator"

    def perform_task(self, instruction: str, session: Any, logging_dir: Path | None = None) -> AgentResult:
        task_id = self._task_id or "task"
        tools = build_session_tool_surface(session)
        # A delegated worker should ACT in this SAME container (edit /testbed, run commands), matching
        # the cascade's local->cloud escalation — not return a text-only MCP answer. Wire the
        # container-acting delegate runner onto the dispatch using the live session.
        self._dispatch.delegate_runner = make_container_delegate_runner(session, max_turns=self._max_turns)
        harness = OrchestratorHarness(
            brain=self._brain,
            dispatch=self._dispatch,
            tools=tools,
            brain_name=self._brain_name,
            max_turns=self._max_turns,
        )
        meta = TaskMeta(
            adp_source="swe-gym", task_id=task_id, domain="swe", prompt=instruction, verifiable=True
        )
        traj = harness.run(meta, self._pool, budget_card=self._budget_card)
        self.last_trajectory = traj

        self._trace_dir.mkdir(parents=True, exist_ok=True)
        (self._trace_dir / f"{task_id}.json").write_text(json.dumps(traj.to_dict(), default=str, indent=2))

        tin = sum((s.tokens_in or 0) for s in traj.steps)
        tout = sum((s.tokens_out or 0) for s in traj.steps)
        return AgentResult(
            total_input_tokens=tin,
            total_output_tokens=tout,
            failure_mode=FailureMode.NONE,
            timestamped_markers=[],
        )
