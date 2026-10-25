"""OrchestratorHarness: drives a brain through the action loop, producing a Trajectory."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from ipj.metrics import (
    UNKNOWN_JOULES_PROVENANCE,
    accelerator_joules_are_complete,
)

from .brain import DEFAULT_SYSTEM, OrchestratorBrain, render_system
from .dispatch import WorkerDispatch, WorkerResult
from .tools import ToolOutcome, ToolSurface
from .trace import (
    Step,
    TaskMeta,
    Trajectory,
    derive_strategy_tags,
    is_valid_nonnegative_number,
)
from .workers import WorkerCard, WorkerPool


@dataclass(slots=True)
class OrchestratorContext:
    task: TaskMeta
    card_catalog: list[WorkerCard]
    tool_catalog: list[dict[str, Any]]
    budget_card: str
    history: list[Step] = field(default_factory=list)


def _add(a: float | int | None, b: float | int | None) -> float | int | None:
    """Add two measurements only when both components are known.

    Treating an absent component as zero produces deceptively complete cost,
    token, latency, and energy totals. Explicit numeric zero remains valid.
    """
    if not is_valid_nonnegative_number(a) or not is_valid_nonnegative_number(b):
        return None
    return a + b


def _merge_component_energy(
    step: Step,
    joules: float | None,
    provenance: str,
    complete: bool,
    device_coverage_complete: bool,
) -> None:
    """Merge one accelerator component without turning a partial sum into a total."""
    if not is_valid_nonnegative_number(step.gpu_energy_j) or not is_valid_nonnegative_number(joules):
        step.gpu_energy_j = None
    else:
        step.gpu_energy_j = step.gpu_energy_j + joules
    same_provenance = step.gpu_energy_provenance == provenance
    merged_complete = (
        step.gpu_energy_complete is True
        and complete is True
        and step.accelerator_device_coverage_complete is True
        and device_coverage_complete is True
        and same_provenance
        and accelerator_joules_are_complete(
            step.gpu_energy_j,
            step.gpu_energy_provenance,
            device_coverage_complete,
        )
    )
    step.gpu_energy_complete = merged_complete
    step.accelerator_device_coverage_complete = merged_complete
    if not merged_complete:
        step.gpu_energy_provenance = UNKNOWN_JOULES_PROVENANCE


class OrchestratorHarness:
    def __init__(
        self,
        brain: OrchestratorBrain,
        dispatch: WorkerDispatch,
        tools: ToolSurface,
        *,
        brain_name: str = "teacher",
        max_turns: int = 30,
    ) -> None:
        self.brain = brain
        self.dispatch = dispatch
        self.tools = tools
        self.brain_name = brain_name
        self.max_turns = max_turns

    def _merge_tool(self, step: Step, target: str, request: dict[str, Any], out: ToolOutcome) -> None:
        step.call_target = target
        step.call_request = request
        step.call_response = out.content
        step.dollars = _add(step.dollars, out.dollars)
        step.latency_s = _add(step.latency_s, out.latency_s)
        _merge_component_energy(
            step,
            out.gpu_energy_j,
            out.gpu_energy_provenance,
            out.gpu_energy_complete,
            out.accelerator_device_coverage_complete,
        )

    def _merge_worker(self, step: Step, alias: str, request: dict[str, Any], wr: WorkerResult) -> None:
        step.call_target = alias
        step.call_request = request
        step.call_response = wr.content
        step.dollars = _add(step.dollars, wr.dollars)
        step.tokens_in = _add(step.tokens_in, wr.tokens_in)
        step.tokens_out = _add(step.tokens_out, wr.tokens_out)
        step.latency_s = _add(step.latency_s, wr.latency_s)
        _merge_component_energy(
            step,
            wr.gpu_energy_j,
            wr.gpu_energy_provenance,
            wr.gpu_energy_complete,
            wr.accelerator_device_coverage_complete,
        )

    def run(self, task: TaskMeta, pool: WorkerPool, *, budget_card: str = "med") -> Trajectory:
        steps: list[Step] = []
        ctx = OrchestratorContext(
            task=task,
            card_catalog=pool.card_catalog(),
            tool_catalog=self.tools.catalog(),
            budget_card=budget_card,
            history=steps,
        )
        # Render the system text ONCE (the exact anonymized context the brain sees) so the SFT
        # linearizer can replay generation input. Falls back to DEFAULT_SYSTEM for brains that
        # carry no system_prompt attribute.
        rendered_system = render_system(ctx, getattr(self.brain, "system_prompt", DEFAULT_SYSTEM))
        final_answer: str | None = None
        for i in range(self.max_turns):
            bs = self.brain.propose(ctx)
            a = bs.action
            step = Step(
                index=i,
                action_type=a.type,
                orch_text=a.raw,
                tokens_in=bs.tokens_in,
                tokens_out=bs.tokens_out,
                dollars=bs.dollars,
                latency_s=bs.latency_s,
                gpu_energy_j=bs.gpu_energy_j,
                gpu_energy_provenance=bs.gpu_energy_provenance,
                gpu_energy_complete=bs.gpu_energy_complete,
                accelerator_device_coverage_complete=(
                    bs.accelerator_device_coverage_complete
                ),
                correlation_id=uuid.uuid4().hex,
            )
            if a.type == "finish":
                final_answer = a.text
                steps.append(step)
                break
            if a.type == "think":
                pass
            elif a.type == "bash":
                out = self.tools.call("bash", {"command": a.text})
                self._merge_tool(step, "bash", {"command": a.text}, out)
            elif a.type == "call_tool":
                out = self.tools.call(a.target, a.args)
                self._merge_tool(step, a.target, dict(a.args), out)
            elif a.type == "call_worker":
                wr = self.dispatch.call_worker(a.target, a.args.get("sub_prompt", ""), a.args.get("context"))
                self._merge_worker(step, a.target, dict(a.args), wr)
            elif a.type == "delegate_task":
                wr = self.dispatch.delegate_task(a.target, task.prompt)
                self._merge_worker(step, a.target, {}, wr)
                final_answer = wr.content
                steps.append(step)
                break
            steps.append(step)
        traj = Trajectory(
            task=task,
            brain=self.brain_name,
            worker_pool_snapshot=pool.snapshot(),
            budget_card=budget_card,
            steps=steps,
            final_answer=final_answer,
            system_prompt=rendered_system,
        )
        traj.strategy_tags = derive_strategy_tags(steps)
        return traj
