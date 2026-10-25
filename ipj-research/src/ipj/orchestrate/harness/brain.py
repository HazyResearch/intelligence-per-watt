"""Orchestrator brains (TeacherBrain/TrainedBrain) and system-prompt rendering over anonymized worker cards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from ipj.metrics import UNKNOWN_JOULES_PROVENANCE

from .actions import Action, ActionParseError, parse_action

if TYPE_CHECKING:
    from .loop import OrchestratorContext

DEFAULT_SYSTEM = (
    "You are an efficiency-aware orchestrator. Each turn, emit EXACTLY ONE JSON action object and nothing else.\n"
    "Actions:\n"
    '  {"action":"think","text":"..."}                      reason privately\n'
    '  {"action":"bash","command":"..."}                    run a shell command in the sandbox\n'
    '  {"action":"call_tool","tool":"<name>","args":{...}}  use a non-LLM tool\n'
    '  {"action":"call_worker","worker":"<alias>","sub_prompt":"...","context":"..."}  '
    "delegate a SELF-CONTAINED subtask to a worker model; you keep control\n"
    '  {"action":"delegate_task","worker":"<alias>"}        hand the WHOLE task to a worker (clean restart)\n'
    '  {"action":"finish","answer":"..."}                   give the final answer\n'
    "Workers are anonymized; choose by the capability/price/latency in their cards. "
    "Respect the Budget: prefer cheaper actions unless the task needs more capability. "
    "Delegate the whole task only when you are confident a single worker should own it end to end."
)


@dataclass(slots=True)
class ChatResponse:
    content: str
    tokens_in: int | None = None
    tokens_out: int | None = None
    dollars: float | None = None
    latency_s: float | None = None
    gpu_energy_j: float | None = None
    gpu_energy_provenance: str = UNKNOWN_JOULES_PROVENANCE
    gpu_energy_complete: bool = False
    accelerator_device_coverage_complete: bool = False


class ChatClient(Protocol):
    def complete(self, messages: list[dict[str, Any]], **params: Any) -> ChatResponse: ...


@dataclass(slots=True)
class BrainStep:
    action: Action
    tokens_in: int | None = None
    tokens_out: int | None = None
    dollars: float | None = None
    latency_s: float | None = None
    gpu_energy_j: float | None = None
    gpu_energy_provenance: str = UNKNOWN_JOULES_PROVENANCE
    gpu_energy_complete: bool = False
    accelerator_device_coverage_complete: bool = False


class OrchestratorBrain(Protocol):
    def propose(self, ctx: "OrchestratorContext") -> BrainStep: ...


def _render_cards(card_catalog: list[Any]) -> str:
    lines = []
    for c in card_catalog:
        energy = "" if c.est_energy_j_per_ktok is None else f", ~{c.est_energy_j_per_ktok} J/ktok"
        lines.append(
            f"- {c.alias}: {c.capability} (price ${c.price_in}/${c.price_out} per Mtok, "
            f"~{c.est_latency_s}s{energy})"
        )
    return "\n".join(lines)


def render_system(ctx: "OrchestratorContext", system_prompt: str) -> str:
    """The exact system text the brain sees at generation time: action grammar + budget +
    anonymized (per-trace jittered) worker cards + tool names. Reused by the SFT linearizer so
    training input matches generation input."""
    return (
        f"{system_prompt}\n\nBudget: {ctx.budget_card}\n\n"
        f"Available workers:\n{_render_cards(ctx.card_catalog)}\n\n"
        f"Available tools: {', '.join(t['name'] for t in ctx.tool_catalog)}"
    )


def build_messages(ctx: "OrchestratorContext", system_prompt: str) -> list[dict[str, Any]]:
    system = render_system(ctx, system_prompt)
    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": ctx.task.prompt},
    ]
    for s in ctx.history:
        msgs.append({"role": "assistant", "content": s.orch_text})
        if s.call_response is not None:
            msgs.append({"role": "tool", "content": s.call_response})
    return msgs


class TeacherBrain:
    def __init__(self, client: ChatClient, *, system_prompt: str = DEFAULT_SYSTEM) -> None:
        self.client = client
        self.system_prompt = system_prompt

    def propose(self, ctx: "OrchestratorContext") -> BrainStep:
        messages = build_messages(ctx, self.system_prompt)
        resp = self.client.complete(messages)
        try:
            action = parse_action(resp.content)
        except ActionParseError:
            # A teacher that answers directly (ignoring the action grammar) is a valid tier-1
            # self-answer; wrap the raw text as a finish so the trace stays usable for SFT.
            action = Action(type="finish", text=resp.content, raw=resp.content)
        return BrainStep(
            action=action,
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
            dollars=resp.dollars,
            latency_s=resp.latency_s,
            gpu_energy_j=resp.gpu_energy_j,
            gpu_energy_provenance=resp.gpu_energy_provenance,
            gpu_energy_complete=resp.gpu_energy_complete,
            accelerator_device_coverage_complete=(
                resp.accelerator_device_coverage_complete
            ),
        )


class TrainedBrain(TeacherBrain):
    """Same emission/parsing as TeacherBrain; distinct class so traces record which brain produced them."""


class EscalatingBrain:
    """Cold-start scaffold for bootstrapping delegation: defer to ``inner`` for the first
    ``force_after`` turns, then FORCE a ``delegate_task`` to ``worker`` if the task hasn't terminated.

    The inner model's own attempts stay on-policy; only the escalation TRIGGER is scaffolded — to seed
    delegation traces for a base model that won't delegate on its own (so SFT can teach the behavior,
    after which a later iteration tests whether the trained model delegates un-forced).
    """

    def __init__(self, inner: OrchestratorBrain, *, force_after: int = 8, worker: str = "cloud-frontier") -> None:
        self.inner = inner
        self.force_after = force_after
        self.worker = worker
        self.system_prompt = getattr(inner, "system_prompt", DEFAULT_SYSTEM)

    def propose(self, ctx: "OrchestratorContext") -> BrainStep:
        if len(ctx.history) >= self.force_after:
            forced = parse_action(f'{{"action":"delegate_task","worker":"{self.worker}"}}')
            return BrainStep(action=forced)  # forced escalation; no inner-model call this turn
        return self.inner.propose(ctx)
