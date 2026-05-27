"""Bridge SDK-wrapped agents (OpenHands, Terminus) onto the IPW EventBus.

OpenHands and Terminus run inside external SDKs whose internal event streams
differ from IPW's native :class:`~ipw.execution.executor.Executor`. This bridge
republishes those streams as canonical bus events — ``AGENT_START/END``,
``TURN_START/END``, ``TOOL_CALL_START/END``, ``LM_INFERENCE_START/END`` —
assigning ``correlation_id`` so :class:`~ipw.telemetry.energy_attribution.EnergyAttribution`
can pair tool/LM energy windows exactly as it does for native agents.

Fidelity is lower than native agents (spec §4.7), by design:

- **OpenHands**: retry is task-level (whole-task retry on error status), not
  turn-level. Each action/observation pair maps to one bus turn.
- **Terminus**: the whole container run is a single window — :meth:`start`
  opens one ``TURN_START`` and :meth:`finish` closes it, attributing energy to
  the entire run.

The bridge never imports ``openhands-sdk`` or ``terminal-bench``. It operates on
a small duck-typed event-shape contract via :func:`default_classify`, so the
same code path works against stub events in tests and real SDK events in
production. If an SDK's event API differs, pass a custom ``classifier``; the
legacy non-bus agent path is unaffected either way.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional

from .eventbus import Event, EventBus
from .events import EventType


class BridgeEventKind(str, Enum):
    """Normalized classification of an external wrapper-agent event."""

    LM_START = "lm_start"
    LM_END = "lm_end"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    OTHER = "other"  # ignored — not republished


@dataclass
class NormalizedEvent:
    """An external event mapped onto the bridge's vocabulary."""

    kind: BridgeEventKind
    tool_name: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)


def default_classify(raw: Any) -> NormalizedEvent:
    """Best-effort classifier for OpenHands-style and stub events.

    Recognizes, in order:

    1. A ``dict`` with a ``"kind"`` key whose value is a :class:`BridgeEventKind`
       (or its string value) — the stub/test contract.
    2. An object whose class name contains ``Action`` / ``Observation`` /
       ``LLM`` — the openhands-sdk shape (``ActionEvent``, ``ObservationEvent``,
       LLM-convertible events), read purely by attribute, no SDK import.

    Anything unrecognized maps to :data:`BridgeEventKind.OTHER` and is dropped.
    """
    if isinstance(raw, dict):
        kind = raw.get("kind")
        if kind is not None:
            try:
                bk = BridgeEventKind(kind)
            except ValueError:
                bk = BridgeEventKind.OTHER
            return NormalizedEvent(
                kind=bk,
                tool_name=raw.get("tool_name"),
                payload={k: v for k, v in raw.items() if k not in ("kind", "tool_name")},
            )
        return NormalizedEvent(kind=BridgeEventKind.OTHER)

    cls_name = type(raw).__name__
    tool_name = getattr(raw, "tool_name", None)
    if "Action" in cls_name:
        return NormalizedEvent(kind=BridgeEventKind.TOOL_START, tool_name=tool_name)
    if "Observation" in cls_name:
        return NormalizedEvent(kind=BridgeEventKind.TOOL_END, tool_name=tool_name)
    # LLM-convertible events carry token/cost metrics but are not tool calls; we
    # do not synthesize LM windows from them by default (no reliable paired
    # start/end). Callers that want LM windows should emit explicit lm_start/
    # lm_end via a custom classifier.
    return NormalizedEvent(kind=BridgeEventKind.OTHER)


class WrapperAgentBridge:
    """Republishes a wrapper agent's lifecycle onto an :class:`EventBus`.

    Typical OpenHands usage::

        bridge = WrapperAgentBridge(bus, agent_name="openhands", task_id=tid)
        bridge.start()
        for raw in sdk_event_stream:
            bridge.on_event(raw)
        bridge.finish(status="success")

    Typical Terminus usage (whole-run window, no per-tool events)::

        bridge = WrapperAgentBridge(bus, agent_name="terminus", task_id=tid)
        bridge.start()          # AGENT_START + TURN_START
        ... run container ...
        bridge.finish()         # TURN_END + AGENT_END
    """

    def __init__(
        self,
        bus: EventBus,
        *,
        agent_name: str,
        task_id: str,
        classifier: Optional[Callable[[Any], NormalizedEvent]] = None,
        open_turn_on_start: bool = True,
    ) -> None:
        """Create a bridge bound to ``bus``.

        Args:
            bus: Target event bus.
            agent_name: Name reported in AGENT_START/END payloads.
            task_id: Task identifier reported in lifecycle payloads.
            classifier: Maps a raw external event to a :class:`NormalizedEvent`.
                Defaults to :func:`default_classify`.
            open_turn_on_start: If True (Terminus mode), :meth:`start` opens a
                single enclosing turn that :meth:`finish` closes. If False
                (OpenHands mode), turns open/close per tool action.
        """
        self._bus = bus
        self._agent_name = agent_name
        self._task_id = task_id
        self._classify = classifier or default_classify
        self._open_turn_on_start = open_turn_on_start

        self._started = False
        self._finished = False
        self._turn_idx = 0
        self._turn_open = False
        # Pending correlation ids keyed by (kind, tool_name) so start/end pair up.
        self._pending_tool: Dict[Optional[str], str] = {}
        self._pending_lm: Optional[str] = None
        self._n_tool_calls = 0

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Emit AGENT_START (and, in Terminus mode, an enclosing TURN_START)."""
        if self._started:
            return
        self._started = True
        self._publish(EventType.AGENT_START, payload={
            "agent_name": self._agent_name, "task_id": self._task_id,
        })
        if self._open_turn_on_start:
            self._open_turn()

    def on_event(self, raw: Any) -> None:
        """Classify one external event and republish it on the bus.

        Unrecognized events (:data:`BridgeEventKind.OTHER`) are dropped. Calling
        before :meth:`start` auto-starts the bridge so callers can't lose the
        first event.
        """
        if not self._started:
            self.start()
        if self._finished:
            return
        norm = self._classify(raw)
        if norm.kind is BridgeEventKind.TOOL_START:
            self._on_tool_start(norm)
        elif norm.kind is BridgeEventKind.TOOL_END:
            self._on_tool_end(norm)
        elif norm.kind is BridgeEventKind.LM_START:
            self._on_lm_start(norm)
        elif norm.kind is BridgeEventKind.LM_END:
            self._on_lm_end(norm)
        # OTHER → ignored

    def finish(self, status: str = "success", n_turns: Optional[int] = None) -> None:
        """Close any open turn and emit AGENT_END. Idempotent."""
        if self._finished:
            return
        self._finished = True
        if self._turn_open:
            self._close_turn(is_final=True)
        self._publish(EventType.AGENT_END, payload={
            "agent_name": self._agent_name,
            "task_id": self._task_id,
            "status": status,
            "n_turns": n_turns if n_turns is not None else self._turn_idx,
        })

    # -- per-event handlers --------------------------------------------------

    def _on_tool_start(self, norm: NormalizedEvent) -> None:
        # In OpenHands mode each action opens its own turn.
        if not self._open_turn_on_start and not self._turn_open:
            self._open_turn()
        cid = str(uuid.uuid4())
        self._pending_tool[norm.tool_name] = cid
        self._n_tool_calls += 1
        self._publish(EventType.TOOL_CALL_START, payload={
            "tool": norm.tool_name, **norm.payload,
        }, correlation_id=cid)

    def _on_tool_end(self, norm: NormalizedEvent) -> None:
        cid = self._pending_tool.pop(norm.tool_name, None) or str(uuid.uuid4())
        self._publish(EventType.TOOL_CALL_END, payload={
            "tool": norm.tool_name, **norm.payload,
        }, correlation_id=cid)
        # OpenHands mode: an observation closes the turn its action opened.
        if not self._open_turn_on_start and self._turn_open:
            self._close_turn(is_final=False)

    def _on_lm_start(self, norm: NormalizedEvent) -> None:
        if not self._open_turn_on_start and not self._turn_open:
            self._open_turn()
        self._pending_lm = str(uuid.uuid4())
        self._publish(EventType.LM_INFERENCE_START, payload=dict(norm.payload),
                      correlation_id=self._pending_lm)

    def _on_lm_end(self, norm: NormalizedEvent) -> None:
        cid = self._pending_lm or str(uuid.uuid4())
        self._pending_lm = None
        self._publish(EventType.LM_INFERENCE_END, payload=dict(norm.payload),
                      correlation_id=cid)

    # -- turn helpers --------------------------------------------------------

    def _open_turn(self) -> None:
        self._turn_open = True
        self._publish(EventType.TURN_START, payload={
            "turn_idx": self._turn_idx, "task_id": self._task_id,
        })

    def _close_turn(self, *, is_final: bool) -> None:
        self._publish(EventType.TURN_END, payload={
            "turn_idx": self._turn_idx, "is_final": is_final,
        })
        self._turn_open = False
        self._turn_idx += 1

    # -- bus plumbing --------------------------------------------------------

    def _publish(
        self, event_type: EventType, *, payload: Dict[str, Any],
        correlation_id: Optional[str] = None,
    ) -> None:
        self._bus.publish(Event(
            event_type=event_type,
            timestamp_ns=time.time_ns(),
            payload=payload,
            turn_id=str(self._turn_idx),
            correlation_id=correlation_id,
        ))


__all__ = [
    "WrapperAgentBridge", "NormalizedEvent", "BridgeEventKind", "default_classify",
]
