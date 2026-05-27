"""Tests for telemetry/wrapper_agent_bridge.py — SDK-agent → EventBus bridge.

Uses a stub event source (dicts with a ``"kind"`` key) so no openhands-sdk or
terminal-bench import is required. Also covers duck-typed classification of
OpenHands-shaped objects (class names containing Action/Observation).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List

from ipw.telemetry.eventbus import Event, EventBus
from ipw.telemetry.events import EventType
from ipw.telemetry.wrapper_agent_bridge import (
    BridgeEventKind,
    WrapperAgentBridge,
    default_classify,
)


def _collect(bus: EventBus) -> List[Event]:
    seen: List[Event] = []
    bus.subscribe(None, seen.append)
    return seen


def test_openhands_action_observation_pairs_by_correlation_id() -> None:
    """An action/observation pair becomes TOOL_CALL_START/END with one cid."""
    bus = EventBus()
    seen = _collect(bus)
    bridge = WrapperAgentBridge(
        bus, agent_name="openhands", task_id="t1", open_turn_on_start=False,
    )
    bridge.start()
    bridge.on_event({"kind": "tool_start", "tool_name": "shell"})
    bridge.on_event({"kind": "tool_end", "tool_name": "shell"})
    bridge.finish()

    types = [e.event_type for e in seen]
    assert types == [
        EventType.AGENT_START,
        EventType.TURN_START,
        EventType.TOOL_CALL_START,
        EventType.TOOL_CALL_END,
        EventType.TURN_END,
        EventType.AGENT_END,
    ]
    start = next(e for e in seen if e.event_type == EventType.TOOL_CALL_START)
    end = next(e for e in seen if e.event_type == EventType.TOOL_CALL_END)
    assert start.correlation_id is not None
    assert start.correlation_id == end.correlation_id  # pairs for energy attribution


def test_terminus_whole_run_single_window() -> None:
    """Terminus mode: start opens one enclosing turn, finish closes it."""
    bus = EventBus()
    seen = _collect(bus)
    bridge = WrapperAgentBridge(
        bus, agent_name="terminus", task_id="t2", open_turn_on_start=True,
    )
    bridge.start()
    bridge.finish(status="success")

    types = [e.event_type for e in seen]
    assert types == [
        EventType.AGENT_START,
        EventType.TURN_START,
        EventType.TURN_END,
        EventType.AGENT_END,
    ]
    end = seen[-1]
    assert end.payload["status"] == "success"
    assert end.payload["n_turns"] == 1


def test_lm_inference_pairs_by_correlation_id() -> None:
    """lm_start/lm_end map to LM_INFERENCE_* with a shared correlation id."""
    bus = EventBus()
    seen = _collect(bus)
    bridge = WrapperAgentBridge(
        bus, agent_name="openhands", task_id="t3", open_turn_on_start=False,
    )
    bridge.on_event({"kind": "lm_start"})
    bridge.on_event({"kind": "lm_end", "tokens": 42})
    bridge.finish()

    lm_start = next(e for e in seen if e.event_type == EventType.LM_INFERENCE_START)
    lm_end = next(e for e in seen if e.event_type == EventType.LM_INFERENCE_END)
    assert lm_start.correlation_id == lm_end.correlation_id
    assert lm_end.payload.get("tokens") == 42


def test_on_event_before_start_auto_starts() -> None:
    """Calling on_event without start() should not drop the first event."""
    bus = EventBus()
    seen = _collect(bus)
    bridge = WrapperAgentBridge(
        bus, agent_name="openhands", task_id="t4", open_turn_on_start=False,
    )
    bridge.on_event({"kind": "tool_start", "tool_name": "git"})
    assert seen[0].event_type == EventType.AGENT_START
    assert any(e.event_type == EventType.TOOL_CALL_START for e in seen)


def test_finish_is_idempotent_and_closes_open_turn() -> None:
    """Double finish() emits AGENT_END once; an open turn is closed."""
    bus = EventBus()
    seen = _collect(bus)
    bridge = WrapperAgentBridge(
        bus, agent_name="terminus", task_id="t5", open_turn_on_start=True,
    )
    bridge.start()
    bridge.finish()
    bridge.finish()  # second call is a no-op
    agent_ends = [e for e in seen if e.event_type == EventType.AGENT_END]
    turn_ends = [e for e in seen if e.event_type == EventType.TURN_END]
    assert len(agent_ends) == 1
    assert len(turn_ends) == 1


def test_default_classify_duck_types_openhands_objects() -> None:
    """Objects whose class name contains Action/Observation classify by shape."""
    Action = type("ActionEvent", (), {})
    Observation = type("ObservationEvent", (), {})
    act = Action()
    act.tool_name = "shell"
    obs = Observation()
    obs.tool_name = "shell"

    assert default_classify(act).kind is BridgeEventKind.TOOL_START
    assert default_classify(act).tool_name == "shell"
    assert default_classify(obs).kind is BridgeEventKind.TOOL_END
    # Unknown objects drop to OTHER.
    assert default_classify(SimpleNamespace(foo=1)).kind is BridgeEventKind.OTHER
    # Unknown dict kind → OTHER, not a crash.
    assert default_classify({"kind": "nonsense"}).kind is BridgeEventKind.OTHER
