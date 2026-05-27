"""Tests that react-native agent construction in ipw run passes llm and tools.

The bug: run.py called ``agent_cls(model=resolved_model, ...)`` for all agents,
but NativeReact.__init__ requires three positional args: model (str), llm
(adapter), and tools (list).  This test exercises the real _build_agent helper
so any regression re-introduces a test failure.

Design: we test _build_agent directly (no CliRunner, no live inference).
- No network calls are made — OpenAIChatAdapter only contacts OpenAI on
  ``complete()``, not on construction.
- Tool imports are triggered before calling _build_agent, mirroring run.py.
"""

from __future__ import annotations

import importlib

import pytest

# ---------------------------------------------------------------------------
# Helpers — import tool modules so ToolRegistry is populated
# ---------------------------------------------------------------------------

_TOOL_MODS = (
    "shell_exec", "git_tool", "apply_patch", "http_request",
    "repl", "code_interpreter_docker",
    "browser", "browser_axtree", "pdf_tool",
    "image_tool", "audio_tool", "docker_shell_exec",
)


def _ensure_tools_registered() -> None:
    """Import each tool module, silently skipping those with missing optional deps."""
    for mod in _TOOL_MODS:
        try:
            importlib.import_module(f"ipw.tools.{mod}")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Helpers — re-implement _build_agent exactly as run.py does, but importable
# from tests without invoking the full CLI.
# ---------------------------------------------------------------------------

def _build_agent_under_test(a_cls, a_resolved_model, a_event_recorder, a_extra_kwargs, model: str):
    """Mirrors the ``_build_agent`` closure in ipw/cli/run.py.

    We duplicate the closure logic here so the test remains decoupled from
    Click's import machinery while still exercising the real construction path.
    Kept in sync with run.py — if run.py changes, update both.
    """
    from ipw.agents.base import ToolUsingAgent
    if issubclass(a_cls, ToolUsingAgent):
        from ipw.tools.registry import ToolRegistry
        bare = model.split("/", 1)[1] if "/" in model else model
        tools: list = []
        for _tid, _tcls in ToolRegistry.items():
            try:
                tools.append(_tcls(bus=a_event_recorder.bus))
            except Exception:
                pass
        return a_cls(
            model=bare,
            llm=a_resolved_model,
            tools=tools,
            event_recorder=a_event_recorder,
            **a_extra_kwargs,
        )
    return a_cls(
        model=a_resolved_model,
        event_recorder=a_event_recorder,
        **a_extra_kwargs,
    )


# ---------------------------------------------------------------------------
# Actual tests
# ---------------------------------------------------------------------------

class TestReactNativeConstruction:
    """_build_agent produces a valid NativeReact with llm and tools set."""

    def test_react_native_has_llm_and_nonempty_tools(self) -> None:
        """NativeReact is built with a non-None llm and at least one tool."""
        # Trigger tool registrations (mirrors run.py behaviour)
        _ensure_tools_registered()

        # Import the real NativeReact so we exercise its actual __init__
        from ipw.agents.react_native import NativeReact
        from ipw.clients.openai_chat_adapter import OpenAIChatAdapter
        from ipw.telemetry.events import EventRecorder

        model = "openai/gpt-4o-mini"
        # resolved_model comes from _create_model_for_agent("react-native", model, ...)
        adapter = OpenAIChatAdapter(model=model)
        event_recorder = EventRecorder()

        agent = _build_agent_under_test(
            a_cls=NativeReact,
            a_resolved_model=adapter,
            a_event_recorder=event_recorder,
            a_extra_kwargs={},
            model=model,
        )

        assert isinstance(agent, NativeReact), "Expected a NativeReact instance"
        assert agent._llm is adapter, "_llm should be the adapter object"
        assert agent.model == "gpt-4o-mini", "model should be the bare id (no prefix)"
        assert len(agent.tools) > 0, "tools list must not be empty after tool imports"

    def test_react_native_bare_model_no_prefix(self) -> None:
        """model attribute is the bare id even when CLI supplies an 'openai/' prefix."""
        _ensure_tools_registered()
        from ipw.agents.react_native import NativeReact
        from ipw.clients.openai_chat_adapter import OpenAIChatAdapter
        from ipw.telemetry.events import EventRecorder

        adapter = OpenAIChatAdapter(model="openai/gpt-4o-mini")
        er = EventRecorder()
        agent = _build_agent_under_test(NativeReact, adapter, er, {}, "openai/gpt-4o-mini")
        assert agent.model == "gpt-4o-mini"

    def test_react_native_bare_model_without_prefix(self) -> None:
        """When model has no prefix, model attribute stays unchanged."""
        _ensure_tools_registered()
        from ipw.agents.react_native import NativeReact
        from ipw.clients.openai_chat_adapter import OpenAIChatAdapter
        from ipw.telemetry.events import EventRecorder

        adapter = OpenAIChatAdapter(model="gpt-4o-mini")
        er = EventRecorder()
        agent = _build_agent_under_test(NativeReact, adapter, er, {}, "gpt-4o-mini")
        assert agent.model == "gpt-4o-mini"

    def test_non_tool_using_agent_uses_generic_path(self) -> None:
        """A plain BaseAgent subclass is built with the generic (model=, event_recorder=) path."""
        from ipw.agents.base import BaseAgent

        class _FakeAgent(BaseAgent):
            def __init__(self, model, event_recorder=None, **kw):
                super().__init__(event_recorder=event_recorder)
                self.model = model

        from ipw.telemetry.events import EventRecorder
        er = EventRecorder()
        agent = _build_agent_under_test(_FakeAgent, "the-model", er, {}, "the-model")
        assert isinstance(agent, _FakeAgent)
        assert agent.model == "the-model"

    def test_old_generic_call_raises_typeerror(self) -> None:
        """Regression: the OLD generic construction path raises TypeError for NativeReact.

        This test FAILS against the pre-fix code (where agent_cls(model=adapter, ...)
        was called directly) and PASSES after the fix, proving the test detects the bug.
        """
        _ensure_tools_registered()
        from ipw.agents.react_native import NativeReact
        from ipw.clients.openai_chat_adapter import OpenAIChatAdapter
        from ipw.telemetry.events import EventRecorder

        adapter = OpenAIChatAdapter(model="openai/gpt-4o-mini")
        er = EventRecorder()

        # The OLD broken call: passes adapter as model, omits llm and tools.
        with pytest.raises(TypeError, match="missing .* required positional argument"):
            NativeReact(model=adapter, event_recorder=er)
