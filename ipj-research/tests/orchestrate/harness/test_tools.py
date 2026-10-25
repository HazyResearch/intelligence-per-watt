from __future__ import annotations

import pytest

from ipj.orchestrate.harness.tools import ToolInfo, ToolOutcome, ToolSurface


class EchoBackend:
    def run(self, **kwargs):
        return ToolOutcome(content=f"echo:{kwargs}", success=True, dollars=0.0)


def _surface():
    backends = {"echo": EchoBackend()}
    infos = {"echo": ToolInfo(name="echo", description="echoes args", schema={"type": "object"})}
    return ToolSurface(backends, infos)


def test_call_dispatches_to_backend():
    out = _surface().call("echo", {"x": 1})
    assert out.success is True
    assert "x" in out.content


def test_catalog_lists_tools():
    cat = _surface().catalog()
    assert cat == [{"name": "echo", "description": "echoes args", "schema": {"type": "object"}}]


def test_unknown_tool_raises():
    with pytest.raises(KeyError):
        _surface().call("nope", {})
