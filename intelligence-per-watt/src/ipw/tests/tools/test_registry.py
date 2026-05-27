"""Tests for tools/registry.py — ToolRegistry."""

from __future__ import annotations

import pytest

from ipw.tools.base import BaseTool, ToolResult, ToolSpec
from ipw.tools.registry import ToolRegistry


class _StubTool(BaseTool):
    spec = ToolSpec(name="stub", description="stub", parameters={})

    async def run(self, **kwargs):
        return ToolResult(content="ok", success=True)


class TestToolRegistry:
    def setup_method(self) -> None:
        # Snapshot existing registry state so tests don't bleed
        self._saved = dict(ToolRegistry.items())
        ToolRegistry.clear()

    def teardown_method(self) -> None:
        ToolRegistry.clear()
        for name, cls in self._saved.items():
            ToolRegistry.register_value(name, cls)

    def test_register_value_and_get(self) -> None:
        ToolRegistry.register_value("stub", _StubTool)
        assert ToolRegistry.get("stub") is _StubTool

    def test_decorator_registers(self) -> None:
        @ToolRegistry.register("decorated")
        class _Decorated(BaseTool):
            spec = ToolSpec(name="decorated", description="d", parameters={})

            async def run(self, **kwargs):
                return ToolResult(content="d", success=True)

        assert ToolRegistry.get("decorated") is _Decorated

    def test_get_unknown_raises_keyerror(self) -> None:
        with pytest.raises(KeyError):
            ToolRegistry.get("nope")

    def test_has_returns_bool(self) -> None:
        ToolRegistry.register_value("stub", _StubTool)
        assert ToolRegistry.has("stub") is True
        assert ToolRegistry.has("missing") is False

    def test_build_tool_descriptions_returns_list_of_specs(self) -> None:
        ToolRegistry.register_value("stub", _StubTool)
        descs = ToolRegistry.build_tool_descriptions(["stub"])
        assert len(descs) == 1
        assert descs[0]["name"] == "stub"
        assert descs[0]["description"] == "stub"
        assert descs[0]["parameters"] == {}

    def test_build_tool_descriptions_skips_unknown(self) -> None:
        ToolRegistry.register_value("stub", _StubTool)
        descs = ToolRegistry.build_tool_descriptions(["stub", "missing"])
        assert len(descs) == 1
        assert descs[0]["name"] == "stub"

    def test_create_instantiates_tool(self) -> None:
        # RegistryBase.create is inherited; agents will call
        # ToolRegistry.create("shell_exec", bus=bus) at runtime. Smoke test it.
        ToolRegistry.register_value("stub", _StubTool)
        instance = ToolRegistry.create("stub")
        assert isinstance(instance, _StubTool)
