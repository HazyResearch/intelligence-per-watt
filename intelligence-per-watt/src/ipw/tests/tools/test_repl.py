"""Tests for tools/repl.py."""

from __future__ import annotations

import asyncio

from ipw.tools.repl import ReplTool


class TestReplTool:
    def test_spec(self) -> None:
        spec = ReplTool.spec
        assert spec.name == "repl"
        assert spec.side_effect_conflict is True

    def test_simple_expression(self) -> None:
        tool = ReplTool()
        try:
            result = asyncio.run(tool.run(code="print(1+1)"))
            assert result.success is True
            assert "2" in result.content
        finally:
            asyncio.run(tool.shutdown())

    def test_state_persists_across_calls(self) -> None:
        tool = ReplTool()
        try:
            asyncio.run(tool.run(code="x = 5"))
            result = asyncio.run(tool.run(code="print(x * 2)"))
            assert result.success is True
            assert "10" in result.content
        finally:
            asyncio.run(tool.shutdown())

    def test_syntax_error_returns_failure(self) -> None:
        tool = ReplTool()
        try:
            result = asyncio.run(tool.run(code="def "))
            assert result.success is False
        finally:
            asyncio.run(tool.shutdown())

    def test_timeout_kills_long_running_code(self) -> None:
        tool = ReplTool()
        try:
            result = asyncio.run(tool.run(code="import time; time.sleep(10)", timeout=0.5))
            assert result.success is False
            assert "timeout" in (result.error or "").lower()
        finally:
            asyncio.run(tool.shutdown())
