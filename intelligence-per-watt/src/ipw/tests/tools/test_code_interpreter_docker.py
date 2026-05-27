"""Tests for tools/code_interpreter_docker.py.

These tests require Docker. They are skipped if `docker` is not on PATH.
"""

from __future__ import annotations

import asyncio
import shutil

import pytest

from ipw.tools.code_interpreter_docker import CodeInterpreterDockerTool

pytestmark = pytest.mark.skipif(
    shutil.which("docker") is None, reason="docker CLI not available"
)


class TestCodeInterpreterDocker:
    def test_spec(self) -> None:
        spec = CodeInterpreterDockerTool.spec
        assert spec.name == "code_interpreter_docker"
        assert spec.requires_docker is True
        assert spec.sandboxed is True

    @pytest.mark.integration
    def test_simple_python_executes(self) -> None:
        result = asyncio.run(CodeInterpreterDockerTool().run(code="print(2 + 2)"))
        assert result.success is True
        assert "4" in result.content

    @pytest.mark.integration
    def test_timeout(self) -> None:
        result = asyncio.run(CodeInterpreterDockerTool().run(
            code="import time; time.sleep(10)", timeout=2,
        ))
        assert result.success is False
        assert "timeout" in (result.error or "").lower()
