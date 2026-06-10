"""Tests for tools/docker_shell_exec.py.

Requires Docker; skipped if `docker` is not on PATH.
"""

from __future__ import annotations

import asyncio
import shutil

import pytest

from ipw.tools.docker_shell_exec import DockerShellExecTool

pytestmark = pytest.mark.skipif(
    shutil.which("docker") is None, reason="docker CLI not available"
)


class TestDockerShellExecMetadata:
    def test_spec(self) -> None:
        spec = DockerShellExecTool.spec
        assert spec.name == "docker_shell_exec"
        assert spec.requires_docker is True
        assert spec.sandboxed is True
        assert spec.side_effect_conflict is True
        assert "command" in spec.parameters

    def test_empty_command_returns_error(self) -> None:
        result = asyncio.run(DockerShellExecTool().run(command=""))
        assert result.success is False


@pytest.mark.integration
class TestDockerShellExecIntegration:
    def test_echo_runs_in_container(self) -> None:
        result = asyncio.run(DockerShellExecTool().run(command="echo hello-docker"))
        assert result.success is True
        assert "hello-docker" in result.content

    def test_failing_command_returns_error(self) -> None:
        result = asyncio.run(DockerShellExecTool().run(command="exit 3"))
        assert result.success is False

    def test_timeout(self) -> None:
        result = asyncio.run(DockerShellExecTool().run(command="sleep 30", timeout=2))
        assert result.success is False
        assert "timeout" in (result.error or "").lower()

    def test_workspace_mounted(self, tmp_path) -> None:
        # A file in the workspace should be visible inside the container at /workspace
        (tmp_path / "marker.txt").write_text("found-it")
        tool = DockerShellExecTool()
        tool.set_workspace(str(tmp_path))
        result = asyncio.run(tool.run(command="cat /workspace/marker.txt"))
        assert result.success is True
        assert "found-it" in result.content
