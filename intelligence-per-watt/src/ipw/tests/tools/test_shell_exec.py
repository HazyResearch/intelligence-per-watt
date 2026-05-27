"""Tests for tools/shell_exec.py."""

from __future__ import annotations

import asyncio

from ipw.tools.shell_exec import ShellExecTool


class TestShellExecTool:
    def test_spec(self) -> None:
        spec = ShellExecTool.spec
        assert spec.name == "shell_exec"
        assert spec.side_effect_conflict is True
        assert "command" in spec.parameters

    def test_simple_command_success(self) -> None:
        result = asyncio.run(ShellExecTool().run(command="echo hello"))
        assert result.success is True
        assert "hello" in result.content

    def test_failing_command_returns_error(self) -> None:
        result = asyncio.run(ShellExecTool().run(command="false"))
        assert result.success is False
        assert result.error is not None

    def test_timeout(self) -> None:
        result = asyncio.run(ShellExecTool().run(command="sleep 10", timeout=0.1))
        assert result.success is False
        assert "timeout" in (result.error or "").lower()

    def test_cwd_argument(self, tmp_path) -> None:
        result = asyncio.run(ShellExecTool().run(command="pwd", cwd=str(tmp_path)))
        assert result.success is True
        # macOS adds /private prefix; tmp_path resolves to either.
        from pathlib import Path
        resolved = Path(str(tmp_path)).resolve()
        assert str(tmp_path) in result.content or str(resolved) in result.content

    def test_stderr_captured(self) -> None:
        result = asyncio.run(ShellExecTool().run(command="echo err >&2"))
        # Output combines stdout+stderr; "err" should appear.
        assert "err" in result.content

    def test_empty_command_rejected(self) -> None:
        result = asyncio.run(ShellExecTool().run(command=""))
        assert result.success is False

    def test_default_cwd_used_when_no_cwd_arg(self, tmp_path) -> None:
        """set_workspace() default_cwd is used when run() doesn't override cwd."""
        tool = ShellExecTool()
        tool.set_workspace(str(tmp_path))
        result = asyncio.run(tool.run(command="pwd"))
        assert result.success is True
        from pathlib import Path
        resolved = Path(str(tmp_path)).resolve()
        assert str(tmp_path) in result.content or str(resolved) in result.content

    def test_explicit_cwd_overrides_workspace(self, tmp_path) -> None:
        """An explicit cwd arg takes precedence over set_workspace()."""
        tool = ShellExecTool()
        tool.set_workspace("/should/be/overridden")
        result = asyncio.run(tool.run(command="pwd", cwd=str(tmp_path)))
        assert result.success is True
        from pathlib import Path
        resolved = Path(str(tmp_path)).resolve()
        assert str(tmp_path) in result.content or str(resolved) in result.content
