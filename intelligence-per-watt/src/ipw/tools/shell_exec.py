"""shell_exec — run a shell command via subprocess and capture combined output.

Ported from /home/ubuntu/lambda-stanford/jonsf/OpenJarvis/src/openjarvis/tools/shell_exec.py
Adapted: returns ToolResult; uses ipw's BaseTool/ToolSpec; marked side_effect_conflict=True
because two concurrent shell_exec invocations on the same workspace can step on
each other.
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

from ipw.tools.base import BaseTool, ToolResult, ToolSpec
from ipw.tools.registry import ToolRegistry

DEFAULT_TIMEOUT_S = 60.0
MAX_OUTPUT_BYTES = 64 * 1024


@ToolRegistry.register("shell_exec")
class ShellExecTool(BaseTool):
    spec = ToolSpec(
        name="shell_exec",
        description=(
            "Execute a shell command and return combined stdout+stderr. "
            "Defaults to a 60-second timeout. Output truncated to 64KB."
        ),
        parameters={
            "command": {"type": "string", "description": "Shell command to run"},
            "cwd": {"type": "string", "description": "Working directory (default: $PWD)"},
            "timeout": {"type": "number", "description": "Seconds (default: 60)"},
        },
        side_effect_conflict=True,
        requires_network=False,
    )

    async def run(
        self,
        command: str = "",
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> ToolResult:
        if not command:
            return ToolResult(content="", success=False, error="empty command")

        cwd = cwd or self._default_cwd or os.getcwd()
        timeout = timeout or DEFAULT_TIMEOUT_S

        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
            return ToolResult(
                content="",
                success=False,
                error=f"timeout after {timeout}s",
            )

        text = stdout.decode("utf-8", errors="replace") if stdout else ""
        truncated = text[:MAX_OUTPUT_BYTES]

        if proc.returncode != 0:
            return ToolResult(
                content=truncated,
                success=False,
                error=f"exit code {proc.returncode}",
                metadata={"returncode": proc.returncode, "command": command},
            )

        return ToolResult(
            content=truncated,
            success=True,
            metadata={"returncode": 0, "command": command},
        )


__all__ = ["ShellExecTool"]
