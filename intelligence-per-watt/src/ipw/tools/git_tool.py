"""git_tool — invoke specific git subcommands via subprocess.

Ported from /home/ubuntu/lambda-stanford/jonsf/OpenJarvis/src/openjarvis/tools/git_tool.py
Subcommands are allowlisted; arbitrary `args` are passed through but the
subcommand name is fixed at dispatch time so the agent cannot inject shell
metacharacters. Uses asyncio.create_subprocess_exec (no shell).
"""

from __future__ import annotations

import asyncio
from typing import List, Optional

from ipw.tools.base import BaseTool, ToolResult, ToolSpec
from ipw.tools.registry import ToolRegistry

_ALLOWED_SUBCOMMANDS = frozenset({
    "status", "diff", "log", "show", "branch", "add", "commit", "checkout",
    "rev-parse", "ls-files", "stash",
})


@ToolRegistry.register("git_tool")
class GitTool(BaseTool):
    spec = ToolSpec(
        name="git_tool",
        description=(
            "Run a git subcommand and return its output. Allowed subcommands: "
            + ", ".join(sorted(_ALLOWED_SUBCOMMANDS))
        ),
        parameters={
            "subcommand": {"type": "string", "description": "git subcommand name"},
            "cwd": {"type": "string", "description": "Path to the git repo"},
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Extra args for the subcommand",
            },
        },
        side_effect_conflict=True,
    )

    async def run(
        self,
        subcommand: str = "",
        cwd: Optional[str] = None,
        args: Optional[List[str]] = None,
        timeout: float = 60.0,
        **kwargs,
    ) -> ToolResult:
        if subcommand not in _ALLOWED_SUBCOMMANDS:
            return ToolResult(
                content="",
                success=False,
                error=f"subcommand {subcommand!r} not allowed; allowed: "
                      + ", ".join(sorted(_ALLOWED_SUBCOMMANDS)),
            )

        cwd = cwd or self._default_cwd
        argv = ["git", subcommand] + list(args or [])
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
            return ToolResult(content="", success=False, error=f"timeout after {timeout}s")

        out = (stdout or b"").decode("utf-8", errors="replace")
        err = (stderr or b"").decode("utf-8", errors="replace")
        if proc.returncode != 0:
            return ToolResult(
                content=out or err,
                success=False,
                error=err.strip() or f"git exit {proc.returncode}",
                metadata={"returncode": proc.returncode},
            )
        return ToolResult(content=out, success=True, metadata={"returncode": 0})


__all__ = ["GitTool"]
