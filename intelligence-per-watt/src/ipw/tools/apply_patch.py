"""apply_patch — apply a unified-diff patch to files in a working directory.

Ported from /home/ubuntu/lambda-stanford/jonsf/OpenJarvis/src/openjarvis/tools/apply_patch.py
Uses GNU `patch` via subprocess. Validates that the patch text starts with a
recognizable diff header before invocation.
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

from ipw.tools.base import BaseTool, ToolResult, ToolSpec
from ipw.tools.registry import ToolRegistry


@ToolRegistry.register("apply_patch")
class ApplyPatchTool(BaseTool):
    spec = ToolSpec(
        name="apply_patch",
        description="Apply a unified-diff patch to files in the working directory.",
        parameters={
            "patch": {"type": "string", "description": "Unified-diff patch text"},
            "cwd": {"type": "string", "description": "Working directory"},
        },
        side_effect_conflict=True,
    )

    async def run(
        self,
        patch: str = "",
        cwd: Optional[str] = None,
        timeout: float = 30.0,
        **kwargs,
    ) -> ToolResult:
        if not patch or not patch.lstrip().startswith(("---", "diff", "Index:")):
            return ToolResult(content="", success=False, error="patch text missing diff header")

        cwd = cwd or self._default_cwd or os.getcwd()
        proc = await asyncio.create_subprocess_exec(
            "patch", "-p1", "--batch",
            cwd=cwd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(patch.encode("utf-8")), timeout=timeout
            )
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
                error=(err.strip() or out.strip() or f"patch exit {proc.returncode}"),
                metadata={"returncode": proc.returncode},
            )
        return ToolResult(content=out, success=True, metadata={"returncode": 0})


__all__ = ["ApplyPatchTool"]
