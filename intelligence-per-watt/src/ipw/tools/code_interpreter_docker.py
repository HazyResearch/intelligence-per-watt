"""code_interpreter_docker — run Python in a fresh Docker container per call.

Ported from /home/ubuntu/lambda-stanford/jonsf/OpenJarvis/src/openjarvis/tools/code_interpreter_docker.py
Each call invokes `docker run --rm` with a fresh container. Use ReplTool if
you need state across calls; this tool is intentionally stateless for safety.

The default image is python:3.13-slim; override via the `image` parameter.
Containers run with --network=none and resource caps (--memory=512m --cpus=1)
to limit blast radius from agent-provided code.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from ipw.tools.base import BaseTool, ToolResult, ToolSpec
from ipw.tools.registry import ToolRegistry

_DEFAULT_IMAGE = "python:3.13-slim"
_MAX_OUTPUT_BYTES = 64 * 1024


# Note: this tool ignores set_workspace() — Docker provides its own filesystem
# isolation. The container starts in / by default; agents can write to /tmp
# within the container without affecting the host.
@ToolRegistry.register("code_interpreter_docker")
class CodeInterpreterDockerTool(BaseTool):
    spec = ToolSpec(
        name="code_interpreter_docker",
        description=(
            "Execute Python code in a fresh, sandboxed Docker container. "
            "Stateless — each call gets a new container. Use repl for stateful work."
        ),
        parameters={
            "code": {"type": "string", "description": "Python code to execute"},
            "image": {"type": "string", "description": f"Docker image (default: {_DEFAULT_IMAGE})"},
            "timeout": {"type": "number", "description": "Seconds (default: 60)"},
        },
        requires_docker=True,
        sandboxed=True,
    )

    async def run(
        self,
        code: str = "",
        image: Optional[str] = None,
        timeout: float = 60.0,
        **kwargs,
    ) -> ToolResult:
        if not code:
            return ToolResult(content="", success=False, error="empty code")

        image = image or _DEFAULT_IMAGE
        argv = [
            "docker", "run", "--rm",
            "--network=none",
            "--memory=512m",
            "--cpus=1",
            "-i",
            image,
            "python", "-c", code,
        ]
        proc = await asyncio.create_subprocess_exec(
            *argv,
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
            return ToolResult(content="", success=False, error=f"timeout after {timeout}s")

        text = (stdout or b"").decode("utf-8", errors="replace")[:_MAX_OUTPUT_BYTES]
        if proc.returncode != 0:
            return ToolResult(
                content=text, success=False,
                error=f"container exit {proc.returncode}",
                metadata={"returncode": proc.returncode},
            )
        return ToolResult(content=text, success=True, metadata={"returncode": 0})


__all__ = ["CodeInterpreterDockerTool"]
