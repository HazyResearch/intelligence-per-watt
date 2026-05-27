"""docker_shell_exec — run a shell command in a fresh Docker container per call.

Ported from OpenJarvis. Each call invokes `docker run --rm` with a fresh
container. Distinct from code_interpreter_docker which runs Python; this tool
runs arbitrary shell commands via `sh -c`.

The default image is ubuntu:22.04; override via the `image` parameter.
Containers run with --network=none and resource caps (--memory=512m --cpus=1)
to limit blast radius from agent-provided commands.

When a workspace directory is set via set_workspace(), it is mounted at
/workspace inside the container and the container starts there.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from ipw.tools.base import BaseTool, ToolResult, ToolSpec
from ipw.tools.registry import ToolRegistry

_DEFAULT_IMAGE = "ubuntu:22.04"
_MAX_OUTPUT_BYTES = 64 * 1024


@ToolRegistry.register("docker_shell_exec")
class DockerShellExecTool(BaseTool):
    spec = ToolSpec(
        name="docker_shell_exec",
        description=(
            "Execute a shell command in a fresh, sandboxed Docker container. "
            "Stateless — each call gets a new container. Distinct from "
            "code_interpreter_docker which runs Python; this runs arbitrary "
            "shell commands via sh -c."
        ),
        parameters={
            "command": {"type": "string", "description": "Shell command to execute"},
            "image": {"type": "string", "description": f"Docker image (default: {_DEFAULT_IMAGE})"},
            "timeout": {"type": "number", "description": "Seconds (default: 60)"},
        },
        requires_docker=True,
        sandboxed=True,
        side_effect_conflict=True,
    )

    async def run(
        self,
        command: str = "",
        image: Optional[str] = None,
        timeout: float = 60.0,
        **kwargs,
    ) -> ToolResult:
        if not command:
            return ToolResult(content="", success=False, error="empty command")

        image = image or _DEFAULT_IMAGE
        argv = [
            "docker", "run", "--rm",
            "--network=none",
            "--memory=512m",
            "--cpus=1",
        ]

        # Mount workspace at /workspace if set
        if self._default_cwd:
            argv += ["-v", f"{self._default_cwd}:/workspace", "-w", "/workspace"]

        argv += ["-i", image, "sh", "-c", command]

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


__all__ = ["DockerShellExecTool"]
