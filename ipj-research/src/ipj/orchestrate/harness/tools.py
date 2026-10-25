"""Non-LLM tool surface: ToolSurface/ToolBackend plus the shared bash/code_interpreter ToolInfo set."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ipj.metrics import UNKNOWN_JOULES_PROVENANCE


@dataclass(slots=True, frozen=True)
class ToolOutcome:
    content: str
    success: bool
    dollars: float | None = None
    latency_s: float | None = None
    gpu_energy_j: float | None = None
    gpu_energy_provenance: str = UNKNOWN_JOULES_PROVENANCE
    gpu_energy_complete: bool = False
    accelerator_device_coverage_complete: bool = False


class ToolBackend(Protocol):
    def run(self, **kwargs: Any) -> ToolOutcome: ...


@dataclass(slots=True, frozen=True)
class ToolInfo:
    name: str
    description: str
    schema: dict[str, Any]


class ToolSurface:
    def __init__(self, backends: dict[str, ToolBackend], infos: dict[str, ToolInfo]) -> None:
        self._backends = backends
        self._infos = infos

    def call(self, tool: str, args: dict[str, Any]) -> ToolOutcome:
        if tool not in self._backends:
            raise KeyError(f"unknown tool {tool!r}; available: {sorted(self._backends)}")
        return self._backends[tool].run(**args)

    def catalog(self) -> list[dict[str, Any]]:
        return [
            {"name": i.name, "description": i.description, "schema": i.schema}
            for i in (self._infos[k] for k in sorted(self._infos))
        ]


def bash_tool_infos(*, container: bool = False) -> dict[str, ToolInfo]:
    """The shared {bash, code_interpreter} ToolInfo surface; ``container`` tweaks wording for in-container tools.

    Both shell tool surfaces (IPW MCP shell and the terminal-bench TmuxSession) advertise the same two
    tools and differ only in description text — this is the single source of that surface.
    """
    where = " in the task container" if container else ""
    code_desc = "run a shell/code command in the container" if container else "run code"
    return {
        "bash": ToolInfo(name="bash", description=f"run a shell command{where}", schema={"type": "object"}),
        "code_interpreter": ToolInfo(name="code_interpreter", description=code_desc, schema={"type": "object"}),
    }
