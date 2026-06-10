"""ForgeCode lightweight coding harness."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ipw.agents.dspy_rlm import DSPyRLM
from ipw.core.registry import AgentRegistry
from ipw.core.types import AgentRunResult


@AgentRegistry.register("forgecode")
class ForgeCode(DSPyRLM):
    """Coding-focused harness that asks for executable code or patches.

    The implementation reuses the IPW tool/event loop from ``dspy-rlm`` and
    adds coding-specific instructions plus an optional per-query workspace.
    """

    def __init__(self, *args: Any, instructions: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(
            *args,
            instructions=instructions
            or (
                "You are a coding agent. Solve the task with minimal, correct changes. "
                "For repository repair tasks, return a unified diff. For programming "
                "contest tasks, return complete Python code only."
            ),
            **kwargs,
        )
        self._workspace: Optional[Path] = None

    def set_workspace(self, workspace_path: str) -> None:
        self._workspace = Path(workspace_path)

    def run(self, input: str, **kwargs: Any) -> AgentRunResult:
        if self._workspace is not None:
            input = (
                f"Workspace: {self._workspace}\n"
                "Use file/code tools against this workspace when needed.\n\n"
                f"{input}"
            )
        return super().run(input, **kwargs)


__all__ = ["ForgeCode"]
