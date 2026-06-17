"""ForgeCode lightweight coding harness."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Optional

from ipw.agents.dspy_rlm import DSPyRLM
from ipw.core.registry import AgentRegistry
from ipw.core.types import AgentRunResult

FORGECODE_DEFAULT_INSTRUCTIONS = """You are a coding agent. Solve the task with minimal, correct changes.
For repository repair tasks, edit files in the provided workspace and return a unified diff.
For programming contest tasks, return complete Python code only.

Repository repair pacing:
- Inspect only the files needed to identify the fix.
- Do not spend the whole run reading files; after one or two inspection commands, make the smallest plausible edit.
- After repeated read-only inspections with an empty git diff, read-only commands will be rejected until you edit the workspace.
- Before finishing, run a targeted check or `git diff --check` when practical.
- If time or turns are nearly exhausted, prioritize leaving a concrete workspace edit and returning the current diff.

Tool-call contract:
- Emit exactly one tool call per response when using tools.
- Put Action: and Action Input: on separate lines.
- Do not include chat-template markers such as <|tool_call|> or <|channel|>.
- Do not emit Final: until the patch or answer is ready.

Valid tool-call examples:

Action: bash
Action Input: grep -R "def target_function" -n .

Action: file_read
Action Input: src/package/module.py

Action: file_write
Action Input: {"path": "src/package/module.py", "content": "replacement file contents"}

When finished with a repository repair, respond exactly like:
Final:
```diff
diff --git a/src/package/module.py b/src/package/module.py
--- a/src/package/module.py
+++ b/src/package/module.py
@@ -1,2 +1,2 @@
-old line
+new line
```
"""


_READ_ONLY_INSPECTION_LIMIT = 3
_READ_ONLY_BASH_RE = re.compile(
    r"^(?:"
    r"cat|"
    r"grep\b|"
    r"rg\b|"
    r"find\b|"
    r"ls\b|"
    r"head\b|"
    r"tail\b|"
    r"sed\s+-n\b|"
    r"git\s+(?:diff|grep|log|show|status)\b"
    r")"
)


def _strip_leading_cd(command: str) -> str:
    return re.sub(r"^\s*cd\s+\S+\s*&&\s*", "", command.strip())


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
            or FORGECODE_DEFAULT_INSTRUCTIONS,
            **kwargs,
        )
        self._workspace: Optional[Path] = None

    def set_workspace(self, workspace_path: str) -> None:
        self._workspace = Path(workspace_path)

    def _workspace_is_git_repo(self) -> bool:
        return self._workspace is not None and (self._workspace / ".git").exists()

    def _workspace_has_diff(self) -> bool:
        if not self._workspace_is_git_repo():
            return False
        completed = subprocess.run(
            ["git", "diff", "--quiet"],
            cwd=self._workspace,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return completed.returncode == 1

    def _workspace_diff(self) -> str:
        if not self._workspace_is_git_repo():
            return ""
        completed = subprocess.run(
            ["git", "diff", "--binary"],
            cwd=self._workspace,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        return completed.stdout if completed.returncode == 0 else ""

    def _is_read_only_inspection(self, tool_name: str, tool_input: str) -> bool:
        normalized_tool = tool_name.lower().strip()
        if normalized_tool == "file_read":
            return True
        if normalized_tool not in {"bash", "shell"}:
            return False
        command = _strip_leading_cd(str(tool_input))
        return bool(_READ_ONLY_BASH_RE.match(command))

    def _pre_tool_observation(
        self,
        *,
        tool_name: str,
        tool_input: str,
        turn_index: int,
        tools_attempted: int,
    ) -> Optional[str]:
        if (
            not self._workspace_is_git_repo()
            or tools_attempted <= _READ_ONLY_INSPECTION_LIMIT
            or self._workspace_has_diff()
            or not self._is_read_only_inspection(tool_name, tool_input)
        ):
            return None
        return (
            "Read-only inspection budget exhausted and git diff is still empty. "
            "Your next response must modify the workspace with a minimal source edit "
            "using bash, shell, or file_write. Do not inspect another file until a "
            "workspace diff exists."
        )

    def _turn_limit_final_prompt(self) -> Optional[str]:
        diff = self._workspace_diff()
        if diff.strip():
            return (
                "Tool budget exhausted. Do not call another tool. Return exactly "
                "the current workspace change as a Final fenced unified diff:\n"
                "Final:\n```diff\n"
                f"{diff[-12000:]}\n"
                "```"
            )
        return (
            "Tool budget exhausted and git diff is still empty. Do not call another "
            "tool. Based on the issue and observations already shown, return your "
            "best minimal unified diff patch attempt now. Respond exactly as:\n"
            "Final:\n```diff\n"
            "diff --git a/path b/path\n"
            "--- a/path\n"
            "+++ b/path\n"
            "@@ ...\n"
            "-old\n"
            "+new\n"
            "```"
        )

    def run(self, input: str, **kwargs: Any) -> AgentRunResult:
        if self._workspace is not None:
            input = (
                f"Workspace: {self._workspace}\n"
                "Use file/code tools against this workspace when needed.\n\n"
                f"{input}"
            )
        return super().run(input, **kwargs)


__all__ = ["ForgeCode"]
