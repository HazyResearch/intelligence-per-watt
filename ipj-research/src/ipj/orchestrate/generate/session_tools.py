"""Container-backed tool surface: a ToolBackend over a terminal-bench TmuxSession."""

from __future__ import annotations

import re
from typing import Any

from ipj.orchestrate.harness.tools import ToolOutcome, ToolSurface, bash_tool_infos

_RC = re.compile(r"__RC=(\d+)")
# get_incremental_output() prefixes its return with one of these labels; strip them so
# they don't leak into the trace content.
_INCREMENTAL_PREFIXES = ("New Terminal Output:\n", "Current Terminal Screen:\n")


def _strip_prefix(text: str) -> str:
    for prefix in _INCREMENTAL_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


class SessionShellBackend:
    """A ToolBackend that runs shell commands inside a terminal-bench TmuxSession (the container)."""

    def __init__(self, session: Any, *, timeout_s: float = 300.0) -> None:
        self._session = session
        self._timeout_s = timeout_s

    def run(self, **kwargs: Any) -> ToolOutcome:
        command = kwargs.get("command", "")
        # Append a return-code marker so we can derive success from the captured output.
        wrapped = f"{command}; echo __RC=$?"
        # Send as a list ending in "Enter": TmuxSession types a bare string but never
        # executes it (no newline), and block=True is silently ignored unless the final
        # key is an Enter/newline. A list ending in "Enter" both runs the command and
        # makes the call block until completion.
        try:
            self._session.send_keys([wrapped, "Enter"], block=True, max_timeout_sec=self._timeout_s)
        except TimeoutError:
            # A long-running or hung command must NOT crash the whole generation. Interrupt it
            # (best-effort Ctrl-C), then surface the timeout to the brain as a FAILED command so it
            # can adapt, keeping the session usable for the next turn.
            try:
                self._session.send_keys(["C-c"], block=False)
            except Exception:
                pass
            partial = self._current_output(self._capture(), [])
            return ToolOutcome(
                content=f"[command timed out after {self._timeout_s:.0f}s and was interrupted]\n{partial}",
                success=False,
            )

        output = self._capture()
        matches = list(_RC.finditer(output))
        # Use the LAST marker: a full scrollback capture returns every past __RC, and the
        # first match is the oldest command's exit code, not this turn's.
        last = matches[-1] if matches else None
        success = last is not None and last.group(1) == "0"
        content = self._current_output(output, matches)
        return ToolOutcome(content=content, success=success)

    def _capture(self) -> str:
        # Prefer get_incremental_output(): it returns only text new since the previous
        # call, so we don't re-emit the entire accumulated scrollback every turn. Fall
        # back to a full capture for sessions that don't expose it.
        getter = getattr(self._session, "get_incremental_output", None)
        if callable(getter):
            return getter()
        return self._session.capture_pane(capture_entire=True)

    def _current_output(self, output: str, matches: list[re.Match[str]]) -> str:
        if not matches:
            return _strip_prefix(output).strip()
        # Return only this command's output: the text after the previous marker (if the
        # capture still carries stale history) and up to this turn's marker.
        start = matches[-2].end() if len(matches) >= 2 else 0
        segment = output[start:matches[-1].start()]
        return _strip_prefix(segment).strip()


def build_session_tool_surface(session: Any) -> ToolSurface:
    backend = SessionShellBackend(session)
    backends = {"bash": backend, "code_interpreter": backend}
    return ToolSurface(backends, bash_tool_infos(container=True))
