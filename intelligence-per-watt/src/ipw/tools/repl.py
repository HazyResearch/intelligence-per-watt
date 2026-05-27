"""Persistent Python REPL tool — maintains state across calls within a session.

Each ReplTool instance owns a single persistent Python subprocess. Variables,
functions, and imports survive across run() calls *including* across separate
asyncio.run() invocations on the same ReplTool instance. shutdown() terminates
the subprocess cleanly.

Boot loop protocol
------------------
The subprocess reads request frames from stdin (one line per request):

    <SENTINEL>:<base64-encoded code>\n

It evaluates the code in a persistent namespace dict, writes stdout/stderr
output, and then emits an end-of-response marker:

    __DONE__<SENTINEL>:<status>\n

on a line by itself. The parent reads lines until it sees that marker.

On timeout the parent kills the subprocess; the next run() call respawns it
automatically.

I/O strategy
------------
A dedicated stdout-reader thread continuously drains the child's stdout into a
queue.Queue. The main thread dequeues lines within a deadline. This avoids the
select() buffering pitfall: Python's TextIOWrapper internal buffer means select()
can report "not ready" even when the next line is already buffered inside
Python's I/O layer. Using a reader thread + Queue side-steps that entirely.

asyncio.to_thread offloads blocking work so the event loop stays responsive.
A threading.Lock serialises concurrent run() calls (not loop-bound, works across
multiple asyncio.run() invocations).
"""

from __future__ import annotations

import asyncio
import base64
import os
import queue
import signal
import subprocess
import sys
import textwrap
import threading
import uuid
from typing import Optional

from ipw.tools.base import BaseTool, ToolResult, ToolSpec
from ipw.tools.registry import ToolRegistry

DEFAULT_TIMEOUT_S: float = 30.0
MAX_OUTPUT_CHARS: int = 32_000

# The boot script is injected verbatim into the child process via -c "...".
# Key design points (ported from OJ boot loop logic):
#   - Reads one line at a time from stdin (line-buffered).
#   - Decodes base64 to recover the real code (handles multi-line snippets).
#   - Redirects stdout/stderr into StringIO during exec so output is captured.
#   - Tries eval first (expression display) then falls back to exec.
#   - Writes captured output, then writes the end sentinel.
#   - Loops forever until stdin closes (parent dies or shutdown() sends EOF).
_BOOT_SCRIPT = textwrap.dedent(r"""
import sys, base64, io, traceback
from contextlib import redirect_stdout, redirect_stderr

_NS = {"__name__": "__main__"}
_DONE_PREFIX = "__DONE__"

def _run_frame(sentinel, code_b64):
    code = base64.b64decode(code_b64).decode("utf-8")
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    status = "ok"
    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            try:
                compiled = compile(code, "<repl>", "eval")
                val = eval(compiled, _NS)
                if val is not None:
                    print(repr(val))
            except SyntaxError:
                compiled = compile(code, "<repl>", "exec")
                exec(compiled, _NS)
    except Exception:
        stderr_buf.write(traceback.format_exc())
        status = "error"

    out = stdout_buf.getvalue()
    err = stderr_buf.getvalue()
    combined = out + err
    if combined:
        sys.stdout.write(combined)
        if not combined.endswith("\n"):
            sys.stdout.write("\n")
    sys.stdout.write(f"{_DONE_PREFIX}{sentinel}:{status}\n")
    sys.stdout.flush()

for raw_line in sys.stdin:
    line = raw_line.rstrip("\n")
    colon = line.index(":")
    sentinel = line[:colon]
    code_b64 = line[colon+1:]
    _run_frame(sentinel, code_b64)
""").strip()

_EOF_SENTINEL = object()  # poison pill for reader thread queue


class _ReplProcess:
    """Wraps a persistent Python subprocess with a reader thread + queue."""

    def __init__(self, cwd: Optional[str] = None) -> None:
        self._proc = subprocess.Popen(
            [sys.executable, "-u", "-c", _BOOT_SCRIPT],
            cwd=cwd,  # falls through to None (parent cwd) if not set
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,  # line-buffered
        )
        self._queue: queue.Queue = queue.Queue()
        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()

    def _reader_loop(self) -> None:
        """Drain child stdout into self._queue; runs in a daemon thread."""
        try:
            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                self._queue.put(line.rstrip("\n"))
        except Exception:
            pass
        finally:
            self._queue.put(_EOF_SENTINEL)

    @property
    def alive(self) -> bool:
        return self._proc.poll() is None

    def send(self, frame: str) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(frame)
        self._proc.stdin.flush()

    def readline(self, timeout_s: float) -> "str | None | object":
        """Return next line from stdout queue, timeout sentinel, or EOF sentinel.

        Returns:
            str   — a line of output (may be empty string for blank lines)
            None  — timed out waiting for output
            _EOF_SENTINEL (the module-level object) — subprocess stdout closed

        Callers must use ``item is _EOF_SENTINEL`` to detect EOF, NOT ``item == ""``,
        because the child can emit genuine empty lines (e.g. blank lines in tracebacks).
        """
        try:
            item = self._queue.get(timeout=timeout_s)
        except queue.Empty:
            return None  # timeout
        # Return _EOF_SENTINEL as-is so the caller can distinguish it from ""
        return item  # type: ignore[return-value]

    def kill(self) -> None:
        """Send SIGKILL; do not wait (reader thread is daemon)."""
        try:
            os.kill(self._proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            self._proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass


@ToolRegistry.register("repl")
class ReplTool(BaseTool):
    """Persistent Python REPL backed by a long-lived subprocess.

    State (variables, imports, functions) survives across run() calls on the
    same instance, even across multiple asyncio.run() invocations.
    Call shutdown() when done.
    """

    spec = ToolSpec(
        name="repl",
        description=(
            "Execute Python code in a persistent REPL. "
            "Variables, functions, and imports persist across calls."
        ),
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute.",
                },
                "timeout": {
                    "type": "number",
                    "description": (
                        f"Seconds before killing the subprocess (default {DEFAULT_TIMEOUT_S})."
                    ),
                },
            },
            "required": ["code"],
        },
        side_effect_conflict=True,
    )

    def __init__(self, bus=None) -> None:
        super().__init__(bus=bus)
        self._repl: Optional[_ReplProcess] = None
        # threading.Lock — works across asyncio.run() boundaries (not loop-bound)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        code: str = "",
        timeout: Optional[float] = None,
        **kwargs,
    ) -> ToolResult:
        if not code or not code.strip():
            return ToolResult(content="", success=False, error="empty code")

        timeout = timeout if timeout is not None else DEFAULT_TIMEOUT_S

        # Offload blocking I/O to a thread so the event loop stays free.
        return await asyncio.to_thread(self._run_sync, code, timeout)

    async def shutdown(self) -> None:
        """Terminate the subprocess cleanly."""
        await asyncio.to_thread(self._shutdown_sync)

    # ------------------------------------------------------------------
    # Synchronous implementation (runs in thread pool)
    # ------------------------------------------------------------------

    def _run_sync(self, code: str, timeout: float) -> ToolResult:
        """Thread-safe synchronous run — serialised by self._lock."""
        import time

        with self._lock:
            if self._repl is None or not self._repl.alive:
                self._repl = _ReplProcess(cwd=self._default_cwd)

            repl = self._repl
            sentinel = uuid.uuid4().hex
            code_b64 = base64.b64encode(code.encode("utf-8")).decode("ascii")
            frame = f"{sentinel}:{code_b64}\n"

            try:
                repl.send(frame)
            except (BrokenPipeError, OSError) as exc:
                return ToolResult(
                    content="",
                    success=False,
                    error=f"subprocess stdin broken: {exc}",
                )

            end_marker = f"__DONE__{sentinel}:"
            output_lines: list[str] = []
            status = "ok"
            deadline = time.monotonic() + timeout

            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    repl.kill()
                    self._repl = None
                    return ToolResult(
                        content="",
                        success=False,
                        error=f"timeout after {timeout}s",
                    )

                line = repl.readline(remaining)
                if line is None:
                    # timeout waiting for this line
                    repl.kill()
                    self._repl = None
                    return ToolResult(
                        content="",
                        success=False,
                        error=f"timeout after {timeout}s",
                    )
                if line is _EOF_SENTINEL:
                    # subprocess stdout closed unexpectedly
                    self._repl = None
                    break
                if line.startswith(end_marker):
                    status = line[len(end_marker):].strip()
                    break
                output_lines.append(line)

        combined = "\n".join(output_lines)
        if output_lines:
            combined += "\n"
        if len(combined) > MAX_OUTPUT_CHARS:
            combined = combined[:MAX_OUTPUT_CHARS] + "\n... (output truncated)"

        success = status == "ok"
        return ToolResult(
            content=combined or "(no output)",
            success=success,
            error=None if success else combined or "execution error",
        )

    def _shutdown_sync(self) -> None:
        with self._lock:
            if self._repl is not None:
                self._repl.kill()
                self._repl = None


__all__ = ["ReplTool"]
