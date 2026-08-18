"""Run an asyncio event loop on a background thread.

:class:`InferenceClient.stream_chat_completion` is synchronous, but some
backends expose async-only APIs (Apple's Foundation Models SDK is one). This
bridges the two without an ``asyncio.run`` per call, which would tear down and
rebuild a loop for every request and break SDKs that hold loop-bound state.

``clients/vllm.py`` carries its own private copy of this class. It is left
alone deliberately: switching it over is a pure move, but ``vllm`` cannot be
imported on Apple Silicon, so the change could not be verified here.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any


class AsyncLoopRunner:
    """Own an event loop on a daemon thread and run coroutines against it."""

    def __init__(self, name: str = "ipw-async") -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name=name, daemon=True)
        self._thread.start()

    def run(self, coro) -> Any:
        """Submit ``coro`` to the background loop and block for its result."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def shutdown(self) -> None:
        if self._loop.is_closed():
            return

        async def _drain() -> None:
            current = asyncio.current_task()
            tasks = [
                task
                for task in asyncio.all_tasks()
                if task is not current and not task.done()
            ]
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        try:
            future = asyncio.run_coroutine_threadsafe(_drain(), self._loop)
            future.result(timeout=5.0)
        except Exception:  # pragma: no cover - shutdown is best-effort
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2.0)
        self._loop.close()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()


__all__ = ["AsyncLoopRunner"]
