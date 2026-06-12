"""Exception hierarchy and classifier for the Executor.

Retry policy:
- RetryableError: transient (network, timeout, rate limit) — retried up to 3 times
- FatalError: structural (malformed output, missing tool, assertion) — no retry
- classify_error() inspects an arbitrary exception and returns one of the above
"""

from __future__ import annotations


class ExecutorError(Exception):
    """Base class for all Executor-raised exceptions."""


class RetryableError(ExecutorError):
    """Transient error — the executor will retry."""


class FatalError(ExecutorError):
    """Non-transient error — the executor will not retry."""


class MalformedOutputError(FatalError):
    """The agent emitted output that could not be parsed."""


class ToolNotFoundError(FatalError):
    """The agent requested a tool that is not registered."""


class RetryExhaustedError(FatalError):
    """A retryable error did not resolve within the allowed retry budget.

    Distinct from a bare FatalError so callers can tell "fatal from attempt 1"
    apart from "retried N times and still failing" without parsing log strings.
    """


_RETRYABLE_TYPES: tuple[type[Exception], ...] = (
    TimeoutError,    # also covers socket.timeout and asyncio.TimeoutError aliases
    ConnectionError,
)


def classify_error(exc: Exception) -> ExecutorError:
    """Classify an arbitrary exception as RetryableError or FatalError.

    If exc is already an ExecutorError, returns it unchanged. Otherwise wraps
    it in the most specific applicable subclass. Unknown exceptions default to
    FatalError (safer than silent retry on a structural bug). Callers must
    not pass KeyboardInterrupt/SystemExit — those should propagate unchanged,
    enforced via the Exception-narrowed signature.
    """
    if isinstance(exc, ExecutorError):
        return exc
    if isinstance(exc, _RETRYABLE_TYPES):
        return RetryableError(str(exc))
    return FatalError(str(exc))


__all__ = [
    "ExecutorError",
    "RetryableError",
    "FatalError",
    "MalformedOutputError",
    "ToolNotFoundError",
    "RetryExhaustedError",
    "classify_error",
]
