"""Tests for execution/errors.py — exception hierarchy + classify_error()."""

from __future__ import annotations

import socket

from ipw.execution.errors import (
    ExecutorError,
    FatalError,
    MalformedOutputError,
    RetryableError,
    ToolNotFoundError,
    classify_error,
)


class TestExceptionHierarchy:
    def test_all_inherit_from_executor_error(self) -> None:
        for exc_cls in (RetryableError, FatalError, MalformedOutputError, ToolNotFoundError):
            assert issubclass(exc_cls, ExecutorError)

    def test_malformed_output_is_fatal(self) -> None:
        assert issubclass(MalformedOutputError, FatalError)

    def test_tool_not_found_is_fatal(self) -> None:
        assert issubclass(ToolNotFoundError, FatalError)


class TestClassifyError:
    def test_timeout_classifies_retryable(self) -> None:
        result = classify_error(TimeoutError("ssh timed out"))
        assert isinstance(result, RetryableError)

    def test_connection_error_classifies_retryable(self) -> None:
        result = classify_error(ConnectionError("server unreachable"))
        assert isinstance(result, RetryableError)

    def test_socket_timeout_classifies_retryable(self) -> None:
        result = classify_error(socket.timeout("read timeout"))
        assert isinstance(result, RetryableError)

    def test_malformed_output_classifies_fatal(self) -> None:
        result = classify_error(MalformedOutputError("bad parse"))
        assert isinstance(result, FatalError)

    def test_tool_not_found_classifies_fatal(self) -> None:
        result = classify_error(ToolNotFoundError("missing"))
        assert isinstance(result, FatalError)

    def test_assertion_error_classifies_fatal(self) -> None:
        result = classify_error(AssertionError("bad invariant"))
        assert isinstance(result, FatalError)

    def test_already_fatal_passes_through(self) -> None:
        original = FatalError("already fatal")
        result = classify_error(original)
        assert result is original

    def test_already_retryable_passes_through(self) -> None:
        original = RetryableError("already retryable")
        result = classify_error(original)
        assert result is original

    def test_unknown_exception_defaults_to_fatal(self) -> None:
        result = classify_error(ValueError("anything"))
        assert isinstance(result, FatalError)

    def test_classify_preserves_message(self) -> None:
        result = classify_error(TimeoutError("specific message"))
        assert "specific message" in str(result)
