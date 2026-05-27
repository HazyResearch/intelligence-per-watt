"""Tests for execution/parsers.py — structured-text + function-calling parsers."""

from __future__ import annotations

import json

import pytest

from ipw.execution.errors import MalformedOutputError
from ipw.execution.parsers import (
    ParsedTurn,
    parse_function_calling,
    parse_structured_text,
)


class TestParsedTurn:
    def test_final_answer(self) -> None:
        p = ParsedTurn(final_answer="done", tool_calls=[])
        assert p.is_final is True
        assert p.tool_calls == []

    def test_tool_calls_no_final(self) -> None:
        p = ParsedTurn(final_answer=None, tool_calls=[{"name": "x", "input": {}}])
        assert p.is_final is False
        assert len(p.tool_calls) == 1


class TestParseStructuredText:
    def test_thought_action_input(self) -> None:
        text = (
            "THOUGHT: I need to compute 2+2.\n"
            "ACTION: calculator\n"
            "INPUT: {\"expr\": \"2+2\"}\n"
        )
        result = parse_structured_text(text)
        assert result.is_final is False
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "calculator"
        assert result.tool_calls[0]["input"] == {"expr": "2+2"}

    def test_final_answer_marker(self) -> None:
        text = "THOUGHT: I have the answer.\nFINAL_ANSWER: 4"
        result = parse_structured_text(text)
        assert result.is_final is True
        assert result.final_answer == "4"
        assert result.tool_calls == []

    def test_multiline_input_json(self) -> None:
        text = (
            "THOUGHT: x\n"
            "ACTION: shell_exec\n"
            "INPUT: {\n  \"command\": \"ls -la\",\n  \"cwd\": \"/tmp\"\n}\n"
        )
        result = parse_structured_text(text)
        assert result.tool_calls[0]["input"] == {"command": "ls -la", "cwd": "/tmp"}

    def test_no_action_no_final_raises_malformed(self) -> None:
        text = "THOUGHT: I'm thinking but never act."
        with pytest.raises(MalformedOutputError):
            parse_structured_text(text)

    def test_action_without_input_raises_malformed(self) -> None:
        text = "THOUGHT: x\nACTION: shell\n"
        with pytest.raises(MalformedOutputError):
            parse_structured_text(text)

    def test_invalid_input_json_raises_malformed(self) -> None:
        text = "THOUGHT: x\nACTION: shell\nINPUT: not-json {oops"
        with pytest.raises(MalformedOutputError):
            parse_structured_text(text)

    def test_multiple_actions_zip_correctly(self) -> None:
        """Two sequential ACTION/INPUT pairs zip in document order — the
        contract test for Executor's parallel dispatch path."""
        text = "ACTION: a\nINPUT: {\"x\": 1}\nACTION: b\nINPUT: {\"y\": 2}\n"
        result = parse_structured_text(text)
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0] == {"name": "a", "input": {"x": 1}}
        assert result.tool_calls[1] == {"name": "b", "input": {"y": 2}}


class TestParseFunctionCalling:
    def test_extract_tool_calls_from_openai_response(self) -> None:
        raw = {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "shell_exec", "arguments": json.dumps({"cmd": "ls"})},
                },
            ],
        }
        result = parse_function_calling(raw)
        assert result.is_final is False
        assert result.tool_calls[0]["name"] == "shell_exec"
        assert result.tool_calls[0]["input"] == {"cmd": "ls"}

    def test_final_content_no_tool_calls(self) -> None:
        raw = {"content": "the answer is 42", "tool_calls": []}
        result = parse_function_calling(raw)
        assert result.is_final is True
        assert result.final_answer == "the answer is 42"

    def test_malformed_arguments_json_raises(self) -> None:
        raw = {
            "content": None,
            "tool_calls": [{"function": {"name": "x", "arguments": "not-json{"}}],
        }
        with pytest.raises(MalformedOutputError):
            parse_function_calling(raw)

    def test_missing_function_key_raises_malformed(self) -> None:
        """Entries without 'function' key must not silently produce name=None."""
        raw = {"content": None, "tool_calls": [{"id": "x", "type": "function"}]}
        with pytest.raises(MalformedOutputError):
            parse_function_calling(raw)

    def test_empty_function_name_raises_malformed(self) -> None:
        """function.name=='' must also fail (treated identically to missing)."""
        raw = {
            "content": None,
            "tool_calls": [{"function": {"name": "", "arguments": "{}"}}],
        }
        with pytest.raises(MalformedOutputError):
            parse_function_calling(raw)
