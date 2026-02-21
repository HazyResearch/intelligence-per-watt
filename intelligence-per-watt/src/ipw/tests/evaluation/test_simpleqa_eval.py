"""Tests for evaluation/simpleqa.py — SimpleQAHandler LLM judge."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ipw.evaluation.simpleqa import SimpleQAHandler, _parse_grade


class TestParseGrade:
    """Test _parse_grade response parsing."""

    def test_correct_yes(self) -> None:
        assert _parse_grade("correct: yes") == "CORRECT"

    def test_correct_no(self) -> None:
        assert _parse_grade("correct: no") == "INCORRECT"

    def test_not_attempted(self) -> None:
        assert _parse_grade("correct: not_attempted") == "NOT_ATTEMPTED"

    def test_case_insensitive(self) -> None:
        assert _parse_grade("correct: YES") == "CORRECT"

    def test_fallback_correct_in_text(self) -> None:
        assert _parse_grade("The answer is CORRECT") == "CORRECT"

    def test_fallback_incorrect_in_text(self) -> None:
        assert _parse_grade("INCORRECT answer") == "INCORRECT"

    def test_fallback_not_attempted(self) -> None:
        assert _parse_grade("NOT_ATTEMPTED response") == "NOT_ATTEMPTED"

    def test_unparseable_defaults_not_attempted(self) -> None:
        assert _parse_grade("random text with no grade") == "NOT_ATTEMPTED"


class TestSimpleQAHandler:
    """Test SimpleQAHandler with mocked LLM client."""

    def _make_handler(self, chat_return: str = "") -> SimpleQAHandler:
        mock_client = MagicMock()
        mock_client.chat.return_value = chat_return
        return SimpleQAHandler(client=mock_client)

    def test_empty_response_returns_false(self) -> None:
        handler = self._make_handler()
        is_correct, meta = handler.evaluate(
            problem="Q",
            reference="A",
            model_answer="",
            metadata={},
        )
        assert is_correct is False
        assert meta["grade"] == "NOT_ATTEMPTED"

    def test_no_ground_truth_returns_none(self) -> None:
        handler = self._make_handler()
        is_correct, meta = handler.evaluate(
            problem="Q",
            reference="",
            model_answer="answer",
            metadata={},
        )
        assert is_correct is None

    def test_correct_grading(self) -> None:
        handler = self._make_handler(
            chat_return="extracted_final_answer: Shakespeare\nreasoning: matches\ncorrect: yes"
        )
        is_correct, meta = handler.evaluate(
            problem="Who wrote Hamlet?",
            reference="William Shakespeare",
            model_answer="Shakespeare",
            metadata={},
        )
        assert is_correct is True
        assert meta["grade"] == "CORRECT"

    def test_incorrect_grading(self) -> None:
        handler = self._make_handler(
            chat_return="extracted_final_answer: Dickens\nreasoning: wrong\ncorrect: no"
        )
        is_correct, meta = handler.evaluate(
            problem="Who wrote Hamlet?",
            reference="William Shakespeare",
            model_answer="Charles Dickens",
            metadata={},
        )
        assert is_correct is False
        assert meta["grade"] == "INCORRECT"

    def test_extracts_answer(self) -> None:
        handler = self._make_handler(
            chat_return="extracted_final_answer: Paris\nreasoning: correct\ncorrect: yes"
        )
        _, meta = handler.evaluate(
            problem="Capital of France?",
            reference="Paris",
            model_answer="Paris",
            metadata={},
        )
        assert meta["extracted_answer"] == "Paris"

    def test_requires_chat_method(self) -> None:
        mock_client = MagicMock(spec=[])  # no chat method
        handler = SimpleQAHandler(client=mock_client)

        with pytest.raises(RuntimeError, match="requires a client with a .chat"):
            handler.evaluate(
                problem="Q",
                reference="A",
                model_answer="answer",
                metadata={},
            )

    def test_api_error_returns_none(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.side_effect = RuntimeError("API down")
        handler = SimpleQAHandler(client=mock_client)

        is_correct, meta = handler.evaluate(
            problem="Q",
            reference="A",
            model_answer="answer",
            metadata={},
        )
        assert is_correct is None
        assert "error" in meta
