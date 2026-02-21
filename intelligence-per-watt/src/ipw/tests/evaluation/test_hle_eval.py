"""Tests for evaluation/hle.py — HLEHandler LLM judge."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ipw.evaluation.hle import HLEHandler, _parse_grade


class TestParseGrade:
    """Test _parse_grade response parsing."""

    def test_correct_yes(self) -> None:
        assert _parse_grade("correct: yes") == "CORRECT"

    def test_correct_no(self) -> None:
        assert _parse_grade("correct: no") == "INCORRECT"

    def test_not_attempted(self) -> None:
        assert _parse_grade("correct: not_attempted") == "NOT_ATTEMPTED"

    def test_fallback_correct(self) -> None:
        assert _parse_grade("The answer is CORRECT") == "CORRECT"

    def test_fallback_incorrect(self) -> None:
        assert _parse_grade("INCORRECT") == "INCORRECT"

    def test_unparseable(self) -> None:
        assert _parse_grade("random text") == "NOT_ATTEMPTED"


class TestHLEHandler:
    """Test HLEHandler with mocked LLM client."""

    def _make_handler(self, chat_return: str = "") -> HLEHandler:
        mock_client = MagicMock()
        mock_client.chat.return_value = chat_return
        return HLEHandler(client=mock_client)

    def test_empty_response(self) -> None:
        handler = self._make_handler()
        is_correct, meta = handler.evaluate(
            problem="Q", reference="A", model_answer="", metadata={}
        )
        assert is_correct is False
        assert meta["grade"] == "NOT_ATTEMPTED"

    def test_no_ground_truth(self) -> None:
        handler = self._make_handler()
        is_correct, meta = handler.evaluate(
            problem="Q", reference="", model_answer="ans", metadata={}
        )
        assert is_correct is None

    def test_correct_grading(self) -> None:
        handler = self._make_handler(
            chat_return="extracted_final_answer: 2.224 MeV\nreasoning: correct\ncorrect: yes"
        )
        is_correct, meta = handler.evaluate(
            problem="Binding energy?",
            reference="2.224 MeV",
            model_answer="2.224 MeV",
            metadata={},
        )
        assert is_correct is True
        assert meta["grade"] == "CORRECT"

    def test_incorrect_grading(self) -> None:
        handler = self._make_handler(
            chat_return="extracted_final_answer: 1.0 MeV\nreasoning: wrong\ncorrect: no"
        )
        is_correct, meta = handler.evaluate(
            problem="Q", reference="A", model_answer="wrong", metadata={}
        )
        assert is_correct is False
        assert meta["grade"] == "INCORRECT"

    def test_requires_chat_method(self) -> None:
        mock_client = MagicMock(spec=[])
        handler = HLEHandler(client=mock_client)

        with pytest.raises(RuntimeError, match="requires a client"):
            handler.evaluate(
                problem="Q", reference="A", model_answer="ans", metadata={}
            )

    def test_api_error_returns_none(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.side_effect = RuntimeError("API error")
        handler = HLEHandler(client=mock_client)

        is_correct, meta = handler.evaluate(
            problem="Q", reference="A", model_answer="ans", metadata={}
        )
        assert is_correct is None
        assert "error" in meta
