"""Tests for evaluation/frames.py — FRAMESHandler LLM judge."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ipw.evaluation.frames import FRAMESHandler


class TestFRAMESHandler:
    """Test FRAMESHandler with mocked LLM client."""

    def _make_handler(self, chat_return: str = "") -> FRAMESHandler:
        mock_client = MagicMock()
        mock_client.chat.return_value = chat_return
        return FRAMESHandler(client=mock_client)

    def test_empty_response_returns_false(self) -> None:
        handler = self._make_handler()
        is_correct, meta = handler.evaluate(
            problem="Q",
            reference="A",
            model_answer="",
            metadata={},
        )
        assert is_correct is False
        assert meta["reason"] == "empty_response"

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
            chat_return="extracted_final_answer: George Lucas\nreasoning: correct\ncorrect: yes"
        )
        is_correct, meta = handler.evaluate(
            problem="Who directed Star Wars?",
            reference="George Lucas",
            model_answer="George Lucas directed it",
            metadata={},
        )
        assert is_correct is True

    def test_incorrect_grading(self) -> None:
        handler = self._make_handler(
            chat_return="extracted_final_answer: Spielberg\nreasoning: wrong\ncorrect: no"
        )
        is_correct, meta = handler.evaluate(
            problem="Who directed Star Wars?",
            reference="George Lucas",
            model_answer="Steven Spielberg",
            metadata={},
        )
        assert is_correct is False

    def test_extracts_answer(self) -> None:
        handler = self._make_handler(
            chat_return="extracted_final_answer: 42\nreasoning: correct\ncorrect: yes"
        )
        _, meta = handler.evaluate(
            problem="Q", reference="42", model_answer="42", metadata={}
        )
        assert meta.get("extracted_answer") == "42"

    def test_requires_chat_method(self) -> None:
        mock_client = MagicMock(spec=[])
        handler = FRAMESHandler(client=mock_client)

        with pytest.raises(RuntimeError, match="requires a client"):
            handler.evaluate(
                problem="Q", reference="A", model_answer="ans", metadata={}
            )

    def test_api_error_returns_none(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.side_effect = RuntimeError("API error")
        handler = FRAMESHandler(client=mock_client)

        is_correct, meta = handler.evaluate(
            problem="Q", reference="A", model_answer="ans", metadata={}
        )
        assert is_correct is None
        assert "error" in meta
