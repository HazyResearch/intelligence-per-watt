"""Tests for evaluation/gaia.py — GAIAHandler exact match and LLM fallback."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ipw.evaluation.gaia import GAIAHandler, _exact_match, _normalize_str


class TestExactMatch:
    """Test GAIA exact-match normalization."""

    def test_exact_string_match(self) -> None:
        assert _exact_match("Paris", "Paris") is True

    def test_case_insensitive(self) -> None:
        assert _exact_match("paris", "Paris") is True

    def test_punctuation_removed(self) -> None:
        assert _exact_match("Paris.", "Paris") is True

    def test_number_match(self) -> None:
        assert _exact_match("42", "42") is True

    def test_number_with_formatting(self) -> None:
        assert _exact_match("$1,000", "1000") is True

    def test_list_match(self) -> None:
        assert _exact_match("a, b, c", "a, b, c") is True

    def test_list_different_length_fails(self) -> None:
        assert _exact_match("a, b", "a, b, c") is False

    def test_empty_model_answer(self) -> None:
        assert _exact_match("None", "42") is False

    def test_none_model_answer(self) -> None:
        assert _exact_match(None, "42") is False

    def test_different_strings(self) -> None:
        assert _exact_match("London", "Paris") is False


class TestNormalizeStr:
    """Test string normalization."""

    def test_removes_spaces(self) -> None:
        assert _normalize_str("hello world") == "helloworld"

    def test_lowercases(self) -> None:
        assert _normalize_str("HELLO") == "hello"

    def test_removes_punctuation(self) -> None:
        assert _normalize_str("hello!") == "hello"

    def test_keep_punctuation_option(self) -> None:
        result = _normalize_str("hello, world!", remove_punct=False)
        assert result == "hello,world!"


class TestGAIAHandler:
    """Test GAIAHandler with mocked LLM client."""

    def _make_handler(self, chat_return: str = "") -> GAIAHandler:
        mock_client = MagicMock()
        mock_client.chat.return_value = chat_return
        return GAIAHandler(client=mock_client)

    def test_exact_match_returns_true(self) -> None:
        handler = self._make_handler()
        is_correct, meta = handler.evaluate(
            problem="Q",
            reference="42",
            model_answer="42",
            metadata={},
        )
        assert is_correct is True
        assert meta["match_type"] == "exact"

    def test_empty_response_returns_false(self) -> None:
        handler = self._make_handler()
        is_correct, meta = handler.evaluate(
            problem="Q",
            reference="42",
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
            model_answer="42",
            metadata={},
        )
        assert is_correct is None
        assert meta["reason"] == "no_ground_truth"

    def test_llm_fallback_correct(self) -> None:
        handler = self._make_handler(
            chat_return="extracted_final_answer: Paris\nreasoning: matches\ncorrect: yes"
        )
        is_correct, meta = handler.evaluate(
            problem="Capital of France?",
            reference="Paris",
            model_answer="The capital is Paris, France",
            metadata={},
        )
        assert is_correct is True
        assert meta["match_type"] == "llm_fallback"

    def test_llm_fallback_incorrect(self) -> None:
        handler = self._make_handler(
            chat_return="extracted_final_answer: London\nreasoning: wrong\ncorrect: no"
        )
        is_correct, meta = handler.evaluate(
            problem="Capital of France?",
            reference="Paris",
            model_answer="London",
            metadata={},
        )
        assert is_correct is False

    def test_llm_fallback_error(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.side_effect = RuntimeError("API error")
        handler = GAIAHandler(client=mock_client)

        is_correct, meta = handler.evaluate(
            problem="Q",
            reference="A",
            model_answer="wrong answer",
            metadata={},
        )
        assert is_correct is False
        assert "error" in meta
