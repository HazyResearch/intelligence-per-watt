"""Tests for evaluation/gdpval.py — GdpvalHandler (rubric LLM-as-judge)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock


def _rubric(items):
    return json.dumps(items)


class TestGdpvalHandler:
    def test_returns_none_when_no_rubric(self) -> None:
        from ipw.evaluation.gdpval import GdpvalHandler

        handler = GdpvalHandler(client=MagicMock())
        is_correct, meta = handler.evaluate(
            problem="task", reference="", model_answer="answer", metadata={}
        )
        assert is_correct is None
        assert meta["reason"] == "no_rubric"

    def test_empty_response_is_false(self) -> None:
        from ipw.evaluation.gdpval import GdpvalHandler

        handler = GdpvalHandler(client=MagicMock())
        is_correct, meta = handler.evaluate(
            problem="task",
            reference="",
            model_answer="",
            metadata={"rubric_json": _rubric([{"criterion": "x", "score": 1}])},
        )
        assert is_correct is False
        assert meta["reason"] == "empty_response"

    def test_all_pass_required_criteria(self) -> None:
        from ipw.evaluation.gdpval import GdpvalHandler

        client = MagicMock()
        client.chat.return_value = "verdict: yes\nreason: looks good"
        handler = GdpvalHandler(client=client)

        rubric = [
            {"rubric_item_id": "a", "criterion": "Has totals", "score": 2, "required": True},
            {"rubric_item_id": "b", "criterion": "Has detail", "score": 1, "required": False},
        ]
        is_correct, meta = handler.evaluate(
            problem="task",
            reference="",
            model_answer="my answer with totals and detail",
            metadata={"rubric_json": _rubric(rubric)},
        )
        assert is_correct is True
        assert meta["match_type"] == "rubric_llm_judge"
        assert meta["achieved_points"] == 3.0
        assert meta["max_points"] == 3.0
        assert meta["required_failures"] == 0

    def test_required_failure_fails_task(self) -> None:
        from ipw.evaluation.gdpval import GdpvalHandler

        client = MagicMock()
        # First criterion is required and judge says no
        client.chat.side_effect = [
            "verdict: no\nreason: missing totals",
            "verdict: yes\nreason: has detail",
        ]
        handler = GdpvalHandler(client=client)

        rubric = [
            {"rubric_item_id": "a", "criterion": "Has totals", "score": 2, "required": True},
            {"rubric_item_id": "b", "criterion": "Has detail", "score": 1, "required": False},
        ]
        is_correct, meta = handler.evaluate(
            problem="task",
            reference="",
            model_answer="answer",
            metadata={"rubric_json": _rubric(rubric)},
        )
        assert is_correct is False
        assert meta["required_failures"] == 1
        assert meta["achieved_points"] == 1.0

    def test_score_below_threshold_fails(self) -> None:
        from ipw.evaluation.gdpval import GdpvalHandler

        client = MagicMock()
        # Two of five criteria pass; no required → score 2/5 = 0.4 < 0.5
        client.chat.side_effect = [
            "verdict: yes",
            "verdict: yes",
            "verdict: no",
            "verdict: no",
            "verdict: no",
        ]
        handler = GdpvalHandler(client=client)
        rubric = [
            {"criterion": f"c{i}", "score": 1, "required": False} for i in range(5)
        ]
        is_correct, meta = handler.evaluate(
            problem="task",
            reference="",
            model_answer="answer",
            metadata={"rubric_json": _rubric(rubric)},
        )
        assert is_correct is False
        assert meta["score"] == 0.4

    def test_handles_malformed_judge_output(self) -> None:
        from ipw.evaluation.gdpval import GdpvalHandler

        client = MagicMock()
        client.chat.return_value = "garbled response with no verdict line"
        handler = GdpvalHandler(client=client)

        rubric = [{"criterion": "c", "score": 1, "required": False}]
        is_correct, meta = handler.evaluate(
            problem="task",
            reference="",
            model_answer="answer",
            metadata={"rubric_json": _rubric(rubric)},
        )
        # Malformed → treat as not satisfied; score 0 → False
        assert is_correct is False
        assert meta["achieved_points"] == 0.0
