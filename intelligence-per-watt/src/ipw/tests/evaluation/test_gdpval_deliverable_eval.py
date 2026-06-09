"""Tests for deliverable-aware GDPval judging."""

from __future__ import annotations

import json

import pytest


class _JudgeClient:
    def __init__(self, response: str | list[str]) -> None:
        self.responses = response if isinstance(response, list) else [response]
        self.last_user_prompt = ""
        self.calls = 0

    def chat(self, system_prompt: str, user_prompt: str, **kwargs):
        self.last_user_prompt = user_prompt
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


def test_deliverable_judge_reads_output_files(tmp_path) -> None:
    from ipw.evaluation.gdpval_deliverable import GdpvalDeliverableHandler

    output = tmp_path / "summary.txt"
    output.write_text("Revenue total: 123\nDeductions: 45\n", encoding="utf-8")

    client = _JudgeClient(
        json.dumps(
            {
                "verdict": "pass",
                "score": 1.0,
                "missing_or_broken_deliverables": [],
                "criteria": [
                    {
                        "rubric_item_id": "r1",
                        "criterion": "Includes total revenue.",
                        "satisfied": True,
                        "score_awarded": 2,
                        "score_possible": 2,
                        "evidence": "Revenue total is present.",
                    }
                ],
                "notes": "Good.",
            }
        )
    )
    handler = GdpvalDeliverableHandler(client=client)

    is_correct, meta = handler.evaluate(
        problem="Prepare a summary.",
        reference="",
        model_answer="Created summary.txt",
        metadata={
            "gdpval_submitted_files": [str(output)],
            "rubric_json": json.dumps(
                [{"rubric_item_id": "r1", "criterion": "Includes total revenue.", "score": 2}]
            ),
        },
    )

    assert is_correct is True
    assert meta["score"] == 1.0
    assert "Revenue total" in client.last_user_prompt
    assert meta["deliverables"][0]["sha256"]


def test_deliverable_judge_fails_when_no_files() -> None:
    from ipw.evaluation.gdpval_deliverable import GdpvalDeliverableHandler

    handler = GdpvalDeliverableHandler(client=_JudgeClient("{}"))
    is_correct, meta = handler.evaluate(
        problem="Task",
        reference="",
        model_answer="No files",
        metadata={
            "gdpval_submitted_files": [],
            "rubric_json": json.dumps([{"criterion": "Has output", "score": 1}]),
        },
    )

    assert is_correct is False
    assert meta["reason"] == "no_deliverables_found"


def test_deliverable_judge_repairs_non_json_response(tmp_path) -> None:
    from ipw.evaluation.gdpval_deliverable import GdpvalDeliverableHandler

    output = tmp_path / "summary.txt"
    output.write_text("Revenue total: 123\n", encoding="utf-8")

    client = _JudgeClient(
        [
            "This submission passes. It includes the revenue total.",
            json.dumps(
                {
                    "verdict": "pass",
                    "score": 1.0,
                    "missing_or_broken_deliverables": [],
                    "criteria": [
                        {
                            "rubric_item_id": "r1",
                            "criterion": "Includes total revenue.",
                            "satisfied": True,
                            "score_awarded": 1,
                            "score_possible": 1,
                            "evidence": "Revenue total is present.",
                        }
                    ],
                    "notes": "Pass.",
                }
            ),
        ]
    )

    handler = GdpvalDeliverableHandler(client=client)
    is_correct, meta = handler.evaluate(
        problem="Prepare a summary.",
        reference="",
        model_answer="Created summary.txt",
        metadata={
            "gdpval_submitted_files": [str(output)],
            "rubric_json": json.dumps([{"rubric_item_id": "r1", "criterion": "Includes total revenue.", "score": 1}]),
        },
    )

    assert is_correct is True
    assert client.calls == 2
    assert meta["repair_output"]


def test_deliverable_extractor_handles_image_and_audio_metadata(tmp_path) -> None:
    from ipw.evaluation.deliverables import inspect_file

    pillow = pytest.importorskip("PIL.Image")
    pytest.importorskip("mutagen")

    png = tmp_path / "chart.png"
    image = pillow.new("RGB", (24, 16), color="white")
    image.save(png)

    wav = tmp_path / "clip.wav"
    import wave

    with wave.open(str(wav), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(8000)
        f.writeframes(b"\x00\x00" * 800)

    image_evidence = inspect_file(png)
    audio_evidence = inspect_file(wav)

    assert image_evidence.kind == "image"
    assert image_evidence.metadata["width"] == 24
    assert image_evidence.metadata["height"] == 16
    assert audio_evidence.kind == "audio"
    assert audio_evidence.metadata["duration_seconds"] > 0
