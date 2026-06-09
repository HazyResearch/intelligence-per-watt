"""Tests for the GDPval Stirrup dataset variant."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


def _row() -> dict:
    return {
        "task_id": "task-1",
        "sector": "Information",
        "occupation": "Editor",
        "prompt": "Create a one-page PDF memo.",
        "reference_file_urls": [],
        "reference_file_hf_uris": [],
        "deliverable_files": ["memo.pdf"],
        "deliverable_file_hf_uris": [],
        "rubric_pretty": "PDF memo rubric",
        "rubric_json": json.dumps([{"criterion": "Provides a PDF", "score": 1}]),
    }


@patch("ipw.datasets.gdpval.load_dataset")
def test_gdpval_stirrup_prompt_and_metadata(mock_load_dataset: MagicMock, tmp_path) -> None:
    from ipw.datasets.gdpval_stirrup import GDPvalStirrupDataset

    mock_dataset = MagicMock()
    mock_dataset.to_list.return_value = [_row()]
    mock_load_dataset.return_value = mock_dataset

    dataset = GDPvalStirrupDataset(cache_dir=str(tmp_path), download_files=False)
    record = list(dataset.iter_records())[0]

    assert "finish tool" in record.problem
    assert "<task>" in record.problem
    assert record.dataset_metadata["workload_type"] == "gdpval-stirrup"
    assert record.dataset_metadata["deliverable_files"] == ["memo.pdf"]
    assert dataset.evaluation_method == "gdpval-deliverable"
    assert dataset.requires_serial_telemetry is True
