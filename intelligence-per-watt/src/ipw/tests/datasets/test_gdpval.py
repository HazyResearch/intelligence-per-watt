"""Tests for datasets/gdpval.py — GDPvalDataset."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from ipw.core.types import DatasetRecord


def _sample_row(task_id: str = "t1", with_rubric: bool = True) -> dict:
    return {
        "task_id": task_id,
        "sector": "Professional Services",
        "occupation": "Accountant",
        "prompt": "Prepare a Q4 tax summary based on the attached statements.",
        "reference_files": [],
        "reference_file_urls": [
            "https://example.com/q4_statements.pdf",
        ],
        "reference_file_hf_uris": [
            "hf://datasets/openai/gdpval/files/q4_statements.pdf",
        ],
        "deliverable_files": [],
        "deliverable_file_urls": [],
        "deliverable_file_hf_uris": [],
        "rubric_pretty": "Tax summary rubric",
        "rubric_json": json.dumps(
            [
                {
                    "rubric_item_id": "r1",
                    "criterion": "Includes total revenue.",
                    "score": 2,
                    "required": True,
                },
                {
                    "rubric_item_id": "r2",
                    "criterion": "Includes deductions.",
                    "score": 1,
                    "required": False,
                },
            ]
        ) if with_rubric else "",
    }


class TestGDPvalDataset:
    @patch("ipw.datasets.gdpval.hf_hub_download")
    @patch("ipw.datasets.gdpval.load_dataset")
    def test_iter_records_yields_dataset_records(
        self,
        mock_load_dataset: MagicMock,
        mock_hf_download: MagicMock,
        tmp_path,
    ) -> None:
        from ipw.datasets.gdpval import GDPvalDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [_sample_row("t1"), _sample_row("t2")]
        mock_load_dataset.return_value = mock_dataset

        # Pretend the HF download returns a stub file we can copy
        stub = tmp_path / "_stub.pdf"
        stub.write_bytes(b"stub")
        mock_hf_download.return_value = str(stub)

        dataset = GDPvalDataset(cache_dir=str(tmp_path / "cache"))
        records = list(dataset.iter_records())

        assert len(records) == 2
        assert all(isinstance(r, DatasetRecord) for r in records)
        assert "Q4 tax summary" in records[0].problem
        assert records[0].subject == "Accountant"
        # Each task's inputs dir should be unique
        d1 = records[0].dataset_metadata["gdpval_inputs_dir"]
        d2 = records[1].dataset_metadata["gdpval_inputs_dir"]
        assert d1 != d2

    @patch("ipw.datasets.gdpval.hf_hub_download")
    @patch("ipw.datasets.gdpval.load_dataset")
    def test_max_samples_limits_records(
        self,
        mock_load_dataset: MagicMock,
        mock_hf_download: MagicMock,
        tmp_path,
    ) -> None:
        from ipw.datasets.gdpval import GDPvalDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [_sample_row(f"t{i}") for i in range(10)]
        mock_load_dataset.return_value = mock_dataset
        stub = tmp_path / "_stub.pdf"
        stub.write_bytes(b"x")
        mock_hf_download.return_value = str(stub)

        dataset = GDPvalDataset(
            cache_dir=str(tmp_path / "cache"),
            max_samples=3,
        )
        assert dataset.size() == 3

    @patch("ipw.datasets.gdpval.hf_hub_download")
    @patch("ipw.datasets.gdpval.load_dataset")
    def test_metadata_carries_rubric_and_occupation(
        self,
        mock_load_dataset: MagicMock,
        mock_hf_download: MagicMock,
        tmp_path,
    ) -> None:
        from ipw.datasets.gdpval import GDPvalDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [_sample_row("t1")]
        mock_load_dataset.return_value = mock_dataset
        stub = tmp_path / "_stub.pdf"
        stub.write_bytes(b"x")
        mock_hf_download.return_value = str(stub)

        dataset = GDPvalDataset(cache_dir=str(tmp_path / "cache"))
        record = list(dataset.iter_records())[0]
        meta = record.dataset_metadata
        assert meta["dataset_name"] == "GDPval"
        assert meta["task_id"] == "t1"
        assert meta["occupation"] == "Accountant"
        assert meta["sector"] == "Professional Services"
        assert meta["rubric_json"]  # not empty

    def test_hf_uri_parser_handles_revision_and_urlencoding(self) -> None:
        from ipw.datasets.gdpval import _hf_uri_to_repo_path

        # @revision + URL-encoded filename (real gdpval URI shape)
        repo, path, rev = _hf_uri_to_repo_path(
            "hf://datasets/openai/gdpval@main/reference_files/abc/Population%20v2.xlsx"
        )
        assert repo == "openai/gdpval"
        assert rev == "main"
        assert path == "reference_files/abc/Population v2.xlsx"

        # No revision, no encoding
        repo, path, rev = _hf_uri_to_repo_path(
            "hf://datasets/openai/gdpval/files/x.pdf"
        )
        assert repo == "openai/gdpval"
        assert rev is None
        assert path == "files/x.pdf"

        # Non-HF URIs return None
        assert _hf_uri_to_repo_path("https://example.com/x.pdf") is None
        assert _hf_uri_to_repo_path("") is None

    @patch("ipw.datasets.gdpval.hf_hub_download")
    @patch("ipw.datasets.gdpval.load_dataset")
    def test_skip_download_when_disabled(
        self,
        mock_load_dataset: MagicMock,
        mock_hf_download: MagicMock,
        tmp_path,
    ) -> None:
        from ipw.datasets.gdpval import GDPvalDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [_sample_row("t1")]
        mock_load_dataset.return_value = mock_dataset

        dataset = GDPvalDataset(
            cache_dir=str(tmp_path / "cache"),
            download_files=False,
        )
        records = list(dataset.iter_records())
        mock_hf_download.assert_not_called()
        # Falls back to URLs in the prompt
        assert "example.com" in records[0].problem
