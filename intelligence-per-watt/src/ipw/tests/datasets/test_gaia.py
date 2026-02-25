"""Tests for datasets/gaia.py — GAIADataset."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ipw.core.types import DatasetRecord


class TestGAIADataset:
    """Test GAIADataset with mocked HuggingFace dataset loading."""

    @patch("ipw.datasets.gaia.snapshot_download")
    @patch("ipw.datasets.gaia.load_dataset")
    def test_iter_records_yields_dataset_records(
        self, mock_load_dataset: MagicMock, mock_snapshot: MagicMock, tmp_path
    ) -> None:
        from ipw.datasets.gaia import GAIADataset

        mock_snapshot.return_value = str(tmp_path / "GAIA")
        (tmp_path / "GAIA").mkdir(parents=True)

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {
                "task_id": "task_001",
                "Question": "What is 2+2?",
                "Final answer": "4",
                "Level": 1,
            },
            {
                "task_id": "task_002",
                "Question": "Capital of France?",
                "Final answer": "Paris",
                "Level": 2,
            },
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = GAIADataset(cache_dir=str(tmp_path))
        records = list(dataset.iter_records())

        assert len(records) == 2
        assert all(isinstance(r, DatasetRecord) for r in records)
        assert "What is 2+2?" in records[0].problem
        assert records[0].answer == "4"
        assert "level_1" in records[0].subject

    @patch("ipw.datasets.gaia.snapshot_download")
    @patch("ipw.datasets.gaia.load_dataset")
    def test_size_matches_records(
        self, mock_load_dataset: MagicMock, mock_snapshot: MagicMock, tmp_path
    ) -> None:
        from ipw.datasets.gaia import GAIADataset

        mock_snapshot.return_value = str(tmp_path / "GAIA")
        (tmp_path / "GAIA").mkdir(parents=True)

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {"task_id": "t1", "Question": "Q1", "Final answer": "A1", "Level": 1},
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = GAIADataset(cache_dir=str(tmp_path))
        assert dataset.size() == 1
        assert dataset.size() == len(list(dataset.iter_records()))

    @patch("ipw.datasets.gaia.snapshot_download")
    @patch("ipw.datasets.gaia.load_dataset")
    def test_skips_empty_questions(
        self, mock_load_dataset: MagicMock, mock_snapshot: MagicMock, tmp_path
    ) -> None:
        from ipw.datasets.gaia import GAIADataset

        mock_snapshot.return_value = str(tmp_path / "GAIA")
        (tmp_path / "GAIA").mkdir(parents=True)

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {"task_id": "t1", "Question": "", "Final answer": "A1", "Level": 1},
            {"task_id": "t2", "Question": "Q2", "Final answer": "", "Level": 1},
            {"task_id": "t3", "Question": "Q3", "Final answer": "A3", "Level": 1},
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = GAIADataset(cache_dir=str(tmp_path))
        assert dataset.size() == 1

    @patch("ipw.datasets.gaia.snapshot_download")
    @patch("ipw.datasets.gaia.load_dataset")
    def test_metadata_fields(
        self, mock_load_dataset: MagicMock, mock_snapshot: MagicMock, tmp_path
    ) -> None:
        from ipw.datasets.gaia import GAIADataset

        mock_snapshot.return_value = str(tmp_path / "GAIA")
        (tmp_path / "GAIA").mkdir(parents=True)

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {"task_id": "t1", "Question": "Q", "Final answer": "A", "Level": 2},
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = GAIADataset(cache_dir=str(tmp_path))
        record = list(dataset.iter_records())[0]
        assert record.dataset_metadata["dataset_name"] == "GAIA"
        assert record.dataset_metadata["task_id"] == "t1"
        assert record.dataset_metadata["level"] == 2
