"""Tests for datasets/simpleqa.py — SimpleQADataset."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ipw.core.types import DatasetRecord


class TestSimpleQADataset:
    """Test SimpleQADataset with mocked HuggingFace loading."""

    @patch("ipw.datasets.simpleqa.load_dataset")
    def test_iter_records_yields_dataset_records(
        self, mock_load_dataset: MagicMock
    ) -> None:
        from ipw.datasets.simpleqa import SimpleQADataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {
                "problem": "Who wrote Hamlet?",
                "answer": "William Shakespeare",
                "metadata": '{"topic": "Literature"}',
            },
            {
                "problem": "What is the speed of light?",
                "answer": "299792458 m/s",
                "metadata": '{"topic": "Physics"}',
            },
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = SimpleQADataset()
        records = list(dataset.iter_records())

        assert len(records) == 2
        assert all(isinstance(r, DatasetRecord) for r in records)
        assert "Who wrote Hamlet?" in records[0].problem
        assert records[0].answer == "William Shakespeare"

    @patch("ipw.datasets.simpleqa.load_dataset")
    def test_size(self, mock_load_dataset: MagicMock) -> None:
        from ipw.datasets.simpleqa import SimpleQADataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {"problem": "Q1", "answer": "A1", "metadata": "{}"},
            {"problem": "Q2", "answer": "A2", "metadata": "{}"},
            {"problem": "Q3", "answer": "A3", "metadata": "{}"},
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = SimpleQADataset()
        assert dataset.size() == 3

    @patch("ipw.datasets.simpleqa.load_dataset")
    def test_skips_empty_records(self, mock_load_dataset: MagicMock) -> None:
        from ipw.datasets.simpleqa import SimpleQADataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {"problem": "", "answer": "A1", "metadata": "{}"},
            {"problem": "Q2", "answer": "", "metadata": "{}"},
            {"problem": "Q3", "answer": "A3", "metadata": "{}"},
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = SimpleQADataset()
        assert dataset.size() == 1

    @patch("ipw.datasets.simpleqa.load_dataset")
    def test_metadata_topic_parsed(self, mock_load_dataset: MagicMock) -> None:
        from ipw.datasets.simpleqa import SimpleQADataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {"problem": "Q", "answer": "A", "metadata": '{"topic": "Science"}'},
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = SimpleQADataset()
        record = list(dataset.iter_records())[0]
        assert record.subject == "Science"
        assert record.dataset_metadata["dataset_name"] == "SimpleQA"

    @patch("ipw.datasets.simpleqa.load_dataset")
    def test_max_samples(self, mock_load_dataset: MagicMock) -> None:
        from ipw.datasets.simpleqa import SimpleQADataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {"problem": f"Q{i}", "answer": f"A{i}", "metadata": "{}"}
            for i in range(10)
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = SimpleQADataset(max_samples=3)
        assert dataset.size() == 3
