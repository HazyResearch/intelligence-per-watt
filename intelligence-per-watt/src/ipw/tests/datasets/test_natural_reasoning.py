"""Tests for datasets/natural_reasoning.py — NaturalReasoningDataset."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ipw.core.types import DatasetRecord


class TestNaturalReasoningDataset:
    """Test NaturalReasoningDataset with mocked HuggingFace loading."""

    @patch("ipw.datasets.natural_reasoning.load_dataset")
    def test_iter_records_yields_dataset_records(
        self, mock_load_dataset: MagicMock
    ) -> None:
        from ipw.datasets.natural_reasoning import NaturalReasoningDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {
                "question": "What is the square root of 144?",
                "answer": "12",
                "category": "Mathematics",
            },
            {
                "question": "What causes rain?",
                "answer": "Condensation of water vapor in the atmosphere",
                "category": "Science",
            },
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = NaturalReasoningDataset()
        records = list(dataset.iter_records())

        assert len(records) == 2
        assert all(isinstance(r, DatasetRecord) for r in records)
        assert "square root of 144" in records[0].problem
        assert records[0].answer == "12"

    @patch("ipw.datasets.natural_reasoning.load_dataset")
    def test_size(self, mock_load_dataset: MagicMock) -> None:
        from ipw.datasets.natural_reasoning import NaturalReasoningDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {"question": f"Q{i}", "answer": f"A{i}"} for i in range(7)
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = NaturalReasoningDataset()
        assert dataset.size() == 7

    @patch("ipw.datasets.natural_reasoning.load_dataset")
    def test_skips_empty_records(self, mock_load_dataset: MagicMock) -> None:
        from ipw.datasets.natural_reasoning import NaturalReasoningDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {"question": "", "answer": "A1"},
            {"question": "Q2", "answer": ""},
            {"question": "Q3", "answer": "A3"},
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = NaturalReasoningDataset()
        assert dataset.size() == 1

    @patch("ipw.datasets.natural_reasoning.load_dataset")
    def test_metadata_populated(self, mock_load_dataset: MagicMock) -> None:
        from ipw.datasets.natural_reasoning import NaturalReasoningDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {
                "question": "Q",
                "answer": "A",
                "category": "Physics",
                "difficulty": "hard",
            },
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = NaturalReasoningDataset()
        record = list(dataset.iter_records())[0]
        assert record.dataset_metadata["dataset_name"] == "Natural Reasoning"
        assert record.dataset_metadata["category"] == "Physics"
        assert record.dataset_metadata["difficulty"] == "hard"
        assert record.subject == "Physics"

    @patch("ipw.datasets.natural_reasoning.load_dataset")
    def test_max_samples(self, mock_load_dataset: MagicMock) -> None:
        from ipw.datasets.natural_reasoning import NaturalReasoningDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {"question": f"Q{i}", "answer": f"A{i}"} for i in range(10)
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = NaturalReasoningDataset(max_samples=4)
        assert dataset.size() == 4

    @patch("ipw.datasets.natural_reasoning.load_dataset")
    def test_alternative_field_names(self, mock_load_dataset: MagicMock) -> None:
        from ipw.datasets.natural_reasoning import NaturalReasoningDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {"problem": "Solve x + 2 = 5", "solution": "x = 3", "source": "Algebra"},
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = NaturalReasoningDataset()
        records = list(dataset.iter_records())
        assert len(records) == 1
        assert records[0].answer == "x = 3"
        assert records[0].subject == "Algebra"
