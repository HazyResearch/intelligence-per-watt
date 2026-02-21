"""Tests for datasets/hle.py — HLEDataset."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ipw.core.types import DatasetRecord


class TestHLEDataset:
    """Test HLEDataset with mocked HuggingFace loading."""

    @patch("ipw.datasets.hle.load_dataset")
    def test_iter_records_yields_dataset_records(
        self, mock_load_dataset: MagicMock
    ) -> None:
        from ipw.datasets.hle import HLEDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {
                "question": "What is the binding energy of deuterium?",
                "answer": "2.224 MeV",
                "category": "Physics",
                "difficulty": "expert",
                "id": "hle_001",
            },
            {
                "question": "Describe the Hodge conjecture",
                "answer": "A fundamental conjecture in algebraic geometry",
                "category": "Mathematics",
                "difficulty": "expert",
                "id": "hle_002",
            },
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = HLEDataset()
        records = list(dataset.iter_records())

        assert len(records) == 2
        assert all(isinstance(r, DatasetRecord) for r in records)
        assert records[0].answer == "2.224 MeV"
        assert records[0].subject == "Physics"

    @patch("ipw.datasets.hle.load_dataset")
    def test_size(self, mock_load_dataset: MagicMock) -> None:
        from ipw.datasets.hle import HLEDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {"question": "Q1", "answer": "A1", "category": "Math"},
            {"question": "Q2", "answer": "A2", "category": "Physics"},
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = HLEDataset()
        assert dataset.size() == 2

    @patch("ipw.datasets.hle.load_dataset")
    def test_text_only_filters_multimodal(
        self, mock_load_dataset: MagicMock
    ) -> None:
        from ipw.datasets.hle import HLEDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {"question": "Q1", "answer": "A1", "category": "Math"},
            {"question": "Q2", "answer": "A2", "category": "Art", "image": "img.png"},
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = HLEDataset(text_only=True)
        assert dataset.size() == 1

    @patch("ipw.datasets.hle.load_dataset")
    def test_text_only_false_includes_multimodal(
        self, mock_load_dataset: MagicMock
    ) -> None:
        from ipw.datasets.hle import HLEDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {"question": "Q1", "answer": "A1", "category": "Math"},
            {"question": "Q2", "answer": "A2", "category": "Art", "image": "img.png"},
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = HLEDataset(text_only=False)
        assert dataset.size() == 2

    @patch("ipw.datasets.hle.load_dataset")
    def test_metadata_fields(self, mock_load_dataset: MagicMock) -> None:
        from ipw.datasets.hle import HLEDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {
                "question": "Q",
                "answer": "A",
                "category": "Biology",
                "difficulty": "hard",
                "id": "hle_42",
            },
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = HLEDataset()
        record = list(dataset.iter_records())[0]
        assert record.dataset_metadata["dataset_name"] == "HLE"
        assert record.dataset_metadata["task_id"] == "hle_42"
        assert record.dataset_metadata["category"] == "Biology"

    @patch("ipw.datasets.hle.load_dataset")
    def test_skips_empty_records(self, mock_load_dataset: MagicMock) -> None:
        from ipw.datasets.hle import HLEDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {"question": "", "answer": "A1"},
            {"question": "Q2", "answer": ""},
            {"question": "Q3", "answer": "A3", "category": "C"},
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = HLEDataset()
        assert dataset.size() == 1
