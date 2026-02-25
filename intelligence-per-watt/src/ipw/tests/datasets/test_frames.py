"""Tests for datasets/frames.py — FRAMESDataset."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ipw.core.types import DatasetRecord


class TestFRAMESDataset:
    """Test FRAMESDataset with mocked HuggingFace loading."""

    @patch("ipw.datasets.frames.load_dataset")
    def test_iter_records_yields_dataset_records(
        self, mock_load_dataset: MagicMock
    ) -> None:
        from ipw.datasets.frames import FRAMESDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {
                "Prompt": "Who directed the first Star Wars film?",
                "Answer": "George Lucas",
                "reasoning_types": "single_hop",
                "wiki_links": ["https://en.wikipedia.org/wiki/Star_Wars"],
            },
            {
                "Prompt": "Multi-hop question",
                "Answer": "Answer here",
                "reasoning_types": ["multi_hop", "temporal"],
                "wiki_links": [],
            },
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = FRAMESDataset()
        records = list(dataset.iter_records())

        assert len(records) == 2
        assert all(isinstance(r, DatasetRecord) for r in records)
        assert "Star Wars" in records[0].problem
        assert records[0].answer == "George Lucas"

    @patch("ipw.datasets.frames.load_dataset")
    def test_size(self, mock_load_dataset: MagicMock) -> None:
        from ipw.datasets.frames import FRAMESDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {"Prompt": "Q1", "Answer": "A1"},
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = FRAMESDataset()
        assert dataset.size() == 1

    @patch("ipw.datasets.frames.load_dataset")
    def test_metadata_contains_reasoning_types(
        self, mock_load_dataset: MagicMock
    ) -> None:
        from ipw.datasets.frames import FRAMESDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {
                "Prompt": "Q",
                "Answer": "A",
                "reasoning_types": "multi_hop",
                "wiki_links": ["link1", "link2"],
            },
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = FRAMESDataset()
        record = list(dataset.iter_records())[0]
        assert record.dataset_metadata["dataset_name"] == "FRAMES"
        assert record.dataset_metadata["reasoning_types"] == "multi_hop"
        assert len(record.dataset_metadata["wiki_links"]) == 2

    @patch("ipw.datasets.frames.load_dataset")
    def test_skips_empty_questions(self, mock_load_dataset: MagicMock) -> None:
        from ipw.datasets.frames import FRAMESDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {"Prompt": "", "Answer": "A1"},
            {"Prompt": "Q2", "Answer": ""},
            {"Prompt": "Q3", "Answer": "A3"},
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = FRAMESDataset()
        assert dataset.size() == 1
