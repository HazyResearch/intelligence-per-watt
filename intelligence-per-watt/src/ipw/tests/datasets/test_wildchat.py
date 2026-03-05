"""Tests for datasets/wildchat.py — WildChatDataset."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ipw.core.types import DatasetRecord


class TestWildChatDataset:
    """Test WildChatDataset with mocked HuggingFace loading."""

    @patch("ipw.datasets.wildchat.load_dataset")
    def test_iter_records_yields_dataset_records(
        self, mock_load_dataset: MagicMock
    ) -> None:
        from ipw.datasets.wildchat import WildChatDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {
                "conversation": [
                    {"role": "user", "content": "Hello, how are you?"},
                    {"role": "assistant", "content": "I'm doing well, thanks!"},
                ],
                "language": "English",
                "model": "gpt-4",
            },
            {
                "conversation": [
                    {"role": "user", "content": "What is Python?"},
                    {"role": "assistant", "content": "Python is a programming language."},
                ],
                "language": "English",
                "model": "gpt-3.5-turbo",
            },
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = WildChatDataset()
        records = list(dataset.iter_records())

        assert len(records) == 2
        assert all(isinstance(r, DatasetRecord) for r in records)
        assert "Hello, how are you?" in records[0].problem
        assert records[0].answer == "I'm doing well, thanks!"

    @patch("ipw.datasets.wildchat.load_dataset")
    def test_size(self, mock_load_dataset: MagicMock) -> None:
        from ipw.datasets.wildchat import WildChatDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {
                "conversation": [
                    {"role": "user", "content": f"Q{i}"},
                    {"role": "assistant", "content": f"A{i}"},
                ],
            }
            for i in range(5)
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = WildChatDataset()
        assert dataset.size() == 5

    @patch("ipw.datasets.wildchat.load_dataset")
    def test_skips_records_without_assistant_response(
        self, mock_load_dataset: MagicMock
    ) -> None:
        from ipw.datasets.wildchat import WildChatDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {
                "conversation": [
                    {"role": "user", "content": "Hello"},
                ],
            },
            {
                "conversation": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"},
                ],
            },
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = WildChatDataset()
        assert dataset.size() == 1

    @patch("ipw.datasets.wildchat.load_dataset")
    def test_skips_short_conversations(
        self, mock_load_dataset: MagicMock
    ) -> None:
        from ipw.datasets.wildchat import WildChatDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {"conversation": []},
            {"conversation": [{"role": "user", "content": "Hi"}]},
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = WildChatDataset()
        assert dataset.size() == 0

    @patch("ipw.datasets.wildchat.load_dataset")
    def test_metadata_populated(self, mock_load_dataset: MagicMock) -> None:
        from ipw.datasets.wildchat import WildChatDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {
                "conversation": [
                    {"role": "user", "content": "Test"},
                    {"role": "assistant", "content": "Response"},
                ],
                "language": "French",
                "model": "claude-3",
            },
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = WildChatDataset()
        record = list(dataset.iter_records())[0]
        assert record.dataset_metadata["dataset_name"] == "WildChat"
        assert record.dataset_metadata["language"] == "French"
        assert record.dataset_metadata["source_model"] == "claude-3"
        assert record.subject == "French"

    @patch("ipw.datasets.wildchat.load_dataset")
    def test_max_samples(self, mock_load_dataset: MagicMock) -> None:
        from ipw.datasets.wildchat import WildChatDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {
                "conversation": [
                    {"role": "user", "content": f"Q{i}"},
                    {"role": "assistant", "content": f"A{i}"},
                ],
            }
            for i in range(10)
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = WildChatDataset(max_samples=3)
        assert dataset.size() == 3
