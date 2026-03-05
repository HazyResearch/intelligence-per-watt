from __future__ import annotations

import json
from typing import NoReturn
from unittest.mock import MagicMock, patch

import pytest

from ipw.core.types import DatasetRecord
from ipw.datasets.ipw import IPWDataset


def _skip_missing_dataset(exc: FileNotFoundError) -> NoReturn:
    pytest.skip(str(exc))
    raise AssertionError("pytest.skip is expected to abort the test")


@pytest.fixture(scope="module")
def dataset() -> IPWDataset:
    try:
        return IPWDataset()
    except FileNotFoundError as exc:
        _skip_missing_dataset(exc)
    # The skip helper never returns; this line satisfies the type checker.
    raise AssertionError("unreachable")


def test_dataset_iterates_records(dataset: IPWDataset) -> None:
    first = next(iter(dataset.iter_records()))
    assert first.problem
    assert first.answer


def test_dataset_size_nonzero(dataset: IPWDataset) -> None:
    assert dataset.size() > 0


class TestIPWDatasetScoring:
    @pytest.fixture(autouse=True)
    def mock_registries(self):
        # We need to mock ClientRegistry to avoid real OpenAIClient instantiation
        with patch("ipw.core.registry.ClientRegistry.create") as mock_client_create:
            mock_client = MagicMock()
            mock_client_create.return_value = mock_client

            # We also need to mock EvaluationRegistry to avoid real handler lookup/instantiation
            with patch("ipw.core.registry.EvaluationRegistry.create") as mock_eval_create:
                yield mock_eval_create, mock_client

    def test_score_uses_wildchat_for_allenai_wildchat(self, mock_registries) -> None:
        mock_eval_create, _ = mock_registries
        mock_handler = MagicMock()
        mock_eval_create.return_value = mock_handler
        mock_handler.evaluate.return_value = (True, {"reason": "mocked"})

        ipw_dataset = IPWDataset()

        # Create a mock record matching 'allenai/WildChat'
        record = DatasetRecord(
            problem="test problem",
            answer="test answer",
            subject="test subject",
            dataset_metadata={
                "dataset_metadata": json.dumps({
                    "config": {
                        "dataset_name": "allenai/WildChat",
                        "verification_method": "some_fallback_method" # Should be ignored
                    }
                })
            }
        )
        response = "model response"

        is_correct, meta = ipw_dataset.score(record, response)

        # Check arguments passed to EvaluationRegistry.create
        # args[0] should be 'wildchat'
        args, kwargs = mock_eval_create.call_args
        assert args[0] == "wildchat"

        mock_handler.evaluate.assert_called_once_with(
            problem=record.problem,
            reference=record.answer,
            model_answer=response,
            metadata=json.loads(record.dataset_metadata["dataset_metadata"])
        )
        assert is_correct is True
        assert meta == {"reason": "mocked"}

    def test_score_uses_natural_reasoning_for_facebook_natural_reasoning(self, mock_registries) -> None:
        mock_eval_create, _ = mock_registries
        mock_handler = MagicMock()
        mock_eval_create.return_value = mock_handler
        mock_handler.evaluate.return_value = (False, {"reason": "mocked_false"})

        ipw_dataset = IPWDataset()

        # Create a mock record matching 'facebook/natural_reasoning'
        record = DatasetRecord(
            problem="reasoning problem",
            answer="reasoning answer",
            subject="natural_reasoning",
            dataset_metadata={
                "dataset_metadata": json.dumps({
                    "config": {
                        "dataset_name": "facebook/natural_reasoning",
                        "verification_method": "some_other_fallback_method" # Should be ignored
                    }
                })
            }
        )
        response = "model reasoning response"

        is_correct, meta = ipw_dataset.score(record, response)

        args, kwargs = mock_eval_create.call_args
        assert args[0] == "natural_reasoning"

        assert is_correct is False
        assert meta == {"reason": "mocked_false"}

    def test_score_raises_error_for_unmapped_dataset(self, mock_registries) -> None:
        ipw_dataset = IPWDataset()

        record = DatasetRecord(
            problem="fallback problem",
            answer="fallback answer",
            subject="general",
            dataset_metadata={
                "dataset_metadata": json.dumps({
                    "config": {
                        "dataset_name": "unmapped/dataset",
                        "verification_method": "math-500"
                    }
                })
            }
        )
        response = "model fallback response"

        with pytest.raises(RuntimeError, match="Could not determine evaluation method for dataset"):
            ipw_dataset.score(record, response)

    def test_score_raises_error_if_no_method_found(self, mock_registries) -> None:
        # The logic raises BEFORE calling EvaluationRegistry.create, so mocking it doesn't matter for the exception
        # but we need the fixture to avoid unrelated errors if any

        ipw_dataset = IPWDataset()

        # Create a mock record with an unmapped dataset_name and no verification_method
        record = DatasetRecord(
            problem="error problem",
            answer="error answer",
            subject="general",
            dataset_metadata={
                "dataset_metadata": json.dumps({
                    "config": {
                        "dataset_name": "completely/unknown"
                        # No verification_method here
                    }
                })
            }
        )
        response = "model error response"

        with pytest.raises(RuntimeError, match="Could not determine evaluation method for dataset"):
            ipw_dataset.score(record, response)
