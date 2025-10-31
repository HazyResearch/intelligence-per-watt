"""Dataset loading helpers for regression analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

from datasets import Dataset, DatasetDict, load_from_disk


def load_metrics_dataset(metrics_dir: Path) -> Dataset:
    """Load the profiler metrics dataset stored on disk."""

    data = load_from_disk(str(metrics_dir))
    if isinstance(data, DatasetDict):
        if not data:
            raise RuntimeError(f"No splits found in dataset at {metrics_dir!s}")
        dataset = next(iter(data.values()))
    else:
        dataset = data

    if not isinstance(dataset, Dataset):
        raise RuntimeError(f"Unsupported dataset object returned for {metrics_dir!s}: {type(dataset)!r}")
    if len(dataset) == 0:
        raise RuntimeError(f"Empty dataset at {metrics_dir!s}")
    return dataset


def resolve_model_name(dataset: Dataset, requested: str | None, metrics_dir: Path) -> str:
    """Determine which model within the dataset should be analyzed."""

    available = _collect_available_models(dataset)

    if requested:
        if requested in available:
            return requested
        if available:
            models_text = ", ".join(sorted(available))
            raise RuntimeError(
                f"Model '{requested}' not found in dataset at '{metrics_dir}'. Available models: {models_text}."
            )
        raise RuntimeError(
            f"Model '{requested}' not found in dataset at '{metrics_dir}', and the dataset does not contain model metrics."
        )

    unique_models = list(dict.fromkeys(available))
    if len(unique_models) == 1:
        return unique_models[0]
    if not unique_models:
        raise RuntimeError(
            f"Could not infer model name from dataset at '{metrics_dir}'. Specify a model explicitly."
        )
    models_text = ", ".join(sorted(unique_models))
    raise RuntimeError(
        f"Dataset at '{metrics_dir}' contains metrics for multiple models ({models_text}). Specify a model to choose one."
    )


def iter_model_entries(dataset: Dataset, model_name: str) -> Iterator[Mapping[str, object]]:
    """Yield the per-record metrics for the requested model."""

    for example in dataset:
        metrics_map = example.get("model_metrics") if isinstance(example, Mapping) else None
        if not isinstance(metrics_map, Mapping):
            continue
        entry = metrics_map.get(model_name)
        if isinstance(entry, Mapping):
            yield entry


def collect_run_metadata(entries: Iterable[Mapping[str, object]]) -> tuple[dict[str, object], dict[str, object]]:
    """Collect the first non-empty system and GPU metadata structures."""

    system_info: dict[str, object] = {}
    gpu_info: dict[str, object] = {}

    for entry in entries:
        if not system_info:
            maybe_system = entry.get("system_info")
            if isinstance(maybe_system, Mapping):
                system_info = dict(maybe_system)
        if not gpu_info:
            maybe_gpu = entry.get("gpu_info")
            if isinstance(maybe_gpu, Mapping):
                gpu_info = dict(maybe_gpu)
        if system_info and gpu_info:
            break

    return system_info, gpu_info


def _collect_available_models(dataset: Dataset) -> Sequence[str]:
    models: list[str] = []
    for example in dataset:
        metrics_map = example.get("model_metrics") if isinstance(example, Mapping) else None
        if not isinstance(metrics_map, Mapping):
            continue
        for key, value in metrics_map.items():
            if value:
                models.append(key)
        if len(set(models)) > 1:
            break
    return list(dict.fromkeys(models))


__all__ = [
    "collect_run_metadata",
    "iter_model_entries",
    "load_metrics_dataset",
    "resolve_model_name",
]
