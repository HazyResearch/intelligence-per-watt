from __future__ import annotations

import json
import logging
import math
import shutil
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from tqdm.auto import tqdm

from ..core.registry import AnalysisRegistry, ClientRegistry, DatasetRegistry
from ..core.types import DatasetRecord
from .base import AnalysisContext, AnalysisProvider, AnalysisResult
from .helpers import load_metrics_dataset, resolve_model_name

LOGGER = logging.getLogger(__name__)

# Rail sets the per-record metrics can be aggregated over. See _resolve_basis.
_SOC_BASIS = "soc"
_GPU_BASIS = "gpu"


@dataclass(slots=True)
class _AccuracyCounters:
    correct: int = 0
    incorrect: int = 0
    unevaluated: int = 0
    failed: int = 0


@dataclass(slots=True)
class _EfficiencyAccumulator:
    energy_joules: list[float] = field(default_factory=list)
    power_watts: list[float] = field(default_factory=list)
    latency_seconds: list[float] = field(default_factory=list)
    zero_energy_imputations: list[tuple[float | None, float | None]] = field(
        default_factory=list
    )
    correct_with_energy: int = 0
    total_with_energy: int = 0
    correct_with_power: int = 0
    total_with_power: int = 0
    derived_power_samples: int = 0
    power_metric_samples: int = 0
    zero_energy_values: int = 0
    zero_power_values: int = 0

    def register(
        self,
        *,
        is_correct: bool,
        energy_joules: Any,
        latency_seconds: Any,
        power_watts: Any,
    ) -> None:
        energy_value, energy_provided = _normalize_positive_number(energy_joules)
        latency_value, _ = _normalize_positive_number(latency_seconds)
        power_value, power_provided = _normalize_positive_number(power_watts)

        if latency_value is not None:
            self.latency_seconds.append(latency_value)

        power_for_imputation = power_value

        if energy_value is not None:
            self.energy_joules.append(energy_value)
            self.total_with_energy += 1
            if is_correct:
                self.correct_with_energy += 1
        elif energy_provided and _is_non_positive(energy_joules):
            self.zero_energy_values += 1
            self.zero_energy_imputations.append(
                (power_for_imputation, latency_value)
            )

        derived_power = None
        if energy_value is not None and latency_value is not None:
            derived_power = energy_value / latency_value
            if not (math.isfinite(derived_power) and derived_power > 0):
                derived_power = None

        power_source = None
        chosen_power = None
        if derived_power is not None:
            chosen_power = derived_power
            power_source = "derived"
        elif power_value is not None:
            chosen_power = power_value
            power_source = "power_metrics"

        if chosen_power is not None:
            self.power_watts.append(chosen_power)
            self.total_with_power += 1
            if is_correct:
                self.correct_with_power += 1
            if power_source == "derived":
                self.derived_power_samples += 1
            elif power_source == "power_metrics":
                self.power_metric_samples += 1
        elif power_provided and _is_non_positive(power_watts):
            self.zero_power_values += 1

    @property
    def energy_accuracy(self) -> float | None:
        if self.total_with_energy == 0:
            return None
        return self.correct_with_energy / self.total_with_energy

    @property
    def power_accuracy(self) -> float | None:
        if self.total_with_power == 0:
            return None
        return self.correct_with_power / self.total_with_power


@AnalysisRegistry.register("accuracy")
class AccuracyAnalysis(AnalysisProvider):
    """
    Analysis that performs evaluation (if needed) and aggregates accuracy statistics.

    If records are missing evaluation data, this analysis will:
    1. Instantiate the original dataset provider.
    2. Call dataset.score() for each unevaluated record.
    3. Update and persist the results.
    """

    MAX_EVALUATION_ATTEMPTS = 3
    analysis_id = "accuracy"

    def run(self, context: AnalysisContext) -> AnalysisResult:
        results_dir = context.results_dir
        options = dict(context.options)
        requested_model = options.get("model")

        # Load the dataset (HuggingFace dataset object)
        dataset = load_metrics_dataset(results_dir)
        active_model = resolve_model_name(dataset, requested_model, results_dir)

        # Check if we need to run evaluation
        if self._needs_evaluation(dataset, active_model):
            dataset = self._run_evaluation(context, dataset, active_model, options)

        # Aggregate results
        counters = _AccuracyCounters()
        efficiency = _EfficiencyAccumulator()
        records: list[Dict[str, Any]] = []
        bases_seen: set[str] = set()

        # Iterate directly over the HF dataset rows
        # We assume structure: row["model_metrics"][active_model]["evaluation"]
        for row in dataset:
            model_metrics = row.get("model_metrics") or {}
            metrics = model_metrics.get(active_model) or {}
            energy_metrics = _to_mapping(metrics.get("energy_metrics"))
            power_metrics = _to_mapping(metrics.get("power_metrics"))
            latency_metrics = _to_mapping(metrics.get("latency_metrics"))
            evaluation = _to_mapping(metrics.get("evaluation"))
            metadata = _parse_metadata(evaluation.get("metadata")) if evaluation else {}
            model_answers = row.get("model_answers") or {}
            model_answer = model_answers.get(active_model)
            basis, energy_joules, power_watts = _extract_basis_values(
                energy_metrics,
                power_metrics,
                _resolve_basis(
                    energy_metrics, power_metrics, _to_mapping(metrics.get("gpu_info"))
                ),
            )
            bases_seen.add(basis)
            latency_seconds = latency_metrics.get("total_query_seconds")

            records.append(
                {
                    "problem": row.get("problem"),
                    "reference_answer": row.get("answer"),
                    "model_answer": model_answer,
                    "evaluation": dict(evaluation) if evaluation else {},
                }
            )

            if not evaluation:
                counters.unevaluated += 1
                continue

            if metadata.get("evaluation_failed"):
                counters.failed += 1
                continue

            is_correct = evaluation.get("is_correct")
            if is_correct is True:
                counters.correct += 1
            elif is_correct is False:
                counters.incorrect += 1
            else:
                counters.unevaluated += 1

            if isinstance(is_correct, bool):
                efficiency.register(
                    is_correct=is_correct,
                    energy_joules=energy_joules,
                    latency_seconds=latency_seconds,
                    power_watts=power_watts,
                )

        total_scored = counters.correct + counters.incorrect
        accuracy = (
            counters.correct / total_scored if total_scored > 0 else None
        )

        power_stats = _summarize_series(efficiency.power_watts)
        avg_power = power_stats.get("avg")

        energy_values: list[float] = list(efficiency.energy_joules)
        imputed_energy_values: list[float] = []
        for power_value, latency_value in efficiency.zero_energy_imputations:
            if (
                power_value is None
                or latency_value is None
                or not math.isfinite(power_value)
                or not math.isfinite(latency_value)
                or power_value <= 0
                or latency_value <= 0
            ):
                continue
            imputed = power_value * latency_value
            if math.isfinite(imputed) and imputed > 0:
                imputed_energy_values.append(imputed)
                energy_values.append(imputed)

        energy_stats = _summarize_series(
            energy_values, include_total=True
        )
        avg_energy = energy_stats.get("avg")

        intelligence_per_joule = (
            (accuracy / avg_energy)
            if accuracy is not None and avg_energy and avg_energy > 0
            else None
        )
        intelligence_per_watt = (
            (accuracy / avg_power)
            if accuracy is not None and avg_power and avg_power > 0
            else None
        )

        summary_payload: Dict[str, Any] = {
            "model": active_model,
            "correct": counters.correct,
            "incorrect": counters.incorrect,
            "unevaluated": counters.unevaluated,
            "failed": counters.failed,
            "total_scored": total_scored,
            "accuracy": accuracy,
            "intelligence_per_joule": intelligence_per_joule,
            "intelligence_per_watt": intelligence_per_watt,
            "avg_per_query_energy_joules": energy_stats.get("avg"),
            "avg_per_query_power_watts": power_stats.get("avg"),
            "energy_sample_count": energy_stats.get("count"),
            "power_sample_count": power_stats.get("count"),
            # "soc" (GPU+CPU+ANE) or "gpu"; "mixed" if records disagree. The
            # energy and power figures above are only comparable across runs
            # that share a basis.
            "energy_basis": (
                next(iter(bases_seen))
                if len(bases_seen) == 1
                else ("mixed" if bases_seen else None)
            ),
        }

        efficiency_payload = {
            "intelligence_per_joule": intelligence_per_joule,
            "intelligence_per_watt": intelligence_per_watt,
            "energy": {
                **energy_stats,
                "accuracy": efficiency.energy_accuracy,
                "zero_values": efficiency.zero_energy_values,
                "imputed_from_power": (
                    statistics.fmean(imputed_energy_values)
                    if imputed_energy_values
                    else None
                ),
                "imputed_count": len(imputed_energy_values),
            },
            "power": {
                **power_stats,
                "accuracy": efficiency.power_accuracy,
                "zero_values": efficiency.zero_power_values,
                "derived_power_samples": efficiency.derived_power_samples,
                "power_metric_samples": efficiency.power_metric_samples,
            },
        }

        data_payload: Dict[str, Any] = {
            "per_model": {
                active_model: summary_payload,
            },
            "records": {
                active_model: records,
            },
            "efficiency": {active_model: efficiency_payload},
        }

        warnings = []
        if counters.unevaluated:
            warnings.append(
                f"{counters.unevaluated} records remain unevaluated for model '{active_model}'."
            )
        if counters.failed:
            warnings.append(
                f"{counters.failed} records failed evaluation for model '{active_model}'."
            )
        if energy_stats.get("count", 0) == 0:
            warnings.append(
                f"No per-query energy measurements found for model '{active_model}'; intelligence_per_joule unavailable."
            )
        elif efficiency.zero_energy_values:
            if imputed_energy_values:
                warnings.append(
                    f"Imputed energy for {len(imputed_energy_values)} zero/negative readings using per-record power * latency for model '{active_model}'."
                )
            else:
                warnings.append(
                    f"Ignored {efficiency.zero_energy_values} non-positive per-query energy values while computing efficiency metrics for model '{active_model}'."
                )
        if power_stats.get("count", 0) == 0:
            warnings.append(
                f"No per-query power measurements found for model '{active_model}'; intelligence_per_watt unavailable."
            )
        elif efficiency.zero_power_values:
            warnings.append(
                f"Ignored {efficiency.zero_power_values} non-positive per-query power values while computing efficiency metrics for model '{active_model}'."
            )

        artifact_payload = {
            "analysis": self.analysis_id,
            "summary": summary_payload,
            "warnings": warnings,
            "data": data_payload,
        }

        artifact_dir = results_dir / "analysis"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / f"{self.analysis_id}.json"
        artifact_path.write_text(json.dumps(artifact_payload, indent=2, default=str))

        return AnalysisResult(
            analysis=self.analysis_id,
            summary=summary_payload,
            data=data_payload,
            warnings=tuple(warnings),
            artifacts={"report": artifact_path},
        )

    def _needs_evaluation(self, dataset, model_name: str) -> bool:
        """Check if there are records missing evaluation data."""
        for row in dataset:
            model_metrics = row.get("model_metrics") or {}
            metrics = model_metrics.get(model_name) or {}
            evaluation = _to_mapping(metrics.get("evaluation"))
            if not evaluation:
                return True
            is_correct = evaluation.get("is_correct")
            if is_correct is None:
                metadata = _parse_metadata(evaluation.get("metadata"))
                if metadata.get("evaluation_failed") and not _can_retry_evaluation(
                    metadata, self.MAX_EVALUATION_ATTEMPTS
                ):
                    continue
                return True
        return False

    def _run_evaluation(
        self,
        context: AnalysisContext,
        dataset,
        model_name: str,
        options: Mapping[str, Any],
    ):
        """Instantiate dataset provider and score records."""
        eval_client_id = (options.get("eval_client") or "").strip() or None
        eval_base_url = options.get("eval_base_url")
        eval_model = options.get("eval_model")
        eval_client = None

        # 1. Resolve Dataset Provider
        summary_path = context.results_dir / "summary.json"
        if not summary_path.exists():
            LOGGER.warning("No summary.json found, cannot determine dataset provider.")
            return dataset

        try:
            summary = json.loads(summary_path.read_text())
            dataset_id = summary.get("dataset") or summary.get("profiler_config", {}).get("dataset_id")
            dataset_params = summary.get("profiler_config", {}).get("dataset_params", {})

            if not dataset_id:
                LOGGER.warning("Dataset ID not found in summary.")
                return dataset

            provider_cls = DatasetRegistry.get(dataset_id)
            provider = provider_cls(**dataset_params)

            if not hasattr(provider, "score") or not callable(provider.score):
                LOGGER.warning(f"Dataset provider '{dataset_id}' does not support scoring.")
                return dataset

            if eval_client_id:
                try:
                    eval_client = ClientRegistry.create(
                        eval_client_id, eval_base_url, model=eval_model
                    )
                except Exception as e:
                    LOGGER.error(
                        "Failed to instantiate evaluation client '%s': %s",
                        eval_client_id,
                        e,
                    )
                    return dataset

        except Exception as e:
            LOGGER.error(f"Failed to instantiate dataset provider: {e}")
            return dataset

        # 2. Identify tasks
        # We need to map HF dataset rows back to DatasetRecord for scoring
        tasks = []
        for i, row in enumerate(dataset):
            model_metrics = row.get("model_metrics") or {}
            metrics = model_metrics.get(model_name) or {}
            evaluation = _to_mapping(metrics.get("evaluation"))
            is_correct = evaluation.get("is_correct")
            metadata = _parse_metadata(evaluation.get("metadata")) if evaluation else {}

            if not evaluation or is_correct is None:
                if metadata.get("evaluation_failed") and not _can_retry_evaluation(
                    metadata, self.MAX_EVALUATION_ATTEMPTS
                ):
                    continue
                response = row.get("model_answers", {}).get(model_name, "")
                # Reconstruct DatasetRecord
                raw_dataset_metadata = row.get("dataset_metadata")
                if isinstance(raw_dataset_metadata, Mapping):
                    dataset_metadata = dict(raw_dataset_metadata)
                elif raw_dataset_metadata is None:
                    dataset_metadata = {}
                else:
                    # HuggingFace may persist this field as a JSON string
                    dataset_metadata = {"dataset_metadata": raw_dataset_metadata}

                record = DatasetRecord(
                    problem=row.get("problem", ""),
                    answer=row.get("answer", ""),
                    subject=row.get("subject", ""),
                    dataset_metadata=dataset_metadata,
                )
                tasks.append((i, record, response))

        if not tasks:
            return dataset


        # 3. Execute scoring
        results = {}
        max_workers = 10  # Conservative limit

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._safe_score, provider, record, response, eval_client
                ): idx
                for idx, record, response in tasks
            }

            with tqdm(total=len(tasks), desc="Scoring", unit="record") as pbar:
                for future in as_completed(futures):
                    idx = futures[future]
                    is_correct, meta = future.result()
                    results[idx] = (is_correct, meta)
                    pbar.update(1)

        # 4. Update dataset
        # We can't modify the HF dataset in place easily if it's memory mapped.
        # We use map() to create a new one.

        def update_row(row, idx):
            if idx in results:
                is_correct, meta = results[idx]

                # Ensure structure exists
                if "model_metrics" not in row:
                    row["model_metrics"] = {}
                if model_name not in row["model_metrics"]:
                    row["model_metrics"][model_name] = {}

                existing_eval = _to_mapping(
                    row["model_metrics"][model_name].get("evaluation")
                )
                existing_meta = _parse_metadata(existing_eval.get("metadata"))
                meta_payload = dict(_parse_metadata(meta))
                attempts = (
                    max(
                        _evaluation_attempts(existing_meta),
                        _evaluation_attempts(meta_payload),
                    )
                    + 1
                )
                meta_payload["evaluation_attempts"] = attempts

                # Update evaluation field
                # We store it as a dict, consistent with schema
                row["model_metrics"][model_name]["evaluation"] = {
                    "is_correct": is_correct,
                    "metadata": json.dumps(meta_payload, default=str),
                    # Config is legacy/optional now
                    "config": {}
                }
                # Maintain the legacy lm_correctness flag alongside evaluation data
                row["model_metrics"][model_name]["lm_correctness"] = (
                    is_correct if isinstance(is_correct, bool) else None
                )
            return row

        updated_dataset = dataset.map(update_row, with_indices=True)

        # 5. Persist updated dataset
        temp_path = context.results_dir.with_name(
            context.results_dir.name + "_temp_evaluated_dataset"
        )
        backup_path = context.results_dir.with_suffix(".bak")

        if temp_path.exists():
            shutil.rmtree(temp_path)

        try:
            updated_dataset.save_to_disk(str(temp_path))
            dataset_entries = {item.name for item in temp_path.iterdir()}
        except Exception as exc:
            if temp_path.exists():
                shutil.rmtree(temp_path)
            raise RuntimeError("Failed to write evaluated dataset to disk") from exc

        finalized = False
        try:
            if backup_path.exists():
                if not context.results_dir.exists():
                    LOGGER.warning(
                        "Found existing backup with no active results directory; restoring it before update."
                    )
                    backup_path.rename(context.results_dir)
                else:
                    shutil.rmtree(backup_path)

            if context.results_dir.exists():
                context.results_dir.rename(backup_path)

            temp_path.rename(context.results_dir)
            _restore_non_dataset_artifacts(
                backup_path, context.results_dir, dataset_entries
            )
            finalized = True
        except Exception as exc:
            LOGGER.error(
                "Failed to finalize evaluated dataset, attempting rollback: %s", exc
            )
            try:
                if context.results_dir.exists():
                    shutil.rmtree(context.results_dir)
            except Exception as cleanup_exc:
                LOGGER.warning(
                    "Failed to clean partial results directory during rollback: %s",
                    cleanup_exc,
                )
            try:
                if backup_path.exists():
                    backup_path.rename(context.results_dir)
            except Exception as restore_exc:
                LOGGER.error(
                    "Failed to restore original results directory from backup: %s",
                    restore_exc,
                )
            raise
        finally:
            if temp_path.exists():
                try:
                    shutil.rmtree(temp_path)
                except Exception as cleanup_exc:
                    LOGGER.warning("Failed to remove temporary dataset path: %s", cleanup_exc)
            if finalized and backup_path.exists():
                try:
                    shutil.rmtree(backup_path)
                except Exception as cleanup_exc:
                    LOGGER.warning("Failed to remove backup dataset path: %s", cleanup_exc)

        return updated_dataset

    def _safe_score(self, provider, record, response, eval_client):
        try:
            return provider.score(record, response, eval_client=eval_client)
        except Exception as e:
            LOGGER.warning(f"Scoring failed: {e}")
            return None, {"error": str(e), "evaluation_failed": True}


def _to_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _parse_metadata(value: Any) -> Mapping[str, Any]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        return _to_mapping(parsed)
    return _to_mapping(value)


def _evaluation_attempts(metadata: Any) -> int:
    if not isinstance(metadata, Mapping):
        return 0
    try:
        return int(metadata.get("evaluation_attempts", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _can_retry_evaluation(metadata: Mapping[str, Any], max_attempts: int) -> bool:
    return _evaluation_attempts(metadata) < max_attempts


_EPSILON = 1e-12


def _summarize_series(
    values: Sequence[float], *, include_total: bool = False
) -> Dict[str, Any]:
    if not values:
        summary: Dict[str, Any] = {
            "count": 0,
            "avg": None,
            "median": None,
            "min": None,
            "max": None,
        }
        if include_total:
            summary["total"] = None
        return summary

    summary = {
        "count": len(values),
        "avg": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }
    if include_total:
        summary["total"] = sum(values)
    return summary


def _normalize_positive_number(value: Any) -> tuple[float | None, bool]:
    if value is None:
        return None, False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, True
    if not math.isfinite(number):
        return None, True
    if number <= _EPSILON:
        return None, True
    return number, True


def _is_non_positive(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number <= _EPSILON


def _resolve_basis(
    energy_metrics: Mapping[str, Any],
    power_metrics: Mapping[str, Any],
    gpu_info: Any,
) -> str:
    """Return the rail set to aggregate over: ``"soc"`` or ``"gpu"``.

    The runner makes this choice per record and records it. Records profiled
    before that field existed fall back to the vendor, because getting it wrong
    on Apple Silicon is not a rounding error: powermetrics reports GPU, CPU and
    ANE as separate rails, so the GPU rail alone omits whatever the model
    actually ran on -- everything, for an ANE-resident model.
    """
    for source in (energy_metrics, power_metrics):
        recorded = source.get("basis")
        if isinstance(recorded, str) and recorded:
            return recorded
    if isinstance(gpu_info, Mapping):
        if str(gpu_info.get("vendor") or "").strip().lower() == "apple":
            return _SOC_BASIS
    return _GPU_BASIS


def _extract_basis_values(
    energy_metrics: Mapping[str, Any],
    power_metrics: Mapping[str, Any],
    basis: str = _GPU_BASIS,
) -> tuple[str, Any, float | None]:
    """Per-query energy and power from one rail set, and the basis actually read.

    Energy and power fall back to the GPU rail **together**, never one at a
    time: SoC joules divided by GPU watts is a ratio across two different rail
    sets and means nothing. The returned basis is what was actually read, so a
    record that asked for ``soc`` but only carries the GPU rail is reported as
    ``gpu`` rather than mislabelled in the summary.

    Energy is the raw stored value rather than a normalized float, so the caller
    keeps seeing an explicit zero (which triggers power x latency imputation) as
    distinct from a missing measurement.
    """
    if basis == _SOC_BASIS:
        energy = energy_metrics.get("soc_per_query_joules")
        power = _component_power(power_metrics, "soc")
        if energy is not None and power is not None:
            return _SOC_BASIS, energy, power
    return (
        _GPU_BASIS,
        energy_metrics.get("per_query_joules"),
        _component_power(power_metrics, "gpu"),
    )


def _component_power(
    power_metrics: Mapping[str, Any], component: str
) -> float | None:
    """First positive, finite per-query stat from one power component."""
    component_metrics = power_metrics.get(component)
    if not isinstance(component_metrics, Mapping):
        return None
    per_query = component_metrics.get("per_query_watts")
    if not isinstance(per_query, Mapping):
        return None
    for key in ("avg", "median", "max", "min"):
        raw_value = per_query.get(key)
        try:
            candidate = float(raw_value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(candidate) and candidate > 0:
            return candidate
    return None


def _is_dataset_artifact(path: Path, dataset_entries: set[str]) -> bool:
    name = path.name
    if name in dataset_entries:
        return True
    if name in {"dataset_info.json", "state.json", "dataset_dict.json"}:
        return True
    if name.startswith("data-") or name.endswith(".arrow"):
        return True
    if path.is_dir() and (path / "dataset_info.json").exists():
        return True
    return False


def _restore_non_dataset_artifacts(
    backup_path: Path, target_path: Path, dataset_entries: set[str]
) -> None:
    """Copy non-dataset artifacts (e.g., summary/analysis outputs) from backup."""
    if not backup_path.exists():
        return

    entries = set(dataset_entries)
    for item in backup_path.iterdir():
        if _is_dataset_artifact(item, entries):
            continue
        target = target_path / item.name
        if target.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


__all__ = ["AccuracyAnalysis"]
