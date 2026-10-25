"""Trace data model for eval benchmark runs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..core.accounting import rigorous_utility_basis, rigorous_utility_is_eligible
from ..cost.aggregation import is_valid_cost, sum_costs_if_complete


def _sum_measurements_if_complete(
    values: Iterable[float | None],
) -> Optional[float]:
    """Sum finite non-negative measurements only when every component exists."""
    items = tuple(values)
    if not items or any(
        value is None
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in items
    ):
        return None
    return math.fsum(value for value in items if value is not None)


@dataclass
class TurnTrace:
    """Per-turn telemetry data."""

    turn_index: int
    input_tokens: Optional[int] = 0
    output_tokens: Optional[int] = 0
    tool_result_tokens: Optional[int] = 0
    tools_called: List[str] = field(default_factory=list)
    tool_latencies_s: Dict[str, float] = field(default_factory=dict)
    wall_clock_s: float = 0.0
    error: Optional[str] = None
    # Energy and cost fields
    gpu_energy_joules: Optional[float] = None
    cpu_energy_joules: Optional[float] = None
    gpu_power_avg_watts: Optional[float] = None
    cpu_power_avg_watts: Optional[float] = None
    # LM/API charge for this turn. Tool charges are recorded separately so a
    # missing paid action cannot hide behind a known model cost.
    cost_usd: Optional[float] = None
    dram_energy_joules: Optional[float] = None
    peak_watts: Optional[float] = None
    reasoning_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    has_parallel_tools: bool = False
    shared_device_warning: bool = False
    # Phase-0 additive fields must remain after the complete legacy positional
    # tail above for the two-release constructor compatibility window.
    tool_cost_usd: Optional[float] = None
    tool_cost_complete: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tool_result_tokens": self.tool_result_tokens,
            "tools_called": list(self.tools_called),
            "tool_latencies_s": dict(self.tool_latencies_s),
            "wall_clock_s": self.wall_clock_s,
            "error": self.error,
            "gpu_energy_joules": self.gpu_energy_joules,
            "cpu_energy_joules": self.cpu_energy_joules,
            "gpu_power_avg_watts": self.gpu_power_avg_watts,
            "cpu_power_avg_watts": self.cpu_power_avg_watts,
            "cost_usd": self.cost_usd,
            "legacy_lm_only_cost_usd": self.legacy_lm_only_cost_usd,
            "tool_cost_usd": self.tool_cost_usd,
            "tool_cost_complete": self.tool_cost_complete,
            "total_cost_usd": self.total_cost_usd,
            "complete_total_cost_usd": self.complete_total_cost_usd,
            "dram_energy_joules": self.dram_energy_joules,
            "peak_watts": self.peak_watts,
            "reasoning_tokens": self.reasoning_tokens,
            "cached_tokens": self.cached_tokens,
            "has_parallel_tools": self.has_parallel_tools,
            "shared_device_warning": self.shared_device_warning,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TurnTrace:
        return cls(
            turn_index=d["turn_index"],
            input_tokens=d.get("input_tokens", 0),
            output_tokens=d.get("output_tokens", 0),
            tool_result_tokens=d.get("tool_result_tokens", 0),
            tools_called=d.get("tools_called", []),
            tool_latencies_s=d.get("tool_latencies_s", {}),
            wall_clock_s=d.get("wall_clock_s", 0.0),
            error=d.get("error"),
            gpu_energy_joules=d.get("gpu_energy_joules"),
            cpu_energy_joules=d.get("cpu_energy_joules"),
            gpu_power_avg_watts=d.get("gpu_power_avg_watts"),
            cpu_power_avg_watts=d.get("cpu_power_avg_watts"),
            cost_usd=d.get("cost_usd"),
            tool_cost_usd=d.get("tool_cost_usd"),
            tool_cost_complete=d.get("tool_cost_complete", False),
            dram_energy_joules=d.get("dram_energy_joules"),
            peak_watts=d.get("peak_watts"),
            reasoning_tokens=d.get("reasoning_tokens"),
            cached_tokens=d.get("cached_tokens"),
            has_parallel_tools=d.get("has_parallel_tools", False),
            shared_device_warning=d.get("shared_device_warning", False),
        )

    @property
    def total_cost_usd(self) -> Optional[float]:
        """Return complete LM + tool cost for this turn."""
        tool_cost: float | None
        if not self.tools_called:
            tool_cost = 0.0
        elif self.tool_cost_complete:
            tool_cost = self.tool_cost_usd
        else:
            tool_cost = None
        return sum_costs_if_complete((self.cost_usd, tool_cost))

    @property
    def complete_total_cost_usd(self) -> Optional[float]:
        """Explicit alias for the complete LM + tool cost."""

        return self.total_cost_usd

    @property
    def legacy_lm_only_cost_usd(self) -> Optional[float]:
        """Return the pre-Phase-0 LM/API-only cost field.

        This compatibility view deliberately excludes tool charges. New IPD
        accounting must use :attr:`complete_total_cost_usd`, which fails closed
        when a called tool's price is unknown.
        """

        return float(self.cost_usd) if is_valid_cost(self.cost_usd) else None


@dataclass
class QueryTrace:
    """Per-query aggregate telemetry."""

    query_id: str
    workload_type: str
    query_text: str = ""
    response_text: str = ""
    turns: List[TurnTrace] = field(default_factory=list)
    total_wall_clock_s: float = 0.0
    completed: bool = False
    timed_out: bool = False
    # Query-level energy fields (populated even when turns are empty)
    query_gpu_energy_joules: Optional[float] = None
    query_cpu_energy_joules: Optional[float] = None
    query_gpu_power_avg_watts: Optional[float] = None
    query_cpu_power_avg_watts: Optional[float] = None
    # Judge/evaluation energy is recorded separately so query energy remains
    # attributable to the agent prompt window.
    judge_gpu_energy_joules: Optional[float] = None
    judge_cpu_energy_joules: Optional[float] = None
    judge_gpu_power_avg_watts: Optional[float] = None
    judge_cpu_power_avg_watts: Optional[float] = None
    judge_wall_clock_s: Optional[float] = None
    query_mbu_avg_pct: Optional[float] = None
    query_mbu_max_pct: Optional[float] = None
    is_resolved: Optional[bool] = None
    unscorable_reason: Optional[str] = None
    score_metadata: Dict[str, Any] = field(default_factory=dict)
    # True by default; runner sets False for cloud LMs where GPU telemetry is unavailable
    lm_energy_measurable: bool = True
    shared_device_warning: bool = False
    # evaluators may return float, bool, or a dict; float is the common case — cast explicitly when needed
    accuracy_score: Optional[float] = None
    accuracy_metadata: Dict[str, Any] = field(default_factory=dict)
    n_retries: int = 0
    # Phase-0 additive fields must remain after the complete legacy positional
    # tail above for the two-release constructor compatibility window.
    # Historical traces default to unknown and remain descriptive only.
    utility_complete: bool = False
    utility_provenance: str = "unknown"
    utility_calibration_id: Optional[str] = None
    gpu_energy_complete: Optional[bool] = None
    gpu_energy_provenance: Optional[str] = None
    accelerator_device_coverage_complete: Optional[bool] = None
    # A utility score is not scientifically reproducible without identifying
    # both the verifier implementation and the exact task population it scored.
    # These remain unset by legacy runners until they have truthful bindings.
    utility_verifier_id: Optional[str] = None
    utility_verifier_version: Optional[str] = None
    utility_task_manifest_id: Optional[str] = None
    utility_task_membership_id: Optional[str] = None
    utility_task_manifest_size: Optional[int] = None

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    @property
    def total_input_tokens(self) -> Optional[int]:
        if any(t.input_tokens is None for t in self.turns):
            return None
        return sum(t.input_tokens or 0 for t in self.turns)

    @property
    def total_output_tokens(self) -> Optional[int]:
        if any(t.output_tokens is None for t in self.turns):
            return None
        return sum(t.output_tokens or 0 for t in self.turns)

    @property
    def tool_call_count(self) -> int:
        return sum(len(t.tools_called) for t in self.turns)

    @property
    def total_tool_calls(self) -> int:
        return self.tool_call_count

    @property
    def total_gpu_energy_joules(self) -> Optional[float]:
        """Return the legacy partial GPU-energy sum.

        This compatibility view skips missing turns. Rigorous accounting must
        use :attr:`complete_total_gpu_energy_joules`.
        """

        if self.query_gpu_energy_joules is not None:
            return self.query_gpu_energy_joules
        values = [turn.gpu_energy_joules for turn in self.turns if turn.gpu_energy_joules is not None]
        return sum(values) if values else None

    @property
    def complete_total_gpu_energy_joules(self) -> Optional[float]:
        """Return GPU energy only when every required measurement is valid."""

        if self.query_gpu_energy_joules is not None:
            return _sum_measurements_if_complete([self.query_gpu_energy_joules])
        return _sum_measurements_if_complete(turn.gpu_energy_joules for turn in self.turns)

    @property
    def total_cpu_energy_joules(self) -> Optional[float]:
        """Return the legacy partial host/CPU-energy sum."""

        if self.query_cpu_energy_joules is not None:
            return self.query_cpu_energy_joules
        values = [turn.cpu_energy_joules for turn in self.turns if turn.cpu_energy_joules is not None]
        return sum(values) if values else None

    @property
    def complete_total_cpu_energy_joules(self) -> Optional[float]:
        """Return host/CPU energy only when every required measurement is valid."""

        if self.query_cpu_energy_joules is not None:
            return _sum_measurements_if_complete([self.query_cpu_energy_joules])
        return _sum_measurements_if_complete(turn.cpu_energy_joules for turn in self.turns)

    @property
    def total_task_gpu_energy_joules(self) -> Optional[float]:
        values = [value for value in (self.total_gpu_energy_joules, self.judge_gpu_energy_joules) if value is not None]
        return sum(values) if values else None

    @property
    def total_task_cpu_energy_joules(self) -> Optional[float]:
        values = [value for value in (self.total_cpu_energy_joules, self.judge_cpu_energy_joules) if value is not None]
        return sum(values) if values else None

    @property
    def complete_total_task_gpu_energy_joules(self) -> Optional[float]:
        """Return complete query plus judge GPU energy."""

        return _sum_measurements_if_complete([self.complete_total_gpu_energy_joules, self.judge_gpu_energy_joules])

    @property
    def complete_total_task_cpu_energy_joules(self) -> Optional[float]:
        """Return complete query plus judge host/CPU energy."""

        return _sum_measurements_if_complete([self.complete_total_cpu_energy_joules, self.judge_cpu_energy_joules])

    @property
    def total_cost_usd(self) -> Optional[float]:
        """Return the legacy partial sum of known LM/API costs only."""

        return self.legacy_lm_only_cost_usd

    @property
    def complete_total_cost_usd(self) -> Optional[float]:
        """Return complete LM + tool cost, failing closed on any unknown."""

        return sum_costs_if_complete((turn.complete_total_cost_usd for turn in self.turns))

    @property
    def legacy_lm_only_cost_usd(self) -> Optional[float]:
        """Reproduce the legacy partial sum of known LM/API turn costs.

        This alias exists only for the two-release compatibility window. It
        excludes every tool charge and preserves the historical behavior of
        skipping turns whose LM cost is unknown. It is therefore descriptive,
        not eligible for rigorous IPD.
        """

        values = [turn.cost_usd for turn in self.turns if turn.cost_usd is not None]
        if not values or any(not is_valid_cost(value) for value in values):
            return None
        return sum(values)

    @property
    def cost_known_turns(self) -> int:
        return sum(is_valid_cost(turn.complete_total_cost_usd) for turn in self.turns)

    @property
    def cost_unknown_turns(self) -> int:
        return self.num_turns - self.cost_known_turns

    @property
    def cost_coverage(self) -> Optional[float]:
        return self.cost_known_turns / self.num_turns if self.num_turns > 0 else None

    @property
    def total_tokens(self) -> Optional[int]:
        """Total tokens (input + output) across all turns."""
        if self.total_input_tokens is None or self.total_output_tokens is None:
            return None
        return self.total_input_tokens + self.total_output_tokens

    @property
    def rigorous_utility_eligible(self) -> bool:
        return rigorous_utility_is_eligible(
            complete=self.utility_complete,
            provenance=self.utility_provenance,
            calibration_id=self.utility_calibration_id,
            verifier_id=self.utility_verifier_id,
            verifier_version=self.utility_verifier_version,
            task_manifest_id=self.utility_task_manifest_id,
            task_membership_id=self.utility_task_membership_id,
            task_manifest_size=self.utility_task_manifest_size,
        )

    @property
    def rigorous_utility_basis(
        self,
    ) -> tuple[str, str, str, str, str | None, int] | None:
        """Canonical evidence basis used to reject mixed workload aggregates."""
        return rigorous_utility_basis(
            complete=self.utility_complete,
            provenance=self.utility_provenance,
            calibration_id=self.utility_calibration_id,
            verifier_id=self.utility_verifier_id,
            verifier_version=self.utility_verifier_version,
            task_manifest_id=self.utility_task_manifest_id,
            task_membership_id=self.utility_task_membership_id,
            task_manifest_size=self.utility_task_manifest_size,
        )

    @property
    def avg_gpu_power_watts(self) -> Optional[float]:
        """Mean GPU power across turns; falls back to query-level power."""
        values = [t.gpu_power_avg_watts for t in self.turns if t.gpu_power_avg_watts is not None]
        if values:
            return sum(values) / len(values)
        return self.query_gpu_power_avg_watts

    @property
    def avg_cpu_power_watts(self) -> Optional[float]:
        """Mean CPU power across turns; falls back to query-level power."""
        values = [t.cpu_power_avg_watts for t in self.turns if t.cpu_power_avg_watts is not None]
        if values:
            return sum(values) / len(values)
        return self.query_cpu_power_avg_watts

    @property
    def throughput_tokens_per_sec(self) -> Optional[float]:
        """Output tokens per second; None if zero tokens or zero time."""
        output_tokens = self.total_output_tokens
        if output_tokens is not None and output_tokens > 0 and self.total_wall_clock_s > 0:
            return output_tokens / self.total_wall_clock_s
        return None

    @property
    def energy_per_token_joules(self) -> Optional[float]:
        """GPU energy per output token; None if no energy data or zero tokens."""
        gpu_energy = self.total_gpu_energy_joules
        output_tokens = self.total_output_tokens
        if gpu_energy is not None and gpu_energy > 0 and output_tokens is not None and output_tokens > 0:
            return gpu_energy / output_tokens
        return None

    @property
    def complete_energy_per_token_joules(self) -> Optional[float]:
        """GPU energy/output-token only with a complete energy denominator."""

        gpu_energy = self.complete_total_gpu_energy_joules
        output_tokens = self.total_output_tokens
        if gpu_energy is not None and gpu_energy > 0 and output_tokens is not None and output_tokens > 0:
            return gpu_energy / output_tokens
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "workload_type": self.workload_type,
            "query_text": self.query_text,
            "response_text": self.response_text,
            "turns": [t.to_dict() for t in self.turns],
            "num_turns": self.num_turns,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "tool_call_count": self.tool_call_count,
            "total_tool_calls": self.total_tool_calls,
            "total_wall_clock_s": self.total_wall_clock_s,
            "completed": self.completed,
            "timed_out": self.timed_out,
            "query_gpu_energy_joules": self.query_gpu_energy_joules,
            "query_cpu_energy_joules": self.query_cpu_energy_joules,
            "query_gpu_power_avg_watts": self.query_gpu_power_avg_watts,
            "query_cpu_power_avg_watts": self.query_cpu_power_avg_watts,
            "judge_gpu_energy_joules": self.judge_gpu_energy_joules,
            "judge_cpu_energy_joules": self.judge_cpu_energy_joules,
            "judge_gpu_power_avg_watts": self.judge_gpu_power_avg_watts,
            "judge_cpu_power_avg_watts": self.judge_cpu_power_avg_watts,
            "judge_wall_clock_s": self.judge_wall_clock_s,
            "total_gpu_energy_joules": self.total_gpu_energy_joules,
            "complete_total_gpu_energy_joules": self.complete_total_gpu_energy_joules,
            "total_cpu_energy_joules": self.total_cpu_energy_joules,
            "complete_total_cpu_energy_joules": self.complete_total_cpu_energy_joules,
            "total_task_gpu_energy_joules": self.total_task_gpu_energy_joules,
            "total_task_cpu_energy_joules": self.total_task_cpu_energy_joules,
            "complete_total_task_gpu_energy_joules": (self.complete_total_task_gpu_energy_joules),
            "complete_total_task_cpu_energy_joules": (self.complete_total_task_cpu_energy_joules),
            "query_mbu_avg_pct": self.query_mbu_avg_pct,
            "query_mbu_max_pct": self.query_mbu_max_pct,
            "throughput_tokens_per_sec": self.throughput_tokens_per_sec,
            "energy_per_output_token_joules": self.energy_per_token_joules,
            "complete_energy_per_output_token_joules": (self.complete_energy_per_token_joules),
            "total_cost_usd": self.total_cost_usd,
            "legacy_lm_only_cost_usd": self.legacy_lm_only_cost_usd,
            "complete_total_cost_usd": self.complete_total_cost_usd,
            "cost_known_turns": self.cost_known_turns,
            "cost_unknown_turns": self.cost_unknown_turns,
            "cost_coverage": self.cost_coverage,
            "is_resolved": self.is_resolved,
            "unscorable_reason": self.unscorable_reason,
            "score_metadata": dict(self.score_metadata),
            "utility_complete": self.utility_complete,
            "utility_provenance": self.utility_provenance,
            "utility_calibration_id": self.utility_calibration_id,
            "utility_verifier_id": self.utility_verifier_id,
            "utility_verifier_version": self.utility_verifier_version,
            "utility_task_manifest_id": self.utility_task_manifest_id,
            "utility_task_membership_id": self.utility_task_membership_id,
            "utility_task_manifest_size": self.utility_task_manifest_size,
            "lm_energy_measurable": self.lm_energy_measurable,
            "shared_device_warning": self.shared_device_warning,
            "gpu_energy_complete": self.gpu_energy_complete,
            "gpu_energy_provenance": self.gpu_energy_provenance,
            "accelerator_device_coverage_complete": (self.accelerator_device_coverage_complete),
            "accuracy_score": self.accuracy_score,
            "accuracy_metadata": dict(self.accuracy_metadata),
            "n_retries": self.n_retries,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> QueryTrace:
        return cls(
            query_id=d["query_id"],
            workload_type=d["workload_type"],
            query_text=d.get("query_text", ""),
            response_text=d.get("response_text", ""),
            turns=[TurnTrace.from_dict(t) for t in d.get("turns", [])],
            total_wall_clock_s=d.get("total_wall_clock_s", 0.0),
            completed=d.get("completed", False),
            timed_out=d.get("timed_out", False),
            query_gpu_energy_joules=d.get("query_gpu_energy_joules"),
            query_cpu_energy_joules=d.get("query_cpu_energy_joules"),
            query_gpu_power_avg_watts=d.get("query_gpu_power_avg_watts"),
            query_cpu_power_avg_watts=d.get("query_cpu_power_avg_watts"),
            judge_gpu_energy_joules=d.get("judge_gpu_energy_joules"),
            judge_cpu_energy_joules=d.get("judge_cpu_energy_joules"),
            judge_gpu_power_avg_watts=d.get("judge_gpu_power_avg_watts"),
            judge_cpu_power_avg_watts=d.get("judge_cpu_power_avg_watts"),
            judge_wall_clock_s=d.get("judge_wall_clock_s"),
            query_mbu_avg_pct=d.get("query_mbu_avg_pct"),
            query_mbu_max_pct=d.get("query_mbu_max_pct"),
            is_resolved=d.get("is_resolved"),
            unscorable_reason=d.get("unscorable_reason"),
            score_metadata=d.get("score_metadata") or {},
            utility_complete=d.get("utility_complete", False),
            utility_provenance=d.get("utility_provenance", "unknown"),
            utility_calibration_id=d.get("utility_calibration_id"),
            lm_energy_measurable=d.get("lm_energy_measurable", True),
            shared_device_warning=d.get("shared_device_warning", False),
            gpu_energy_complete=d.get("gpu_energy_complete"),
            gpu_energy_provenance=d.get("gpu_energy_provenance"),
            accelerator_device_coverage_complete=d.get("accelerator_device_coverage_complete"),
            utility_verifier_id=d.get("utility_verifier_id"),
            utility_verifier_version=d.get("utility_verifier_version"),
            utility_task_manifest_id=d.get("utility_task_manifest_id"),
            utility_task_membership_id=d.get("utility_task_membership_id"),
            utility_task_manifest_size=d.get("utility_task_manifest_size"),
            accuracy_score=d.get("accuracy_score"),
            accuracy_metadata=d.get("accuracy_metadata", {}),
            n_retries=d.get("n_retries", 0),
        )

    def save_jsonl(self, path: Path) -> None:
        """Append this trace as a JSONL line."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(self.to_dict()) + "\n")

    @classmethod
    def load_jsonl(cls, path: Path) -> List[QueryTrace]:
        """Load traces from a JSONL file."""
        traces = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    traces.append(cls.from_dict(json.loads(line)))
        return traces

    @staticmethod
    def to_hf_dataset(traces: List[QueryTrace]) -> Any:
        """Convert a list of QueryTrace objects to a HuggingFace Dataset.

        Returns:
            A datasets.Dataset with one row per trace.
        """
        from datasets import Dataset

        rows = []
        for trace in traces:
            rows.append(
                {
                    "query_id": trace.query_id,
                    "workload_type": trace.workload_type,
                    "query_text": trace.query_text,
                    "response_text": trace.response_text,
                    "num_turns": trace.num_turns,
                    "total_input_tokens": trace.total_input_tokens,
                    "total_output_tokens": trace.total_output_tokens,
                    "total_tool_calls": trace.total_tool_calls,
                    "total_wall_clock_s": trace.total_wall_clock_s,
                    "total_gpu_energy_joules": trace.total_gpu_energy_joules,
                    "total_cpu_energy_joules": trace.total_cpu_energy_joules,
                    "complete_total_gpu_energy_joules": trace.complete_total_gpu_energy_joules,
                    "complete_total_cpu_energy_joules": trace.complete_total_cpu_energy_joules,
                    "judge_gpu_energy_joules": trace.judge_gpu_energy_joules,
                    "judge_cpu_energy_joules": trace.judge_cpu_energy_joules,
                    "judge_wall_clock_s": trace.judge_wall_clock_s,
                    "total_task_gpu_energy_joules": trace.total_task_gpu_energy_joules,
                    "total_task_cpu_energy_joules": trace.total_task_cpu_energy_joules,
                    "total_tokens": trace.total_tokens,
                    "total_cost_usd": trace.total_cost_usd,
                    "legacy_lm_only_cost_usd": trace.legacy_lm_only_cost_usd,
                    "complete_total_cost_usd": trace.complete_total_cost_usd,
                    "cost_known_turns": trace.cost_known_turns,
                    "cost_unknown_turns": trace.cost_unknown_turns,
                    "cost_coverage": trace.cost_coverage,
                    "avg_gpu_power_watts": trace.avg_gpu_power_watts,
                    "avg_cpu_power_watts": trace.avg_cpu_power_watts,
                    "throughput_tokens_per_sec": trace.throughput_tokens_per_sec,
                    "energy_per_token_joules": trace.energy_per_token_joules,
                    "complete_energy_per_token_joules": (trace.complete_energy_per_token_joules),
                    "query_mbu_avg_pct": trace.query_mbu_avg_pct,
                    "query_mbu_max_pct": trace.query_mbu_max_pct,
                    "completed": trace.completed,
                    "timed_out": trace.timed_out,
                    "is_resolved": trace.is_resolved,
                    "unscorable_reason": trace.unscorable_reason,
                    "score_metadata_json": json.dumps(trace.score_metadata),
                    "utility_complete": trace.utility_complete,
                    "utility_provenance": trace.utility_provenance,
                    "utility_calibration_id": trace.utility_calibration_id,
                    "utility_verifier_id": trace.utility_verifier_id,
                    "utility_verifier_version": trace.utility_verifier_version,
                    "utility_task_manifest_id": trace.utility_task_manifest_id,
                    "utility_task_membership_id": trace.utility_task_membership_id,
                    "utility_task_manifest_size": trace.utility_task_manifest_size,
                    "accelerator_device_coverage_complete": (trace.accelerator_device_coverage_complete),
                    "trace_json": json.dumps(trace.to_dict()),
                }
            )
        return Dataset.from_list(rows)


__all__ = ["TurnTrace", "QueryTrace"]
