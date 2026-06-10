"""Trace data model for eval benchmark runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


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
    cost_usd: Optional[float] = None

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
        )


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
    query_mbu_avg_pct: Optional[float] = None
    query_mbu_max_pct: Optional[float] = None
    is_resolved: Optional[bool] = None
    unscorable_reason: Optional[str] = None
    score_metadata: Dict[str, Any] = field(default_factory=dict)

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
        if self.query_gpu_energy_joules is not None:
            return self.query_gpu_energy_joules
        values = [t.gpu_energy_joules for t in self.turns if t.gpu_energy_joules is not None]
        if values:
            return sum(values)
        return None

    @property
    def total_cpu_energy_joules(self) -> Optional[float]:
        if self.query_cpu_energy_joules is not None:
            return self.query_cpu_energy_joules
        values = [t.cpu_energy_joules for t in self.turns if t.cpu_energy_joules is not None]
        if values:
            return sum(values)
        return None

    @property
    def total_cost_usd(self) -> Optional[float]:
        values = [t.cost_usd for t in self.turns if t.cost_usd is not None]
        return sum(values) if values else None

    @property
    def total_tokens(self) -> Optional[int]:
        """Total tokens (input + output) across all turns."""
        if self.total_input_tokens is None or self.total_output_tokens is None:
            return None
        return self.total_input_tokens + self.total_output_tokens

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
            "query_mbu_avg_pct": self.query_mbu_avg_pct,
            "query_mbu_max_pct": self.query_mbu_max_pct,
            "throughput_tokens_per_sec": self.throughput_tokens_per_sec,
            "energy_per_output_token_joules": self.energy_per_token_joules,
            "total_cost_usd": self.total_cost_usd,
            "is_resolved": self.is_resolved,
            "unscorable_reason": self.unscorable_reason,
            "score_metadata": dict(self.score_metadata),
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
            query_mbu_avg_pct=d.get("query_mbu_avg_pct"),
            query_mbu_max_pct=d.get("query_mbu_max_pct"),
            is_resolved=d.get("is_resolved"),
            unscorable_reason=d.get("unscorable_reason"),
            score_metadata=d.get("score_metadata") or {},
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
            rows.append({
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
                "total_tokens": trace.total_tokens,
                "total_cost_usd": trace.total_cost_usd,
                "avg_gpu_power_watts": trace.avg_gpu_power_watts,
                "avg_cpu_power_watts": trace.avg_cpu_power_watts,
                "throughput_tokens_per_sec": trace.throughput_tokens_per_sec,
                "energy_per_token_joules": trace.energy_per_token_joules,
                "query_mbu_avg_pct": trace.query_mbu_avg_pct,
                "query_mbu_max_pct": trace.query_mbu_max_pct,
                "completed": trace.completed,
                "timed_out": trace.timed_out,
                "is_resolved": trace.is_resolved,
                "unscorable_reason": trace.unscorable_reason,
                "score_metadata_json": json.dumps(trace.score_metadata),
                "trace_json": json.dumps(trace.to_dict()),
            })
        return Dataset.from_list(rows)


__all__ = ["TurnTrace", "QueryTrace"]
