"""Agentic runner for multi-turn agent benchmarking with energy telemetry."""

from __future__ import annotations

import logging
import math
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Sequence

from tqdm.auto import tqdm

from ..agents.base import BaseAgent
from ..core.types import AgentRunResult, DatasetRecord
from ..datasets.base import DatasetProvider
from ..execution.telemetry_session import TelemetrySample, TelemetrySession
from ..execution.trace import QueryTrace, TurnTrace
from ..execution.types import (
    ComputeMetrics,
    CostMetrics,
    EnergyMetrics,
    LatencyMetrics,
    MemoryMetrics,
    MetricStats,
    ModelMetrics,
    PowerComponentMetrics,
    PowerMetrics,
    ProfilingRecord,
    TokenMetrics,
)
from ..telemetry.events import EventRecorder, EventType

LOGGER = logging.getLogger(__name__)


class AgenticRunner:
    """Orchestrate multi-turn agent runs with energy telemetry correlation.

    Similar to ProfilerRunner but designed for agentic workloads where a single
    query may involve multiple LLM turns and tool calls.
    """

    _FLUSH_INTERVAL = 50

    def __init__(
        self,
        agent: BaseAgent,
        dataset: DatasetProvider,
        telemetry_session: Optional[TelemetrySession] = None,
        config: Optional[dict[str, Any]] = None,
        event_recorder: Optional[EventRecorder] = None,
    ) -> None:
        self._agent = agent
        self._dataset = dataset
        self._telemetry = telemetry_session
        self._config = config or {}
        self._event_recorder = event_recorder if event_recorder is not None else EventRecorder()
        self._traces: list[QueryTrace] = []
        self._records: list[ProfilingRecord] = []

    async def run(self, max_queries: Optional[int] = None) -> list[QueryTrace]:
        """Run the agent over the dataset, collecting traces and telemetry.

        Args:
            max_queries: Maximum number of queries to process. None means all.

        Returns:
            List of QueryTrace objects with energy-correlated telemetry.
        """
        total = max_queries or self._dataset.size()
        model = self._config.get("model", "unknown")

        with tqdm(total=total, desc="Agent run", unit="query") as progress:
            for index, record in enumerate(self._dataset):
                if index >= total:
                    break

                trace = await self._run_single_query(index, record, model)
                self._traces.append(trace)

                # Build profiling record for energy data persistence
                profiling_record = self._build_profiling_record(
                    record, trace, model
                )
                self._records.append(profiling_record)

                if len(self._traces) % self._FLUSH_INTERVAL == 0:
                    LOGGER.debug(
                        "Processed %d/%d queries", len(self._traces), total
                    )

                progress.update(1)

        return self._traces

    async def _run_single_query(
        self,
        index: int,
        record: DatasetRecord,
        model: str,
    ) -> QueryTrace:
        """Run a single query through the agent with telemetry capture."""
        query_id = f"q{index:04d}"
        workload_type = record.dataset_metadata.get("workload_type", "agentic")

        # Capture telemetry window around the agent call
        start_time = time.time()
        telemetry_samples_before = (
            list(self._telemetry.readings()) if self._telemetry else []
        )

        self._event_recorder.clear()

        # Run the agent
        try:
            result: AgentRunResult = self._agent.run(record.problem)
        except Exception as exc:
            LOGGER.warning("Agent failed on query %s: %s", query_id, exc)
            end_time = time.time()
            trace = QueryTrace(
                query_id=query_id,
                workload_type=str(workload_type),
                query_text=record.problem,
                response_text=str(exc),
                total_wall_clock_s=end_time - start_time,
                completed=False,
            )
            return trace

        end_time = time.time()

        # Collect telemetry samples for this query window
        readings: list[TelemetrySample] = []
        if self._telemetry:
            readings = list(self._telemetry.window(start_time, end_time))

        # Build turn traces from event recorder
        events = self._event_recorder.get_events()
        turns = self._build_turn_traces(events, readings)

        trace = QueryTrace(
            query_id=query_id,
            workload_type=str(workload_type),
            query_text=record.problem,
            response_text=result.content,
            turns=turns,
            total_wall_clock_s=end_time - start_time,
            completed=True,
        )

        # Correlate energy data with trace
        trace = self._correlate_energy(trace, readings)

        return trace

    def _build_turn_traces(
        self,
        events: list,
        readings: list[TelemetrySample],
    ) -> list[TurnTrace]:
        """Build TurnTrace objects from recorded events."""
        turns: list[TurnTrace] = []
        current_turn_index = 0
        current_turn_start: Optional[float] = None
        current_tools: list[str] = []
        current_tool_latencies: dict[str, float] = {}
        tool_start_times: dict[str, float] = {}
        input_tokens = 0
        output_tokens = 0

        for event in events:
            etype = event.event_type

            if etype == EventType.LM_INFERENCE_START:
                current_turn_start = event.timestamp

            elif etype == EventType.LM_INFERENCE_END:
                wall_clock = 0.0
                if current_turn_start is not None:
                    wall_clock = event.timestamp - current_turn_start

                input_tokens = event.metadata.get("prompt_tokens", 0)
                output_tokens = event.metadata.get("completion_tokens", 0)

                # Get energy readings for this turn window
                turn_gpu_energy = None
                turn_cpu_energy = None
                turn_gpu_power_avg = None
                turn_cpu_power_avg = None

                if current_turn_start is not None and readings:
                    turn_readings = [
                        s for s in readings
                        if current_turn_start <= s.timestamp <= event.timestamp
                    ]
                    if turn_readings:
                        gpu_energies = [
                            s.reading.energy_joules for s in turn_readings
                            if s.reading.energy_joules is not None
                            and math.isfinite(s.reading.energy_joules)
                        ]
                        if len(gpu_energies) >= 2:
                            delta = gpu_energies[-1] - gpu_energies[0]
                            turn_gpu_energy = delta if delta >= 0 else None

                        cpu_energies = [
                            s.reading.cpu_energy_joules for s in turn_readings
                            if s.reading.cpu_energy_joules is not None
                            and math.isfinite(s.reading.cpu_energy_joules)
                        ]
                        if len(cpu_energies) >= 2:
                            delta = cpu_energies[-1] - cpu_energies[0]
                            turn_cpu_energy = delta if delta >= 0 else None

                        gpu_powers = [
                            s.reading.power_watts for s in turn_readings
                            if s.reading.power_watts is not None
                            and math.isfinite(s.reading.power_watts)
                        ]
                        if gpu_powers:
                            turn_gpu_power_avg = statistics.mean(gpu_powers)

                        cpu_powers = [
                            s.reading.cpu_power_watts for s in turn_readings
                            if s.reading.cpu_power_watts is not None
                            and math.isfinite(s.reading.cpu_power_watts)
                        ]
                        if cpu_powers:
                            turn_cpu_power_avg = statistics.mean(cpu_powers)

                turn = TurnTrace(
                    turn_index=current_turn_index,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    tools_called=list(current_tools),
                    tool_latencies_s=dict(current_tool_latencies),
                    wall_clock_s=wall_clock,
                    gpu_energy_joules=turn_gpu_energy,
                    cpu_energy_joules=turn_cpu_energy,
                    gpu_power_avg_watts=turn_gpu_power_avg,
                    cpu_power_avg_watts=turn_cpu_power_avg,
                )
                turns.append(turn)

                # Reset for next turn
                current_turn_index += 1
                current_turn_start = None
                current_tools = []
                current_tool_latencies = {}
                input_tokens = 0
                output_tokens = 0

            elif etype == EventType.TOOL_CALL_START:
                tool_name = event.metadata.get("tool", "unknown")
                tool_start_times[tool_name] = event.timestamp

            elif etype == EventType.TOOL_CALL_END:
                tool_name = event.metadata.get("tool", "unknown")
                current_tools.append(tool_name)
                start_ts = tool_start_times.pop(tool_name, None)
                if start_ts is not None:
                    current_tool_latencies[tool_name] = (
                        event.timestamp - start_ts
                    )

        # If there were events but no complete turn, create a synthetic one
        if not turns and events:
            turns.append(
                TurnTrace(
                    turn_index=0,
                    tools_called=current_tools,
                    tool_latencies_s=current_tool_latencies,
                )
            )

        return turns

    def _correlate_energy(
        self,
        trace: QueryTrace,
        readings: list[TelemetrySample],
    ) -> QueryTrace:
        """Correlate energy readings with the trace at the query level.

        If per-turn energy was not populated from events (e.g., no event
        recorder), distribute energy evenly across turns based on wall clock.
        """
        if not readings or not trace.turns:
            return trace

        # Check if any turns already have energy data
        has_turn_energy = any(
            t.gpu_energy_joules is not None for t in trace.turns
        )
        if has_turn_energy:
            return trace

        # Compute total query energy
        gpu_energies = [
            s.reading.energy_joules for s in readings
            if s.reading.energy_joules is not None
            and math.isfinite(s.reading.energy_joules)
        ]
        total_gpu_energy = None
        if len(gpu_energies) >= 2:
            delta = gpu_energies[-1] - gpu_energies[0]
            total_gpu_energy = delta if delta >= 0 else None

        cpu_energies = [
            s.reading.cpu_energy_joules for s in readings
            if s.reading.cpu_energy_joules is not None
            and math.isfinite(s.reading.cpu_energy_joules)
        ]
        total_cpu_energy = None
        if len(cpu_energies) >= 2:
            delta = cpu_energies[-1] - cpu_energies[0]
            total_cpu_energy = delta if delta >= 0 else None

        # Distribute proportionally by wall clock time
        total_wall = sum(t.wall_clock_s for t in trace.turns)
        if total_wall > 0:
            for turn in trace.turns:
                fraction = turn.wall_clock_s / total_wall
                if total_gpu_energy is not None:
                    turn.gpu_energy_joules = total_gpu_energy * fraction
                if total_cpu_energy is not None:
                    turn.cpu_energy_joules = total_cpu_energy * fraction

        return trace

    def _build_profiling_record(
        self,
        record: DatasetRecord,
        trace: QueryTrace,
        model: str,
    ) -> ProfilingRecord:
        """Build a ProfilingRecord from a completed query trace."""
        total_input_tokens = trace.total_input_tokens
        total_output_tokens = trace.total_output_tokens
        total_seconds = trace.total_wall_clock_s

        # Energy metrics from trace
        gpu_energy = trace.total_gpu_energy_joules
        cpu_energy = sum(
            (t.cpu_energy_joules for t in trace.turns if t.cpu_energy_joules is not None),
            0.0,
        ) or None

        energy_metrics = EnergyMetrics(
            per_query_joules=gpu_energy,
            total_joules=gpu_energy,
            cpu_per_query_joules=cpu_energy,
            cpu_total_joules=cpu_energy,
        )

        # Latency
        per_token_ms = None
        throughput = None
        if total_output_tokens > 0 and total_seconds > 0:
            per_token_ms = (total_seconds * 1000.0) / total_output_tokens
            throughput = total_output_tokens / total_seconds

        latency_metrics = LatencyMetrics(
            per_token_ms=per_token_ms,
            throughput_tokens_per_sec=throughput,
            total_query_seconds=total_seconds,
        )

        # Cost
        cost = trace.total_cost_usd
        cost_metrics = CostMetrics(total_cost_usd=cost)

        model_metrics = ModelMetrics(
            compute_metrics=ComputeMetrics(),
            energy_metrics=energy_metrics,
            latency_metrics=latency_metrics,
            memory_metrics=MemoryMetrics(),
            power_metrics=PowerMetrics(),
            temperature_metrics=MetricStats(),
            token_metrics=TokenMetrics(
                input=total_input_tokens,
                output=total_output_tokens,
                total=total_input_tokens + total_output_tokens,
            ),
            cost=cost_metrics,
            lm_response=trace.response_text,
        )

        return ProfilingRecord(
            problem=record.problem,
            answer=record.answer,
            dataset_metadata=dict(record.dataset_metadata),
            subject=record.subject,
            model_answers={model: trace.response_text},
            model_metrics={model: model_metrics},
        )

    @property
    def traces(self) -> list[QueryTrace]:
        """Return collected traces."""
        return list(self._traces)

    @property
    def records(self) -> list[ProfilingRecord]:
        """Return collected profiling records."""
        return list(self._records)


__all__ = ["AgenticRunner"]
