"""Agentic runner for multi-turn agent benchmarking with energy telemetry."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
import re
import statistics
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

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
    PowerMetrics,
    ProfilingRecord,
    TokenMetrics,
)
from ..telemetry.events import EventRecorder, EventType

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Energy computation helpers
# ---------------------------------------------------------------------------


def _compute_energy_delta(
    readings: list[TelemetrySample],
    field: str,
) -> float | None:
    """Compute energy delta from first to last reading for *field*."""
    values = [
        getattr(s.reading, field)
        for s in readings
        if getattr(s.reading, field, None) is not None
        and math.isfinite(getattr(s.reading, field))
    ]
    if len(values) >= 2:
        delta = values[-1] - values[0]
        return delta if delta >= 0 else None
    return None


def _compute_power_avg(
    readings: list[TelemetrySample],
    field: str,
) -> float | None:
    """Compute average power across readings for *field*."""
    values = [
        getattr(s.reading, field)
        for s in readings
        if getattr(s.reading, field, None) is not None
        and math.isfinite(getattr(s.reading, field))
    ]
    return statistics.mean(values) if values else None


# ---------------------------------------------------------------------------
# Patch extraction helpers
# ---------------------------------------------------------------------------

_FENCED_DIFF_RE = re.compile(
    r"```(?:diff|patch)\s*\n(.*?)```", re.DOTALL
)
_UNIFIED_DIFF_MARKERS = ("diff --git", "--- a/", "+++ b/", "@@ ")


def _extract_patch(text: str) -> Optional[str]:
    """Extract a unified-diff patch from agent response text.

    Looks for fenced ``diff`` code blocks first, then falls back to raw
    unified-diff markers.  Returns ``None`` when no patch is detected.
    """
    # 1. Fenced ```diff blocks
    fenced = _FENCED_DIFF_RE.findall(text)
    if fenced:
        return "\n\n".join(block.strip() for block in fenced)

    # 2. Raw unified diff markers
    lines = text.splitlines()
    patch_lines: list[str] = []
    in_diff = False
    for line in lines:
        if any(line.startswith(m) for m in _UNIFIED_DIFF_MARKERS):
            in_diff = True
        if in_diff:
            patch_lines.append(line)

    if patch_lines:
        return "\n".join(patch_lines)
    return None


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
        run_dir: Optional[Path] = None,
        concurrency: int = 1,
        agent_factory: Optional[Callable[[], BaseAgent]] = None,
        query_timeout: Optional[float] = None,
    ) -> None:
        self._agent = agent
        self._dataset = dataset
        self._telemetry = telemetry_session
        self._config = config or {}
        self._event_recorder = event_recorder if event_recorder is not None else EventRecorder()
        self._run_dir = run_dir
        self._traces: list[QueryTrace] = []
        self._records: list[ProfilingRecord] = []
        self._concurrency = max(1, concurrency)
        self._agent_factory = agent_factory
        self._query_timeout = query_timeout
        self._results_lock = threading.Lock()

    async def run(self, max_queries: Optional[int] = None) -> list[QueryTrace]:
        """Run the agent over the dataset, collecting traces and telemetry.

        Args:
            max_queries: Maximum number of queries to process. None means all.

        Returns:
            List of QueryTrace objects with energy-correlated telemetry.
        """
        total = max_queries or self._dataset.size()
        model = self._config.get("model", "unknown")

        # Collect the records we'll process
        work_items: list[tuple[int, DatasetRecord]] = []
        for index, record in enumerate(self._dataset):
            if index >= total:
                break
            work_items.append((index, record))

        if self._concurrency <= 1:
            return await self._run_sequential(work_items, model)
        return await self._run_concurrent(work_items, model)

    async def _run_sequential(
        self,
        work_items: list[tuple[int, DatasetRecord]],
        model: str,
    ) -> list[QueryTrace]:
        """Original sequential execution path."""
        with tqdm(total=len(work_items), desc="Agent run", unit="query") as progress:
            for index, record in work_items:
                query_id = f"q{index:04d}"
                start_time = time.time()
                try:
                    fut = self._run_single_query(
                        index, record, model, self._agent, self._event_recorder
                    )
                    if self._query_timeout:
                        trace = await asyncio.wait_for(fut, timeout=self._query_timeout)
                    else:
                        trace = await fut
                except asyncio.TimeoutError:
                    elapsed = time.time() - start_time
                    LOGGER.warning(
                        "Query %s timed out after %.0fs (limit=%ss)",
                        query_id, elapsed, self._query_timeout,
                    )
                    workload_type = record.dataset_metadata.get("workload_type", "agentic")
                    trace = QueryTrace(
                        query_id=query_id,
                        workload_type=str(workload_type),
                        query_text=record.problem,
                        response_text=f"Query timed out after {elapsed:.0f}s",
                        total_wall_clock_s=elapsed,
                        completed=False,
                        is_resolved=record.dataset_metadata.get("is_resolved"),
                    )
                self._traces.append(trace)

                profiling_record = self._build_profiling_record(
                    record, trace, model
                )
                self._records.append(profiling_record)

                if self._run_dir:
                    self._save_query_artifacts(index, record, trace)

                if len(self._traces) % self._FLUSH_INTERVAL == 0:
                    LOGGER.debug(
                        "Processed %d/%d queries",
                        len(self._traces),
                        len(work_items),
                    )

                progress.update(1)

        return self._traces

    async def _run_concurrent(
        self,
        work_items: list[tuple[int, DatasetRecord]],
        model: str,
    ) -> list[QueryTrace]:
        """Run tasks concurrently using a thread pool.

        Each task gets its own agent instance (from agent_factory) to avoid
        shared state conflicts.  Results are collected in index order.
        """
        total = len(work_items)
        LOGGER.info(
            "Running %d queries with concurrency=%d",
            total,
            self._concurrency,
        )

        # Pre-allocate result slots so we preserve index ordering
        result_slots: list[Optional[tuple[QueryTrace, ProfilingRecord]]] = [
            None
        ] * total
        progress = tqdm(total=total, desc="Agent run", unit="query")
        semaphore = asyncio.Semaphore(self._concurrency)
        loop = asyncio.get_event_loop()

        async def _process(slot: int, index: int, record: DatasetRecord) -> None:
            async with semaphore:
                # Each concurrent task gets a fresh agent + event recorder
                if self._agent_factory is not None:
                    agent = self._agent_factory()
                else:
                    agent = copy.deepcopy(self._agent)
                recorder = EventRecorder()

                query_id = f"q{index:04d}"
                start_time = time.time()

                try:
                    # Run the blocking work in a thread, with optional timeout
                    fut = loop.run_in_executor(
                        None,
                        self._run_single_query_sync,
                        index,
                        record,
                        model,
                        agent,
                        recorder,
                    )
                    if self._query_timeout:
                        trace = await asyncio.wait_for(fut, timeout=self._query_timeout)
                    else:
                        trace = await fut
                except asyncio.TimeoutError:
                    elapsed = time.time() - start_time
                    LOGGER.warning(
                        "Query %s timed out after %.0fs (limit=%ss)",
                        query_id, elapsed, self._query_timeout,
                    )
                    workload_type = record.dataset_metadata.get("workload_type", "agentic")
                    trace = QueryTrace(
                        query_id=query_id,
                        workload_type=str(workload_type),
                        query_text=record.problem,
                        response_text=f"Query timed out after {elapsed:.0f}s",
                        total_wall_clock_s=elapsed,
                        completed=False,
                        is_resolved=record.dataset_metadata.get("is_resolved"),
                    )

                profiling_record = self._build_profiling_record(
                    record, trace, model
                )

                if self._run_dir:
                    self._save_query_artifacts(index, record, trace)

                with self._results_lock:
                    result_slots[slot] = (trace, profiling_record)
                    progress.update(1)

        tasks = [
            _process(slot, index, record)
            for slot, (index, record) in enumerate(work_items)
        ]
        await asyncio.gather(*tasks)
        progress.close()

        # Collect results in original order
        for slot_result in result_slots:
            if slot_result is not None:
                trace, profiling_record = slot_result
                self._traces.append(trace)
                self._records.append(profiling_record)

        return self._traces

    def _run_single_query_sync(
        self,
        index: int,
        record: DatasetRecord,
        model: str,
        agent: BaseAgent,
        event_recorder: EventRecorder,
    ) -> QueryTrace:
        """Synchronous wrapper for _run_single_query (used by thread pool)."""
        return asyncio.run(
            self._run_single_query(index, record, model, agent, event_recorder)
        )

    async def _run_single_query(
        self,
        index: int,
        record: DatasetRecord,
        model: str,
        agent: Optional[BaseAgent] = None,
        event_recorder: Optional[EventRecorder] = None,
    ) -> QueryTrace:
        """Run a single query through the agent with telemetry capture."""
        agent = agent or self._agent
        event_recorder = event_recorder or self._event_recorder

        query_id = f"q{index:04d}"
        workload_type = record.dataset_metadata.get("workload_type", "agentic")

        # Capture telemetry window around the agent call
        start_time = time.time()
        _telemetry_samples_before = (  # noqa: F841
            list(self._telemetry.readings()) if self._telemetry else []
        )

        event_recorder.clear()

        # Set up per-query workspace for agents that support it
        if self._run_dir and hasattr(agent, "set_workspace"):
            instance_id = record.dataset_metadata.get("instance_id", "")
            slug = re.sub(r"[^a-zA-Z0-9_-]", "_", str(instance_id))[:80]
            workspace = (
                self._run_dir / "artifacts" / f"q{index:04d}_{slug}" / "workspace"
            )
            workspace.mkdir(parents=True, exist_ok=True)
            agent.set_workspace(str(workspace))

        # Create per-task execution environment (e.g. Docker for TerminalBench)
        from contextlib import nullcontext

        task_env = self._dataset.create_task_env(record)
        ctx = task_env if task_env is not None else nullcontext()

        try:
            with ctx:
                # set_task_metadata INSIDE context so metadata has session
                agent.set_task_metadata(record.dataset_metadata)

                result: AgentRunResult = agent.run(record.problem)

                if task_env is not None:
                    task_env.run_tests()
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
                is_resolved=record.dataset_metadata.get("is_resolved"),
            )
            return trace

        end_time = time.time()

        # Collect telemetry samples for this query window
        readings: list[TelemetrySample] = []
        if self._telemetry:
            readings = list(self._telemetry.window(start_time, end_time))

        # Build turn traces from event recorder
        events = event_recorder.get_events()
        turns = self._build_turn_traces(events, readings)

        # When EventRecorder captured nothing, create a synthetic turn from
        # AgentRunResult so token counts and wall clock are preserved.
        if not turns and (result.input_tokens > 0 or result.output_tokens > 0):
            turns = [TurnTrace(
                turn_index=0,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                wall_clock_s=end_time - start_time,
                cost_usd=result.cost_usd if result.cost_usd is not None else None,
            )]

        # Backfill tokens from AgentRunResult when turns have zero tokens
        # (e.g. OpenHands fires lm_inference events without token metadata)
        if turns and result.input_tokens > 0 and result.output_tokens > 0:
            total_turn_in = sum(t.input_tokens for t in turns)
            total_turn_out = sum(t.output_tokens for t in turns)
            if total_turn_in == 0 and total_turn_out == 0:
                turns[0].input_tokens = result.input_tokens
                turns[0].output_tokens = result.output_tokens
                turns[0].wall_clock_s = turns[0].wall_clock_s or (end_time - start_time)
                if result.cost_usd is not None and turns[0].cost_usd is None:
                    turns[0].cost_usd = result.cost_usd

        # Always compute query-level energy from telemetry window
        query_gpu_energy = _compute_energy_delta(readings, "energy_joules")
        query_cpu_energy = _compute_energy_delta(readings, "cpu_energy_joules")
        query_gpu_power_avg = _compute_power_avg(readings, "power_watts")
        query_cpu_power_avg = _compute_power_avg(readings, "cpu_power_watts")

        trace = QueryTrace(
            query_id=query_id,
            workload_type=str(workload_type),
            query_text=record.problem,
            response_text=result.content,
            turns=turns,
            total_wall_clock_s=end_time - start_time,
            completed=True,
            query_gpu_energy_joules=query_gpu_energy,
            query_cpu_energy_joules=query_cpu_energy,
            query_gpu_power_avg_watts=query_gpu_power_avg,
            query_cpu_power_avg_watts=query_cpu_power_avg,
            is_resolved=record.dataset_metadata.get("is_resolved"),
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

    def _save_query_artifacts(
        self,
        index: int,
        record: DatasetRecord,
        trace: QueryTrace,
    ) -> None:
        """Save per-query artifacts to structured subdirectories."""
        assert self._run_dir is not None
        instance_id = record.dataset_metadata.get("instance_id", "")
        slug = re.sub(r"[^a-zA-Z0-9_-]", "_", str(instance_id))[:80]
        query_dir = self._run_dir / "artifacts" / f"q{index:04d}_{slug}"
        query_dir.mkdir(parents=True, exist_ok=True)

        # response.txt — full agent response
        (query_dir / "response.txt").write_text(
            trace.response_text or "", encoding="utf-8"
        )

        # metadata.json — query-level metadata
        meta: dict[str, object] = {
            "query_id": trace.query_id,
            "instance_id": str(instance_id),
            "completed": trace.completed,
            "wall_clock_s": trace.total_wall_clock_s,
            "num_turns": trace.num_turns,
        }
        # Include select dataset metadata
        for key in ("repo", "base_commit", "dataset_name", "is_resolved", "test_results"):
            val = record.dataset_metadata.get(key)
            if val is not None:
                meta[key] = val
        (query_dir / "metadata.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8"
        )

        # patch.diff — extracted patch (if present)
        patch = _extract_patch(trace.response_text or "")
        if patch:
            (query_dir / "patch.diff").write_text(patch, encoding="utf-8")

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

        # Energy metrics from trace (per-turn sums, falling back to query-level)
        gpu_energy = trace.total_gpu_energy_joules
        cpu_energy = trace.total_cpu_energy_joules

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

        # Cost — use trace cost if available, otherwise compute from pricing.
        # Note: AgentRunResult.cost_usd defaults to 0.0 (not Optional), so
        # treat 0.0 as "not provided" and try pricing tables. The localhost
        # fallback below ensures local models still get cost=0.0.
        cost = trace.total_cost_usd
        if (cost is None or cost == 0.0) and total_input_tokens > 0:
            from ..cost.pricing import calculate_cost

            provider = self._config.get("provider", "")
            cost = calculate_cost(provider, model, total_input_tokens, total_output_tokens)
            if cost == 0.0:
                cost = None

        # Local models (localhost inference) have zero dollar cost
        base_url = self._config.get("client_base_url", "")
        if cost is None and ("localhost" in base_url or "127.0.0.1" in base_url):
            cost = 0.0

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
