"""Agentic runner for multi-turn agent benchmarking with energy telemetry."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import math
import re
import signal
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
    DerivedEfficiencyMetrics,
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


class _QueryTimeout(BaseException):
    """Internal timeout sentinel that bypasses per-query Exception handling."""


def _raise_query_timeout(_signum: int, _frame: Any) -> None:
    raise _QueryTimeout()


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


def _estimate_energy_from_power(
    readings: list[TelemetrySample],
    power_field: str,
    duration_s: float,
) -> float | None:
    """Fallback: energy ≈ avg_power × duration when cumulative counters unavailable."""
    if duration_s <= 0:
        return None
    avg_power = _compute_power_avg(readings, power_field)
    if avg_power is not None and avg_power > 0:
        return avg_power * duration_s
    return None


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

    async def run(
        self,
        max_queries: Optional[int] = None,
        *,
        start_index: int = 0,
    ) -> list[QueryTrace]:
        """Run the agent over the dataset, collecting traces and telemetry.

        Args:
            max_queries: Upper record index to process. None means all.
            start_index: First dataset record index to process.

        Returns:
            List of QueryTrace objects with energy-correlated telemetry.
        """
        start_index = max(0, int(start_index))
        end_index = max_queries if max_queries is not None else self._dataset.size()
        model = self._config.get("model", "unknown")

        # Collect the records we'll process
        work_items: list[tuple[int, DatasetRecord]] = []
        for index, record in enumerate(self._dataset):
            if index < start_index:
                continue
            if end_index is not None and index >= end_index:
                break
            work_items.append((index, record))

        if self._run_dir:
            self._write_subset_manifest(work_items, start_index, end_index)
            self._initialize_incremental_outputs()

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
                    if (
                        self._query_timeout
                        and threading.current_thread() is threading.main_thread()
                        and hasattr(signal, "setitimer")
                    ):
                        old_handler = signal.getsignal(signal.SIGALRM)
                        signal.signal(signal.SIGALRM, _raise_query_timeout)
                        signal.setitimer(signal.ITIMER_REAL, float(self._query_timeout))
                        try:
                            trace = await self._run_single_query(
                                index, record, model, self._agent, self._event_recorder
                            )
                        finally:
                            signal.setitimer(signal.ITIMER_REAL, 0)
                            signal.signal(signal.SIGALRM, old_handler)
                    else:
                        fut = self._run_single_query(
                            index, record, model, self._agent, self._event_recorder
                        )
                        if self._query_timeout:
                            trace = await asyncio.wait_for(fut, timeout=self._query_timeout)
                        else:
                            trace = await fut
                except (asyncio.TimeoutError, _QueryTimeout):
                    elapsed = time.time() - start_time
                    LOGGER.warning(
                        "Query %s timed out after %.0fs (limit=%ss)",
                        query_id, elapsed, self._query_timeout,
                    )
                    trace = self._build_timeout_trace(
                        query_id=query_id,
                        record=record,
                        elapsed=elapsed,
                        start_time=start_time,
                        event_recorder=self._event_recorder,
                    )
                self._traces.append(trace)

                # Log per-task latency
                status = "TIMEOUT" if trace.timed_out else ("OK" if trace.completed else "FAIL")
                LOGGER.info(
                    "Task %s: %s in %.1fs",
                    query_id, status, trace.total_wall_clock_s,
                )

                profiling_record = self._build_profiling_record(
                    record, trace, model
                )
                self._records.append(profiling_record)

                if self._run_dir:
                    self._save_query_artifacts(index, record, trace)
                    self._flush_incremental_outputs(trace, self._traces)

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
                    trace = self._build_timeout_trace(
                        query_id=query_id,
                        record=record,
                        elapsed=elapsed,
                        start_time=start_time,
                        event_recorder=recorder,
                    )

                # Log per-task latency
                status = "TIMEOUT" if trace.timed_out else ("OK" if trace.completed else "FAIL")
                LOGGER.info(
                    "Task %s: %s in %.1fs",
                    query_id, status, trace.total_wall_clock_s,
                )

                profiling_record = self._build_profiling_record(
                    record, trace, model
                )

                if self._run_dir:
                    self._save_query_artifacts(index, record, trace)

                with self._results_lock:
                    result_slots[slot] = (trace, profiling_record)
                    if self._run_dir:
                        partial_traces = [
                            item[0] for item in result_slots if item is not None
                        ]
                        self._flush_incremental_outputs(trace, partial_traces)
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

    def _build_timeout_trace(
        self,
        *,
        query_id: str,
        record: DatasetRecord,
        elapsed: float,
        start_time: float,
        event_recorder: EventRecorder,
    ) -> QueryTrace:
        """Create a metric-complete trace for a timed-out query."""
        end_time = time.time()
        readings = list(self._telemetry.window(start_time, end_time)) if self._telemetry else []
        turns = self._build_turn_traces(event_recorder.get_events(), readings)
        if not turns:
            turns = [
                TurnTrace(
                    turn_index=0,
                    input_tokens=None,
                    output_tokens=None,
                    wall_clock_s=elapsed,
                )
            ]
        elif sum((t.input_tokens or 0) + (t.output_tokens or 0) for t in turns) == 0:
            turns[0].input_tokens = None
            turns[0].output_tokens = None
            turns[0].wall_clock_s = turns[0].wall_clock_s or elapsed

        query_gpu_energy = _compute_energy_delta(readings, "energy_joules")
        query_cpu_energy = _compute_energy_delta(readings, "cpu_energy_joules")
        query_gpu_power_avg = _compute_power_avg(readings, "power_watts")
        query_cpu_power_avg = _compute_power_avg(readings, "cpu_power_watts")
        if query_gpu_energy is None and readings:
            query_gpu_energy = _estimate_energy_from_power(readings, "power_watts", elapsed)
        if query_cpu_energy is None and readings:
            query_cpu_energy = _estimate_energy_from_power(readings, "cpu_power_watts", elapsed)

        query_mbu_avg = None
        query_mbu_max = None
        if readings:
            mbu_values = [
                s.reading.gpu_memory_bandwidth_utilization_pct
                for s in readings
                if getattr(s.reading, "gpu_memory_bandwidth_utilization_pct", None) is not None
                and s.reading.gpu_memory_bandwidth_utilization_pct >= 0
            ]
            if mbu_values:
                query_mbu_avg = statistics.mean(mbu_values)
                query_mbu_max = max(mbu_values)

        workload_type = record.dataset_metadata.get("workload_type", "agentic")
        trace = QueryTrace(
            query_id=query_id,
            workload_type=str(workload_type),
            query_text=record.problem,
            response_text=f"Query timed out after {elapsed:.0f}s",
            turns=turns,
            total_wall_clock_s=elapsed,
            completed=False,
            timed_out=True,
            query_gpu_energy_joules=query_gpu_energy,
            query_cpu_energy_joules=query_cpu_energy,
            query_gpu_power_avg_watts=query_gpu_power_avg,
            query_cpu_power_avg_watts=query_cpu_power_avg,
            query_mbu_avg_pct=query_mbu_avg,
            query_mbu_max_pct=query_mbu_max,
            is_resolved=record.dataset_metadata.get("is_resolved"),
            unscorable_reason="timeout",
            score_metadata={"reason": "timeout", "timeout_s": self._query_timeout},
        )
        return self._correlate_energy(trace, readings)

    def _initialize_incremental_outputs(self) -> None:
        """Prepare per-query output files that are updated during the run."""
        assert self._run_dir is not None
        self._run_dir.mkdir(parents=True, exist_ok=True)
        (self._run_dir / "traces.jsonl").write_text("", encoding="utf-8")

    def _write_subset_manifest(
        self,
        work_items: list[tuple[int, DatasetRecord]],
        start_index: int,
        end_index: int | None,
    ) -> None:
        """Write a stable manifest for the exact dataset records in this run."""
        assert self._run_dir is not None
        self._run_dir.mkdir(parents=True, exist_ok=True)
        records = [
            self._subset_record_entry(index, record)
            for index, record in work_items
        ]
        subset_hash = hashlib.sha256(
            json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        manifest = {
            "schema_version": 1,
            "dataset": self._config.get("dataset"),
            "model": self._config.get("model"),
            "agent": self._config.get("agent"),
            "start_index": start_index,
            "end_index": end_index,
            "subset_size": len(records),
            "subset_hash": subset_hash,
            "records": records,
        }
        self._config["subset"] = {
            "subset_hash": subset_hash,
            "subset_size": len(records),
            "start_index": start_index,
            "end_index": end_index,
        }
        (self._run_dir / "subset_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @staticmethod
    def _subset_record_entry(index: int, record: DatasetRecord) -> dict[str, Any]:
        metadata = record.dataset_metadata or {}
        id_keys = (
            "task_id",
            "instance_id",
            "question_id",
            "uid",
            "original_index",
            "id",
        )
        stable_ids = {
            key: str(metadata[key])
            for key in id_keys
            if metadata.get(key) is not None
        }
        query_hash = hashlib.sha256(record.problem.encode("utf-8")).hexdigest()
        return {
            "index": index,
            "query_id": f"q{index:04d}",
            "query_hash": query_hash,
            "subject": record.subject,
            "stable_ids": stable_ids,
        }

    def _flush_incremental_outputs(
        self,
        trace: QueryTrace,
        traces_for_summary: list[QueryTrace],
    ) -> None:
        """Persist completed query trace data before the full run finishes."""
        assert self._run_dir is not None
        try:
            with (self._run_dir / "traces.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(trace.to_dict()) + "\n")

            from ..execution.exporters import export_summary_json

            export_summary_json(
                traces_for_summary,
                self._config,
                self._run_dir / "summary.json",
            )
        except Exception as exc:
            LOGGER.warning("Failed to flush incremental trace output: %s", exc)

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
            if hasattr(self._dataset, "prepare_workspace"):
                try:
                    self._dataset.prepare_workspace(record, workspace)
                except Exception as exc:
                    LOGGER.warning("Workspace setup failed for %s: %s", query_id, exc)
                    record.dataset_metadata["workspace_prepared"] = False
                    record.dataset_metadata["workspace_error"] = str(exc)
            agent.set_workspace(str(workspace))

        # Create per-task execution environment (e.g. Docker for TerminalBench)
        from contextlib import nullcontext

        # Inject model into metadata so task envs can use unique container names
        record.dataset_metadata.setdefault("model", model)

        task_env = self._dataset.create_task_env(record)
        ctx = task_env if task_env is not None else nullcontext()

        try:
            with ctx:
                # set_task_metadata INSIDE context so metadata has session
                agent.set_task_metadata(record.dataset_metadata)

                result: AgentRunResult = agent.run(record.problem)
                agent_metadata = dict(result.metadata or {})
                if agent_metadata:
                    record.dataset_metadata["agent_metadata"] = agent_metadata
                    token_source = agent_metadata.get("token_source")
                    if token_source is not None:
                        record.dataset_metadata["token_source"] = token_source

                if task_env is not None:
                    task_env.run_tests()
                    if (
                        "is_resolved" in record.dataset_metadata
                        and hasattr(self._dataset, "score")
                    ):
                        try:
                            is_correct, score_metadata = self._dataset.score(
                                record, result.content
                            )
                            record.dataset_metadata["is_resolved"] = is_correct
                            if score_metadata:
                                record.dataset_metadata["score_metadata"] = score_metadata
                            if is_correct is None:
                                reason = str(score_metadata.get("reason", "unscorable"))
                                record.dataset_metadata["unscorable_reason"] = reason
                        except Exception as score_exc:
                            LOGGER.warning("Scoring failed for %s: %s", query_id, score_exc)
                            record.dataset_metadata["unscorable_reason"] = "scoring_error"
                            record.dataset_metadata["score_metadata"] = {
                                "reason": "scoring_error",
                                "error": str(score_exc),
                            }
                elif hasattr(self._dataset, "score") and record.answer:
                    try:
                        is_correct, score_metadata = self._dataset.score(record, result.content)
                        record.dataset_metadata["is_resolved"] = is_correct
                        if score_metadata:
                            record.dataset_metadata["score_metadata"] = score_metadata
                        if is_correct is None:
                            reason = str(score_metadata.get("reason", "unscorable"))
                            record.dataset_metadata["unscorable_reason"] = reason
                    except Exception as score_exc:
                        LOGGER.warning("Scoring failed for %s: %s", query_id, score_exc)
                        record.dataset_metadata["unscorable_reason"] = "scoring_error"
                        record.dataset_metadata["score_metadata"] = {
                            "reason": "scoring_error",
                            "error": str(score_exc),
                        }
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
                unscorable_reason=str(
                    record.dataset_metadata.get("unscorable_reason", "agent_error")
                ),
                score_metadata={
                    "reason": str(
                        record.dataset_metadata.get("unscorable_reason", "agent_error")
                    ),
                    "error": str(exc),
                },
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
        result_has_token_counts = (
            result.input_tokens is not None
            and result.output_tokens is not None
        )
        if (
            not turns
            and result_has_token_counts
            and (result.input_tokens > 0 or result.output_tokens > 0)
        ):
            turns = [TurnTrace(
                turn_index=0,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                wall_clock_s=end_time - start_time,
                cost_usd=result.cost_usd if result.cost_usd is not None else None,
            )]

        # Backfill tokens from AgentRunResult when turns have zero tokens
        # (e.g. OpenHands fires lm_inference events without token metadata)
        if (
            turns
            and result_has_token_counts
            and result.input_tokens > 0
            and result.output_tokens > 0
        ):
            total_turn_in = sum(t.input_tokens or 0 for t in turns)
            total_turn_out = sum(t.output_tokens or 0 for t in turns)
            if total_turn_in == 0 and total_turn_out == 0:
                turns[0].input_tokens = result.input_tokens
                turns[0].output_tokens = result.output_tokens
                turns[0].wall_clock_s = turns[0].wall_clock_s or (end_time - start_time)
                if result.cost_usd is not None and turns[0].cost_usd is None:
                    turns[0].cost_usd = result.cost_usd
            else:
                missing_token_turns = [
                    t
                    for t in turns
                    if t.input_tokens is None
                    and t.output_tokens is None
                    and not t.tools_called
                    and t.error is None
                ]
                if missing_token_turns:
                    known_in = sum(
                        t.input_tokens or 0
                        for t in turns
                        if t.input_tokens is not None
                    )
                    known_out = sum(
                        t.output_tokens or 0
                        for t in turns
                        if t.output_tokens is not None
                    )
                    remaining_in = result.input_tokens - known_in
                    remaining_out = result.output_tokens - known_out
                    if remaining_in >= 0 and remaining_out >= 0:
                        missing_token_turns[0].input_tokens = remaining_in
                        missing_token_turns[0].output_tokens = remaining_out

        # Always compute query-level energy from telemetry window
        query_gpu_energy = _compute_energy_delta(readings, "energy_joules")
        query_cpu_energy = _compute_energy_delta(readings, "cpu_energy_joules")
        query_gpu_power_avg = _compute_power_avg(readings, "power_watts")
        query_cpu_power_avg = _compute_power_avg(readings, "cpu_power_watts")

        # Fallback: estimate energy from average power when cumulative counters
        # have fewer than 2 samples (no delta possible).
        duration = end_time - start_time
        if query_gpu_energy is None and readings:
            query_gpu_energy = _estimate_energy_from_power(readings, "power_watts", duration)
        if query_cpu_energy is None and readings:
            query_cpu_energy = _estimate_energy_from_power(readings, "cpu_power_watts", duration)

        # Extract MBU from telemetry samples
        query_mbu_avg = None
        query_mbu_max = None
        if readings:
            mbu_values = [
                s.reading.gpu_memory_bandwidth_utilization_pct
                for s in readings
                if getattr(s.reading, 'gpu_memory_bandwidth_utilization_pct', None) is not None
                and s.reading.gpu_memory_bandwidth_utilization_pct >= 0
            ]
            if mbu_values:
                query_mbu_avg = statistics.mean(mbu_values)
                query_mbu_max = max(mbu_values)

        score_metadata = dict(record.dataset_metadata.get("score_metadata") or {})
        agent_metadata = dict(record.dataset_metadata.get("agent_metadata") or {})
        if agent_metadata:
            score_metadata.setdefault("agent_metadata", agent_metadata)
            if agent_metadata.get("token_source") is not None:
                score_metadata.setdefault("token_source", agent_metadata["token_source"])

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
            query_mbu_avg_pct=query_mbu_avg,
            query_mbu_max_pct=query_mbu_max,
            is_resolved=record.dataset_metadata.get("is_resolved"),
            unscorable_reason=(
                str(record.dataset_metadata["unscorable_reason"])
                if record.dataset_metadata.get("unscorable_reason") is not None
                else None
            ),
            score_metadata=score_metadata,
        )

        # Correlate energy data with trace
        trace = self._correlate_energy(trace, readings)

        return trace

    def _build_turn_traces(
        self,
        events: list,
        readings: list[TelemetrySample],
    ) -> list[TurnTrace]:
        """Build TurnTrace objects from recorded events.

        A turn is modeled as one LLM call plus the tool calls produced by that
        model response. Tool calls usually happen after the ``lm_inference_end``
        event, so they are attached to the most recent completed LLM turn and
        extend that turn's telemetry window.
        """
        turn_records: list[dict[str, Any]] = []
        current_turn: Optional[dict[str, Any]] = None
        tool_start_times: dict[str, list[tuple[float, Optional[dict[str, Any]]]]] = {}

        def _new_turn(start_ts: float) -> dict[str, Any]:
            return {
                "start": start_ts,
                "end": start_ts,
                "input_tokens": None,
                "output_tokens": None,
                "cost_usd": None,
                "tools_called": [],
                "tool_latencies_s": {},
            }

        def _latency_key(latencies: dict[str, float], tool_name: str) -> str:
            if tool_name not in latencies:
                return tool_name
            suffix = 2
            while f"{tool_name}#{suffix}" in latencies:
                suffix += 1
            return f"{tool_name}#{suffix}"

        def _attach_tool(
            record: Optional[dict[str, Any]],
            tool_name: str,
            latency: Optional[float],
            end_ts: float,
        ) -> None:
            nonlocal current_turn
            if record is None:
                record = current_turn or (turn_records[-1] if turn_records else None)
            if record is None:
                record = _new_turn(end_ts)
                turn_records.append(record)
            record["tools_called"].append(tool_name)
            if latency is not None:
                record["tool_latencies_s"][
                    _latency_key(record["tool_latencies_s"], tool_name)
                ] = latency
            record["end"] = max(record["end"], end_ts)

        for event in events:
            etype = event.event_type

            if etype == EventType.LM_INFERENCE_START:
                current_turn = _new_turn(event.timestamp)

            elif etype == EventType.LM_INFERENCE_END:
                if current_turn is None:
                    current_turn = _new_turn(event.timestamp)
                current_turn["end"] = max(current_turn["end"], event.timestamp)
                current_turn["input_tokens"] = event.metadata.get("prompt_tokens")
                current_turn["output_tokens"] = event.metadata.get("completion_tokens")
                if event.metadata.get("cost_usd") is not None:
                    current_turn["cost_usd"] = event.metadata.get("cost_usd")
                turn_records.append(current_turn)
                current_turn = None

            elif etype == EventType.TOOL_CALL_START:
                tool_name = event.metadata.get("tool", "unknown")
                owner = current_turn or (turn_records[-1] if turn_records else None)
                tool_start_times.setdefault(tool_name, []).append(
                    (event.timestamp, owner)
                )

            elif etype == EventType.TOOL_CALL_END:
                tool_name = event.metadata.get("tool", "unknown")
                starts = tool_start_times.get(tool_name, [])
                start_ts = None
                owner = None
                if starts:
                    start_ts, owner = starts.pop()
                    if not starts:
                        tool_start_times.pop(tool_name, None)
                latency = (
                    event.timestamp - start_ts
                    if start_ts is not None
                    else None
                )
                _attach_tool(owner, tool_name, latency, event.timestamp)

        if current_turn is not None:
            turn_records.append(current_turn)

        turns: list[TurnTrace] = []
        for turn_index, record in enumerate(turn_records):
            start = record["start"]
            end = record["end"]
            wall_clock = max(0.0, end - start)
            turn_readings = [
                s for s in readings
                if start <= s.timestamp <= end
            ]
            turn_gpu_energy = _compute_energy_delta(turn_readings, "energy_joules")
            turn_cpu_energy = _compute_energy_delta(turn_readings, "cpu_energy_joules")
            turn_gpu_power_avg = _compute_power_avg(turn_readings, "power_watts")
            turn_cpu_power_avg = _compute_power_avg(turn_readings, "cpu_power_watts")

            if turn_gpu_energy is None and turn_readings:
                turn_gpu_energy = _estimate_energy_from_power(
                    turn_readings, "power_watts", wall_clock
                )
            if turn_cpu_energy is None and turn_readings:
                turn_cpu_energy = _estimate_energy_from_power(
                    turn_readings, "cpu_power_watts", wall_clock
                )

            turns.append(
                TurnTrace(
                    turn_index=turn_index,
                    input_tokens=record["input_tokens"],
                    output_tokens=record["output_tokens"],
                    tools_called=list(record["tools_called"]),
                    tool_latencies_s=dict(record["tool_latencies_s"]),
                    wall_clock_s=wall_clock,
                    gpu_energy_joules=turn_gpu_energy,
                    cpu_energy_joules=turn_cpu_energy,
                    gpu_power_avg_watts=turn_gpu_power_avg,
                    cpu_power_avg_watts=turn_cpu_power_avg,
                    cost_usd=record.get("cost_usd"),
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

        # Fallback: estimate from power when cumulative counters unavailable
        total_wall = sum(t.wall_clock_s for t in trace.turns)
        if total_gpu_energy is None and total_wall > 0:
            total_gpu_energy = _estimate_energy_from_power(readings, "power_watts", total_wall)
        if total_cpu_energy is None and total_wall > 0:
            total_cpu_energy = _estimate_energy_from_power(readings, "cpu_power_watts", total_wall)

        # Distribute proportionally by wall clock time
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
            "timed_out": trace.timed_out,
            "wall_clock_s": trace.total_wall_clock_s,
            "num_turns": trace.num_turns,
        }
        # Include select dataset metadata
        for key in (
            "repo",
            "base_commit",
            "dataset_name",
            "is_resolved",
            "unscorable_reason",
            "score_metadata",
            "test_results",
            "agent_metadata",
            "token_source",
        ):
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

        # Per-token energy normalization
        energy_per_output_token = None
        energy_per_total_token = None
        total_tokens = (
            total_input_tokens + total_output_tokens
            if total_input_tokens is not None and total_output_tokens is not None
            else None
        )
        if gpu_energy is not None and gpu_energy > 0:
            if total_output_tokens is not None and total_output_tokens > 0:
                energy_per_output_token = gpu_energy / total_output_tokens
            if total_tokens is not None and total_tokens > 0:
                energy_per_total_token = gpu_energy / total_tokens

        energy_metrics = EnergyMetrics(
            per_query_joules=gpu_energy,
            total_joules=gpu_energy,
            cpu_per_query_joules=cpu_energy,
            cpu_total_joules=cpu_energy,
            energy_per_output_token_joules=energy_per_output_token,
            energy_per_total_token_joules=energy_per_total_token,
        )

        # Latency
        per_token_ms = None
        throughput = None
        if total_output_tokens is not None and total_output_tokens > 0 and total_seconds > 0:
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
        if (
            (cost is None or cost == 0.0)
            and total_input_tokens is not None
            and total_output_tokens is not None
            and total_input_tokens > 0
        ):
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

        # Power metrics from trace
        power_metrics = PowerMetrics(
            gpu=PowerComponentMetrics(
                per_query_watts=MetricStats(avg=trace.avg_gpu_power_watts),
            ),
            cpu=PowerComponentMetrics(
                per_query_watts=MetricStats(avg=trace.avg_cpu_power_watts),
            ),
        )

        # Derived efficiency
        throughput_per_watt = None
        avg_gpu_power = trace.avg_gpu_power_watts
        if throughput is not None and avg_gpu_power is not None and avg_gpu_power > 0:
            throughput_per_watt = throughput / avg_gpu_power

        model_metrics = ModelMetrics(
            compute_metrics=ComputeMetrics(),
            energy_metrics=energy_metrics,
            latency_metrics=latency_metrics,
            memory_metrics=MemoryMetrics(),
            power_metrics=power_metrics,
            temperature_metrics=MetricStats(),
            token_metrics=TokenMetrics(
                input=total_input_tokens,
                output=total_output_tokens,
                total=total_tokens,
            ),
            efficiency=DerivedEfficiencyMetrics(
                throughput_per_watt=throughput_per_watt,
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
