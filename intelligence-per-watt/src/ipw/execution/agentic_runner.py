"""Agentic runner for multi-turn agent benchmarking with energy telemetry."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import logging
import math
import os
import re
import signal
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from tqdm.auto import tqdm

from ..agents.base import BaseAgent, ToolUsingAgent
from ..core.types import DatasetRecord
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
from ..telemetry.energy_attribution import EnergyAttribution
from ..telemetry.events import EventRecorder, EventType
from .executor import Executor
from .preflight import run_preflight

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
    start_time: float | None = None,
    end_time: float | None = None,
) -> float | None:
    """Compute energy delta from first to last reading for *field*."""
    points = sorted(
        (
            (s.timestamp, getattr(s.reading, field))
            for s in readings
            if getattr(s.reading, field, None) is not None
            and math.isfinite(getattr(s.reading, field))
        ),
        key=lambda item: item[0],
    )
    if len(points) < 2:
        return None

    if start_time is not None and end_time is not None and end_time >= start_time:
        if start_time > points[-1][0] or end_time < points[0][0]:
            delta = points[-1][1] - points[0][1]
            return delta if delta >= 0 else None
        start_value = _interpolate_cumulative_value(points, start_time)
        end_value = _interpolate_cumulative_value(points, end_time)
        if start_value is None or end_value is None:
            return None
        delta = end_value - start_value
        return delta if delta >= 0 else None

    delta = points[-1][1] - points[0][1]
    if delta >= 0:
        return delta
    return None


def _interpolate_cumulative_value(
    points: list[tuple[float, float]],
    timestamp: float,
) -> float | None:
    """Interpolate a cumulative telemetry counter at an exact timestamp."""
    if not points:
        return None
    if timestamp <= points[0][0]:
        return points[0][1]
    if timestamp >= points[-1][0]:
        return points[-1][1]

    previous_time, previous_value = points[0]
    for current_time, current_value in points[1:]:
        if current_time < timestamp:
            previous_time, previous_value = current_time, current_value
            continue
        span = current_time - previous_time
        if span <= 0:
            return current_value
        fraction = (timestamp - previous_time) / span
        return previous_value + ((current_value - previous_value) * fraction)
    return points[-1][1]


def _collect_telemetry_window(
    telemetry: TelemetrySession,
    start_time: float,
    end_time: float,
) -> list[TelemetrySample]:
    """Collect samples inside a window plus nearest boundary samples.

    The extra boundary samples let cumulative-energy deltas be interpolated at
    the exact prompt/judge start and end times without resetting the underlying
    telemetry counter.
    """
    samples = sorted(list(telemetry.readings()), key=lambda sample: sample.timestamp)
    if not samples:
        return list(telemetry.window(start_time, end_time))

    before = None
    inside: list[TelemetrySample] = []
    after = None

    for sample in samples:
        if sample.timestamp < start_time:
            before = sample
        elif sample.timestamp <= end_time:
            inside.append(sample)
        else:
            after = sample
            break

    selected: list[TelemetrySample] = []
    for sample in [before, *inside, after]:
        if sample is None:
            continue
        if selected and selected[-1].timestamp == sample.timestamp:
            continue
        selected.append(sample)
    return selected


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


def _workspace_git_diff(workspace: Path) -> str:
    if not (workspace / ".git").exists():
        return ""
    try:
        result = subprocess.run(
            ["git", "diff", "--binary"],
            cwd=str(workspace),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
    except Exception:
        return ""
    return result.stdout if result.returncode == 0 else ""


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
        agent_factory: Optional[Callable[..., BaseAgent]] = None,
        query_timeout: Optional[float] = None,
        max_attempts: int = 3,
        max_turns: int = 10,
        require_dedicated_hardware: bool = False,
    ) -> None:
        """Initialise the agentic runner.

        ``max_attempts`` is the **attempt count** passed directly to
        ``Executor(max_attempts_per_turn=max_attempts)``.  A value of 1 means
        no retry; 3 (the default) allows two retries after the first attempt.
        A caller using *retries* semantics (extra attempts beyond the first)
        should convert ``retries + 1`` to the attempt count before passing it
        here.
        """
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
        self._max_attempts = max_attempts
        self._max_turns = max_turns
        self._require_dedicated_hardware = require_dedicated_hardware
        self._results_lock = threading.Lock()
        self._preflight_done: bool = False
        self._preflight_baseline_dirty: bool = False
        self._energy_attribution: Optional[EnergyAttribution] = None
        self._attach_energy_attribution()

    def _attach_energy_attribution(self) -> None:
        """Attach the shadow EnergyAttribution subscriber to the recorder bus.

        The subscriber observes TOOL_CALL_*/LM_INFERENCE_* events as they flow
        through the bus and emits ENERGY_ATTRIBUTED events for downstream
        consumers to read.
        """
        if self._telemetry is None:
            return
        if self._energy_attribution is not None:
            return
        self._energy_attribution = EnergyAttribution(
            bus=self._event_recorder.bus,
            session=self._telemetry,
            is_cloud_fn=lambda evt: bool(evt.payload.get("is_cloud", False)),
            shared_device_warning_fn=lambda: self._preflight_baseline_dirty,
        )

    def _run_preflight_if_needed(self, *, strict: bool = False) -> None:
        """Run hardware preflight once per runner lifetime.

        Whole-device energy measurement inflates per-query attribution proportionally
        to other GPU/CPU workloads. Preflight samples baseline utilization once;
        the resulting `_preflight_baseline_dirty` flag propagates to subsequent
        TurnTrace/QueryTrace records as `shared_device_warning`.
        """
        if self._preflight_done:
            return
        result = run_preflight(strict=strict)
        self._preflight_done = True
        self._preflight_baseline_dirty = result.shared_device_baseline_dirty
        if self._preflight_baseline_dirty:
            for msg in result.warnings:
                LOGGER.warning("Preflight: %s", msg)

    async def run(
        self,
        max_queries: Optional[int] = None,
        *,
        start_index: int = 0,
    ) -> list[QueryTrace]:
        """Run the agent over the dataset, collecting traces and telemetry.

        Args:
            max_queries: Maximum number of records to process from start_index.
                None means all.
            start_index: First dataset record index to process.

        Returns:
            List of QueryTrace objects with energy-correlated telemetry.
        """
        start_index = max(0, int(start_index))
        end_index = (
            start_index + max(0, int(max_queries))
            if max_queries is not None
            else self._dataset.size()
        )
        self._run_preflight_if_needed(strict=self._require_dedicated_hardware)
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
                recorder = EventRecorder()
                agent, recorder = self._create_concurrent_agent(recorder)

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

    def _create_concurrent_agent(
        self,
        recorder: EventRecorder,
    ) -> tuple[BaseAgent, EventRecorder]:
        """Create a per-task agent and return the recorder it will write to."""
        if self._agent_factory is not None:
            agent = self._call_agent_factory(recorder)
        else:
            agent = copy.deepcopy(self._agent)
            if hasattr(agent, "event_recorder"):
                agent.event_recorder = recorder
            return agent, recorder

        agent_recorder = getattr(agent, "event_recorder", None)
        if isinstance(agent_recorder, EventRecorder):
            return agent, agent_recorder
        if hasattr(agent, "event_recorder"):
            agent.event_recorder = recorder
        return agent, recorder

    def _call_agent_factory(self, recorder: EventRecorder) -> BaseAgent:
        assert self._agent_factory is not None
        try:
            signature = inspect.signature(self._agent_factory)
        except (TypeError, ValueError):
            return self._agent_factory(recorder)

        accepts_positional = any(
            param.kind
            in (
                param.POSITIONAL_ONLY,
                param.POSITIONAL_OR_KEYWORD,
                param.VAR_POSITIONAL,
            )
            for param in signature.parameters.values()
        )
        accepts_event_recorder = (
            "event_recorder" in signature.parameters
            or any(
                param.kind == param.VAR_KEYWORD
                for param in signature.parameters.values()
            )
        )
        if accepts_event_recorder and not accepts_positional:
            return self._agent_factory(event_recorder=recorder)
        if accepts_positional:
            return self._agent_factory(recorder)
        return self._agent_factory()

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
        readings = (
            _collect_telemetry_window(self._telemetry, start_time, end_time)
            if self._telemetry
            else []
        )
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

        query_gpu_energy = _compute_energy_delta(
            readings, "energy_joules", start_time, end_time
        )
        query_cpu_energy = _compute_energy_delta(
            readings, "cpu_energy_joules", start_time, end_time
        )
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

    async def _run_with_executor(
        self,
        index: int,
        record: "DatasetRecord",
        model: str,
        agent: "ToolUsingAgent",
        event_recorder: "EventRecorder",
    ) -> "QueryTrace":
        """Native-agent dispatch path — delegates the turn loop to Executor.

        Seeds the task via agent.set_task() (subclasses must define it),
        runs Executor.execute(), and constructs a QueryTrace from the result.
        Per-turn telemetry population is left to EventBus subscribers
        (EnergyAttribution runs here; richer per-turn TurnTrace population can be
        layered on later).

        A TemporaryDirectory is created per query and set on every agent tool via
        set_workspace() before executor.execute() runs. This keeps agent-driven
        side effects isolated from the project root and from other concurrent
        queries. The tempdir is cleaned up automatically on context exit; tools
        are cleared (set_workspace(None)) in a finally block to prevent stale
        path leakage across query reuse.
        """
        import tempfile

        assert isinstance(agent, ToolUsingAgent)

        query_id = f"q{index:04d}"
        workload_type = record.dataset_metadata.get("workload_type", "agentic")

        event_recorder.clear()
        task_text = record.problem
        if hasattr(agent, "set_task"):
            agent.set_task(task_text)

        # Per-query temp workspace: every agent-driven tool invocation defaults
        # its cwd here, keeping side effects out of the project root and the
        # runner's worktree. Cleaned up automatically on context exit.
        with tempfile.TemporaryDirectory(prefix=f"ipw_q{index:04d}_") as workspace:
            for tool in getattr(agent, "tools", []):
                if hasattr(tool, "set_workspace"):
                    tool.set_workspace(workspace)

            try:
                executor = Executor(
                    bus=event_recorder.bus,
                    max_attempts_per_turn=self._max_attempts,
                )
                result = await executor.execute(
                    agent, task_id=query_id,
                    max_turns=self._max_turns,
                    agent_name=agent.__class__.__name__,
                )
            finally:
                # Clear workspace so tool reuse across queries doesn't leak
                # the now-deleted tempdir path.
                for tool in getattr(agent, "tools", []):
                    if hasattr(tool, "set_workspace"):
                        tool.set_workspace(None)

        return QueryTrace(
            query_id=query_id, workload_type=workload_type,
            query_text=task_text, response_text=result.final_answer or "",
            completed=result.status == "success",
            timed_out=False, n_retries=result.n_retries,
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
        event_recorder = (
            event_recorder if event_recorder is not None else self._event_recorder
        )

        if isinstance(agent, ToolUsingAgent):
            return await self._run_with_executor(index, record, model, agent, event_recorder)

        query_id = f"q{index:04d}"
        workload_type = record.dataset_metadata.get("workload_type", "agentic")

        # Capture prompt and judge/evaluation telemetry in separate windows.
        # The telemetry source can remain cumulative; each field below is a
        # delta over its own wall-clock interval.
        start_time = time.time()
        prompt_end_time: float | None = None
        judge_start_time: float | None = None
        judge_end_time: float | None = None
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

                run_async = getattr(agent, "run_async", None)
                if inspect.iscoroutinefunction(run_async):
                    result = await run_async(record.problem)
                else:
                    result = agent.run(record.problem)
                agent_metadata = dict(result.metadata or {})
                if agent_metadata:
                    record.dataset_metadata["agent_metadata"] = agent_metadata
                    record.dataset_metadata.setdefault("agent_result_metadata", agent_metadata)
                    token_source = agent_metadata.get("token_source")
                    if token_source is not None:
                        record.dataset_metadata["token_source"] = token_source
                    for key in (
                        "gdpval_outputs_dir",
                        "gdpval_submitted_files",
                    ):
                        if key in agent_metadata:
                            record.dataset_metadata[key] = agent_metadata[key]

                prompt_end_time = time.time()

                if task_env is not None:
                    judge_start_time = time.time()
                    try:
                        task_env.run_tests()
                        if (
                            "is_resolved" in record.dataset_metadata
                            and hasattr(self._dataset, "score")
                        ):
                            try:
                                is_correct, score_metadata = self._dataset.score(
                                    record, result.content
                                )
                                score_metadata = score_metadata or {}
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
                    finally:
                        judge_end_time = time.time()
                elif hasattr(self._dataset, "score") and record.answer:
                    judge_start_time = time.time()
                    try:
                        is_correct, score_metadata = self._dataset.score(record, result.content)
                        score_metadata = score_metadata or {}
                        record.dataset_metadata["is_resolved"] = is_correct
                        if score_metadata:
                            record.dataset_metadata["score_metadata"] = score_metadata
                            record.dataset_metadata["evaluation_metadata"] = score_metadata
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
                    finally:
                        judge_end_time = time.time()
        except Exception as exc:
            LOGGER.warning("Agent failed on query %s: %s", query_id, exc)
            failure_time = time.time()
            prompt_end_time = prompt_end_time or failure_time
            prompt_readings = []
            judge_readings = []
            if self._telemetry:
                prompt_readings = _collect_telemetry_window(
                    self._telemetry,
                    start_time,
                    prompt_end_time,
                )
                if judge_start_time is not None:
                    judge_readings = _collect_telemetry_window(
                        self._telemetry,
                        judge_start_time,
                        judge_end_time or failure_time,
                    )
            trace = QueryTrace(
                query_id=query_id,
                workload_type=str(workload_type),
                query_text=record.problem,
                response_text=str(exc),
                total_wall_clock_s=prompt_end_time - start_time,
                completed=False,
                query_gpu_energy_joules=_compute_energy_delta(
                    prompt_readings,
                    "energy_joules",
                    start_time,
                    prompt_end_time,
                ),
                query_cpu_energy_joules=_compute_energy_delta(
                    prompt_readings,
                    "cpu_energy_joules",
                    start_time,
                    prompt_end_time,
                ),
                query_gpu_power_avg_watts=_compute_power_avg(prompt_readings, "power_watts"),
                query_cpu_power_avg_watts=_compute_power_avg(prompt_readings, "cpu_power_watts"),
                judge_gpu_energy_joules=_compute_energy_delta(
                    judge_readings,
                    "energy_joules",
                    judge_start_time,
                    judge_end_time or failure_time,
                ) if judge_start_time is not None else None,
                judge_cpu_energy_joules=_compute_energy_delta(
                    judge_readings,
                    "cpu_energy_joules",
                    judge_start_time,
                    judge_end_time or failure_time,
                ) if judge_start_time is not None else None,
                judge_gpu_power_avg_watts=_compute_power_avg(judge_readings, "power_watts"),
                judge_cpu_power_avg_watts=_compute_power_avg(judge_readings, "cpu_power_watts"),
                judge_wall_clock_s=(
                    (judge_end_time or failure_time) - judge_start_time
                    if judge_start_time is not None
                    else None
                ),
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

        prompt_end_time = prompt_end_time or time.time()

        # Collect telemetry samples for the prompt-only query window. Judge
        # telemetry is kept separate below.
        readings: list[TelemetrySample] = []
        judge_readings: list[TelemetrySample] = []
        if self._telemetry:
            readings = _collect_telemetry_window(
                self._telemetry,
                start_time,
                prompt_end_time,
            )
            if judge_start_time is not None and judge_end_time is not None:
                judge_readings = _collect_telemetry_window(
                    self._telemetry,
                    judge_start_time,
                    judge_end_time,
                )

        # Build turn traces from event recorder
        events = event_recorder.get_events()
        if result.trace is not None and result.trace.turns:
            turns = result.trace.turns
            for turn_index, turn in enumerate(turns):
                turn.turn_index = turn_index
        else:
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
                wall_clock_s=prompt_end_time - start_time,
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
                turns[0].wall_clock_s = turns[0].wall_clock_s or (prompt_end_time - start_time)
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

        # Always compute prompt-only query energy from the prompt window.
        query_gpu_energy = _compute_energy_delta(
            readings,
            "energy_joules",
            start_time,
            prompt_end_time,
        )
        query_cpu_energy = _compute_energy_delta(
            readings,
            "cpu_energy_joules",
            start_time,
            prompt_end_time,
        )
        query_gpu_power_avg = _compute_power_avg(readings, "power_watts")
        query_cpu_power_avg = _compute_power_avg(readings, "cpu_power_watts")
        judge_gpu_energy = _compute_energy_delta(
            judge_readings,
            "energy_joules",
            judge_start_time,
            judge_end_time,
        ) if judge_start_time is not None and judge_end_time is not None else None
        judge_cpu_energy = _compute_energy_delta(
            judge_readings,
            "cpu_energy_joules",
            judge_start_time,
            judge_end_time,
        ) if judge_start_time is not None and judge_end_time is not None else None
        judge_gpu_power_avg = _compute_power_avg(judge_readings, "power_watts")
        judge_cpu_power_avg = _compute_power_avg(judge_readings, "cpu_power_watts")

        # Fallback: estimate energy from average power when cumulative counters
        # have fewer than 2 samples (no delta possible).
        duration = prompt_end_time - start_time
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
            total_wall_clock_s=prompt_end_time - start_time,
            completed=True,
            query_gpu_energy_joules=query_gpu_energy,
            query_cpu_energy_joules=query_cpu_energy,
            query_gpu_power_avg_watts=query_gpu_power_avg,
            query_cpu_power_avg_watts=query_cpu_power_avg,
            judge_gpu_energy_joules=judge_gpu_energy,
            judge_cpu_energy_joules=judge_cpu_energy,
            judge_gpu_power_avg_watts=judge_gpu_power_avg,
            judge_cpu_power_avg_watts=judge_cpu_power_avg,
            judge_wall_clock_s=(
                judge_end_time - judge_start_time
                if judge_start_time is not None and judge_end_time is not None
                else None
            ),
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

        total_wall = sum(max(0.0, t.wall_clock_s or 0.0) for t in trace.turns)

        # Compute total query energy.
        gpu_energies = [
            s.reading.energy_joules for s in readings
            if s.reading.energy_joules is not None
            and math.isfinite(s.reading.energy_joules)
        ]
        total_gpu_energy = trace.query_gpu_energy_joules
        if len(gpu_energies) >= 2:
            delta = gpu_energies[-1] - gpu_energies[0]
            total_gpu_energy = total_gpu_energy if total_gpu_energy is not None else (
                delta if delta >= 0 else None
            )
        total_gpu_power = trace.query_gpu_power_avg_watts or _compute_power_avg(
            readings,
            "power_watts",
        )

        cpu_energies = [
            s.reading.cpu_energy_joules for s in readings
            if s.reading.cpu_energy_joules is not None
            and math.isfinite(s.reading.cpu_energy_joules)
        ]
        total_cpu_energy = trace.query_cpu_energy_joules
        if len(cpu_energies) >= 2:
            delta = cpu_energies[-1] - cpu_energies[0]
            total_cpu_energy = total_cpu_energy if total_cpu_energy is not None else (
                delta if delta >= 0 else None
            )
        total_cpu_power = trace.query_cpu_power_avg_watts or _compute_power_avg(
            readings,
            "cpu_power_watts",
        )

        # Fallback: estimate from power when cumulative counters unavailable
        if total_gpu_energy is None and total_wall > 0:
            total_gpu_energy = _estimate_energy_from_power(readings, "power_watts", total_wall)
        if total_cpu_energy is None and total_wall > 0:
            total_cpu_energy = _estimate_energy_from_power(readings, "cpu_power_watts", total_wall)

        def _fill_missing(
            *,
            energy_attr: str,
            power_attr: str,
            total_energy: float | None,
            total_power: float | None,
        ) -> None:
            missing = [
                turn
                for turn in trace.turns
                if getattr(turn, energy_attr) is None
            ]
            if not missing:
                for turn in trace.turns:
                    if getattr(turn, power_attr) is None:
                        energy = getattr(turn, energy_attr)
                        wall = max(0.0, turn.wall_clock_s or 0.0)
                        if energy is not None and wall > 0:
                            setattr(turn, power_attr, energy / wall)
                        elif total_power is not None:
                            setattr(turn, power_attr, total_power)
                return

            known_energy = sum(
                getattr(turn, energy_attr) or 0.0
                for turn in trace.turns
                if getattr(turn, energy_attr) is not None
            )
            remaining_energy = (
                max(0.0, total_energy - known_energy)
                if total_energy is not None
                else None
            )
            missing_wall = sum(max(0.0, turn.wall_clock_s or 0.0) for turn in missing)

            for turn in missing:
                wall = max(0.0, turn.wall_clock_s or 0.0)
                if remaining_energy is not None:
                    if missing_wall > 0:
                        energy = remaining_energy * (wall / missing_wall)
                    else:
                        energy = remaining_energy / len(missing)
                elif total_power is not None and wall > 0:
                    energy = total_power * wall
                else:
                    energy = None

                if energy is not None:
                    setattr(turn, energy_attr, energy)
                    setattr(
                        turn,
                        power_attr,
                        (energy / wall) if wall > 0 else total_power,
                    )
                elif total_power is not None and getattr(turn, power_attr) is None:
                    setattr(turn, power_attr, total_power)

            for turn in trace.turns:
                if getattr(turn, power_attr) is None:
                    energy = getattr(turn, energy_attr)
                    wall = max(0.0, turn.wall_clock_s or 0.0)
                    if energy is not None and wall > 0:
                        setattr(turn, power_attr, energy / wall)
                    elif total_power is not None:
                        setattr(turn, power_attr, total_power)

        _fill_missing(
            energy_attr="gpu_energy_joules",
            power_attr="gpu_power_avg_watts",
            total_energy=total_gpu_energy,
            total_power=total_gpu_power,
        )
        _fill_missing(
            energy_attr="cpu_energy_joules",
            power_attr="cpu_power_avg_watts",
            total_energy=total_cpu_energy,
            total_power=total_cpu_power,
        )

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
            "query_gpu_energy_joules": trace.query_gpu_energy_joules,
            "judge_gpu_energy_joules": trace.judge_gpu_energy_joules,
            "judge_wall_clock_s": trace.judge_wall_clock_s,
            "total_task_gpu_energy_joules": trace.total_task_gpu_energy_joules,
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
            "gdpval_outputs_dir",
            "gdpval_submitted_files",
            "evaluation_metadata",
        ):
            val = record.dataset_metadata.get(key)
            if val is not None:
                meta[key] = val

        dataset_id = str(getattr(self._dataset, "dataset_id", "") or "")
        if dataset_id in {"swebench", "swefficiency"}:
            workspace_raw = record.dataset_metadata.get("workspace_path")
            workspace = (
                Path(str(workspace_raw))
                if workspace_raw
                else query_dir / "workspace"
            )
            workspace_diff = _workspace_git_diff(workspace)
            (query_dir / "workspace.diff").write_text(
                workspace_diff,
                encoding="utf-8",
            )
            meta["workspace_path"] = str(workspace)
            meta["workspace_git_available"] = (workspace / ".git").exists()
            meta["workspace_diff_bytes"] = len(workspace_diff.encode("utf-8"))

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
            judge_gpu_joules=trace.judge_gpu_energy_joules,
            judge_cpu_joules=trace.judge_cpu_energy_joules,
            total_task_gpu_joules=trace.total_task_gpu_energy_joules,
            total_task_cpu_joules=trace.total_task_cpu_energy_joules,
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

        # Power metrics from trace. Prefer the query-level telemetry average
        # when available; per-turn averages may be attribution-derived.
        avg_gpu_power = trace.query_gpu_power_avg_watts
        if avg_gpu_power is None:
            avg_gpu_power = trace.avg_gpu_power_watts
        avg_cpu_power = trace.query_cpu_power_avg_watts
        if avg_cpu_power is None:
            avg_cpu_power = trace.avg_cpu_power_watts
        power_metrics = PowerMetrics(
            gpu=PowerComponentMetrics(
                per_query_watts=MetricStats(avg=avg_gpu_power),
            ),
            cpu=PowerComponentMetrics(
                per_query_watts=MetricStats(avg=avg_cpu_power),
            ),
        )

        # Derived efficiency
        throughput_per_watt = None
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
