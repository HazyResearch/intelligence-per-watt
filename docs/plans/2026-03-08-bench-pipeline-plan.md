# ipw bench Pipeline Completion — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the `ipw bench` pipeline so that a single command handles server spin-up, telemetry, benchmark execution, accuracy evaluation, MBU aggregation, normalized statistics, and results export with IPJ/IPW.

**Architecture:** Extend the existing `ipw bench` pipeline by adding accuracy scoring inline after trace collection, extracting MBU from telemetry samples already being captured, computing IPJ/IPW/normalized stats, and enriching both terminal output and `summary.json`. Then update documentation.

**Tech Stack:** Python (Click CLI, Rich, dataclasses), existing Rust energy monitor (unchanged)

---

### Task 1: Add MBU Fields to QueryTrace

**Files:**
- Modify: `intelligence-per-watt/src/ipw/execution/trace.py:66-193`
- Test: `intelligence-per-watt/src/ipw/tests/core/test_trace.py`

**Step 1: Write the failing tests**

Add to `intelligence-per-watt/src/ipw/tests/core/test_trace.py`:

```python
def test_mbu_fields_default_none(self) -> None:
    trace = QueryTrace(query_id="q0", workload_type="test")
    assert trace.query_mbu_avg_pct is None
    assert trace.query_mbu_max_pct is None

def test_mbu_fields_in_to_dict(self) -> None:
    trace = QueryTrace(
        query_id="q0",
        workload_type="test",
        query_mbu_avg_pct=65.3,
        query_mbu_max_pct=88.1,
    )
    d = trace.to_dict()
    assert d["query_mbu_avg_pct"] == 65.3
    assert d["query_mbu_max_pct"] == 88.1

def test_mbu_fields_roundtrip(self) -> None:
    trace = QueryTrace(
        query_id="q0",
        workload_type="test",
        query_mbu_avg_pct=65.3,
        query_mbu_max_pct=88.1,
    )
    restored = QueryTrace.from_dict(trace.to_dict())
    assert restored.query_mbu_avg_pct == 65.3
    assert restored.query_mbu_max_pct == 88.1
```

**Step 2: Run tests to verify they fail**

Run: `cd intelligence-per-watt && python -m pytest src/ipw/tests/core/test_trace.py -v -k "mbu" --no-header`
Expected: FAIL — `TypeError: QueryTrace.__init__() got an unexpected keyword argument 'query_mbu_avg_pct'`

**Step 3: Write minimal implementation**

Add two fields to the `QueryTrace` dataclass after `query_cpu_power_avg_watts` (line 82):

```python
query_mbu_avg_pct: Optional[float] = None
query_mbu_max_pct: Optional[float] = None
```

Add to `to_dict()` (after the `query_cpu_power_avg_watts` line):

```python
"query_mbu_avg_pct": self.query_mbu_avg_pct,
"query_mbu_max_pct": self.query_mbu_max_pct,
```

Add to `from_dict()` kwargs:

```python
query_mbu_avg_pct=d.get("query_mbu_avg_pct"),
query_mbu_max_pct=d.get("query_mbu_max_pct"),
```

Add to `to_hf_dataset()` row dict:

```python
"query_mbu_avg_pct": trace.query_mbu_avg_pct,
"query_mbu_max_pct": trace.query_mbu_max_pct,
```

**Step 4: Run tests to verify they pass**

Run: `cd intelligence-per-watt && python -m pytest src/ipw/tests/core/test_trace.py -v --no-header`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add intelligence-per-watt/src/ipw/execution/trace.py intelligence-per-watt/src/ipw/tests/core/test_trace.py
git commit -m "feat: add MBU fields to QueryTrace"
```

---

### Task 2: Extract MBU from Telemetry in AgenticRunner

**Files:**
- Modify: `intelligence-per-watt/src/ipw/execution/agentic_runner.py:466-494`
- Test: `intelligence-per-watt/src/ipw/tests/execution/test_agentic_runner.py`

**Step 1: Write the failing test**

Add to `intelligence-per-watt/src/ipw/tests/execution/test_agentic_runner.py`. This test needs a helper to create mock telemetry samples with MBU data. Check the existing test file for how `TelemetrySample` is mocked — follow the same pattern. The test should verify that when telemetry samples have `gpu_memory_bandwidth_utilization_pct` values, the resulting `QueryTrace` has `query_mbu_avg_pct` and `query_mbu_max_pct` populated.

```python
def test_mbu_extracted_from_telemetry(self):
    """MBU fields on QueryTrace are populated from telemetry samples."""
    # Build mock telemetry samples with MBU data
    # Run a single query through the runner
    # Assert trace.query_mbu_avg_pct and trace.query_mbu_max_pct are set
```

Note: Adapt this to match the existing mock/fixture patterns in the test file.

**Step 2: Run test to verify it fails**

Run: `cd intelligence-per-watt && python -m pytest src/ipw/tests/execution/test_agentic_runner.py -v -k "mbu" --no-header`
Expected: FAIL

**Step 3: Write minimal implementation**

In `agentic_runner.py`, in `_run_single_query()`, after the existing energy computation block (around line 480), add MBU extraction:

```python
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
```

Then pass these to the `QueryTrace` constructor (around line 482-494):

```python
query_mbu_avg_pct=query_mbu_avg,
query_mbu_max_pct=query_mbu_max,
```

**Step 4: Run tests to verify they pass**

Run: `cd intelligence-per-watt && python -m pytest src/ipw/tests/execution/test_agentic_runner.py -v --no-header`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add intelligence-per-watt/src/ipw/execution/agentic_runner.py intelligence-per-watt/src/ipw/tests/execution/test_agentic_runner.py
git commit -m "feat: extract MBU from telemetry samples in AgenticRunner"
```

---

### Task 3: Add Efficiency and Normalized Stats to `export_summary_json`

**Files:**
- Modify: `intelligence-per-watt/src/ipw/execution/exporters.py:61-170`
- Test: `intelligence-per-watt/src/ipw/tests/execution/test_exporters.py`

**Step 1: Write the failing tests**

Add to `intelligence-per-watt/src/ipw/tests/execution/test_exporters.py`:

```python
def _make_traces_with_accuracy() -> list[QueryTrace]:
    """Traces where some queries are resolved and have energy data."""
    return [
        QueryTrace(
            query_id="q0001",
            workload_type="agentic",
            query_text="What is 2+2?",
            response_text="4",
            turns=[
                TurnTrace(
                    turn_index=0,
                    input_tokens=100,
                    output_tokens=50,
                    wall_clock_s=1.0,
                    gpu_energy_joules=5.0,
                    gpu_power_avg_watts=200.0,
                ),
            ],
            total_wall_clock_s=1.0,
            completed=True,
            is_resolved=True,
            query_gpu_energy_joules=5.0,
            query_gpu_power_avg_watts=200.0,
            query_mbu_avg_pct=65.0,
            query_mbu_max_pct=80.0,
        ),
        QueryTrace(
            query_id="q0002",
            workload_type="agentic",
            query_text="Capital of France?",
            response_text="London",
            turns=[
                TurnTrace(
                    turn_index=0,
                    input_tokens=20,
                    output_tokens=5,
                    wall_clock_s=0.5,
                    gpu_energy_joules=2.0,
                    gpu_power_avg_watts=180.0,
                ),
            ],
            total_wall_clock_s=0.5,
            completed=True,
            is_resolved=False,
            query_gpu_energy_joules=2.0,
            query_gpu_power_avg_watts=180.0,
            query_mbu_avg_pct=55.0,
            query_mbu_max_pct=70.0,
        ),
    ]


class TestExportSummaryJsonEfficiency:
    """Test efficiency and normalized statistics in summary JSON."""

    def test_efficiency_section_present(self, tmp_path: Path) -> None:
        traces = _make_traces_with_accuracy()
        path = tmp_path / "summary.json"
        export_summary_json(traces, {"agent": "react"}, path)
        summary = json.loads(path.read_text())
        assert "efficiency" in summary
        eff = summary["efficiency"]
        assert eff["accuracy"] == pytest.approx(0.5)
        assert eff["total_gpu_energy_joules"] == pytest.approx(7.0)
        assert eff["ipj"] is not None
        assert eff["ipw"] is not None

    def test_totals_has_accuracy(self, tmp_path: Path) -> None:
        traces = _make_traces_with_accuracy()
        path = tmp_path / "summary.json"
        export_summary_json(traces, {}, path)
        summary = json.loads(path.read_text())
        assert summary["totals"]["accuracy"] == pytest.approx(0.5)

    def test_mbu_in_statistics(self, tmp_path: Path) -> None:
        traces = _make_traces_with_accuracy()
        path = tmp_path / "summary.json"
        export_summary_json(traces, {}, path)
        summary = json.loads(path.read_text())
        assert "mbu_avg_pct" in summary["statistics"]
        assert summary["statistics"]["mbu_avg_pct"]["avg"] == pytest.approx(60.0)

    def test_normalized_statistics_present(self, tmp_path: Path) -> None:
        # Need > 20 traces for 5% trimming to be meaningful
        traces = []
        for i in range(40):
            traces.append(QueryTrace(
                query_id=f"q{i:04d}",
                workload_type="agentic",
                turns=[TurnTrace(turn_index=0, input_tokens=10, output_tokens=5,
                                 wall_clock_s=float(i + 1), gpu_energy_joules=float(i + 1))],
                total_wall_clock_s=float(i + 1),
                completed=True,
                is_resolved=(i % 2 == 0),
                query_gpu_energy_joules=float(i + 1),
                query_gpu_power_avg_watts=100.0,
            ))
        path = tmp_path / "summary.json"
        export_summary_json(traces, {}, path)
        summary = json.loads(path.read_text())
        assert "normalized_statistics" in summary
        assert "normalized_efficiency" in summary
        ns = summary["normalized_statistics"]
        assert ns["_outliers_removed"]["top_pct"] == 5
        assert ns["_outliers_removed"]["bottom_pct"] == 5
        assert ns["_outliers_removed"]["queries_before"] == 40
        assert ns["_outliers_removed"]["queries_after"] == 36  # 40 - 2*2
        # Trimmed stats should exclude the extremes
        assert ns["wall_clock_s"]["min"] > 1.0  # bottom 5% removed
        assert ns["wall_clock_s"]["max"] < 40.0  # top 5% removed

    def test_normalized_efficiency_present(self, tmp_path: Path) -> None:
        traces = []
        for i in range(40):
            traces.append(QueryTrace(
                query_id=f"q{i:04d}",
                workload_type="agentic",
                turns=[TurnTrace(turn_index=0, input_tokens=10, output_tokens=5,
                                 wall_clock_s=float(i + 1), gpu_energy_joules=float(i + 1))],
                total_wall_clock_s=float(i + 1),
                completed=True,
                is_resolved=(i % 2 == 0),
                query_gpu_energy_joules=float(i + 1),
                query_gpu_power_avg_watts=100.0,
            ))
        path = tmp_path / "summary.json"
        export_summary_json(traces, {}, path)
        summary = json.loads(path.read_text())
        ne = summary["normalized_efficiency"]
        assert ne["accuracy"] is not None
        assert ne["ipj"] is not None

    def test_efficiency_with_no_resolved(self, tmp_path: Path) -> None:
        traces = [
            QueryTrace(
                query_id="q0",
                workload_type="agentic",
                turns=[TurnTrace(turn_index=0, input_tokens=10, output_tokens=5, wall_clock_s=1.0)],
                total_wall_clock_s=1.0,
                completed=True,
            ),
        ]
        path = tmp_path / "summary.json"
        export_summary_json(traces, {}, path)
        summary = json.loads(path.read_text())
        eff = summary["efficiency"]
        # No is_resolved set → accuracy should be None
        assert eff["accuracy"] is None
        assert eff["ipj"] is None
        assert eff["ipw"] is None
```

**Step 2: Run tests to verify they fail**

Run: `cd intelligence-per-watt && python -m pytest src/ipw/tests/execution/test_exporters.py -v -k "Efficiency or normalized or mbu" --no-header`
Expected: FAIL — no `efficiency` key in summary

**Step 3: Write minimal implementation**

In `exporters.py`, modify `export_summary_json()`. After the existing `stats` dict (line 140), add:

```python
# MBU stats
stats["mbu_avg_pct"] = _agg_stats([t.query_mbu_avg_pct for t in traces])

# Accuracy and efficiency
resolved_count = sum(1 for t in traces if t.is_resolved is True)
unresolved_count = sum(1 for t in traces if t.is_resolved is False)
scored_count = resolved_count + unresolved_count
accuracy = resolved_count / scored_count if scored_count > 0 else None

# Compute efficiency metrics
avg_gpu_power = stats["gpu_power_watts"]["avg"]
avg_cpu_power = stats["cpu_power_watts"]["avg"]

ipj = None
ipw = None
if accuracy is not None and accuracy > 0:
    if total_gpu_energy is not None and total_gpu_energy > 0:
        ipj = accuracy / total_gpu_energy
    if avg_gpu_power is not None and avg_gpu_power > 0:
        ipw = accuracy / avg_gpu_power

efficiency = {
    "accuracy": accuracy,
    "total_gpu_energy_joules": total_gpu_energy,
    "total_cpu_energy_joules": total_cpu_energy,
    "avg_gpu_power_watts": avg_gpu_power,
    "avg_cpu_power_watts": avg_cpu_power,
    "ipj": ipj,
    "ipw": ipw,
}

# --- Normalized statistics (trim top/bottom 5% by wall_clock_s) ---
def _compute_normalized(traces_list):
    n = len(traces_list)
    if n < 4:
        # Too few traces to trim meaningfully
        return None, None, n, n
    trim_count = max(1, round(n * 0.05))
    sorted_traces = sorted(traces_list, key=lambda t: t.total_wall_clock_s)
    trimmed = sorted_traces[trim_count : n - trim_count]

    norm_stats = {
        "wall_clock_s": _agg_stats([t.total_wall_clock_s for t in trimmed]),
        "gpu_energy_joules": _agg_stats([t.total_gpu_energy_joules for t in trimmed]),
        "cpu_energy_joules": _agg_stats([t.total_cpu_energy_joules for t in trimmed]),
        "gpu_power_watts": _agg_stats([t.avg_gpu_power_watts for t in trimmed]),
        "cpu_power_watts": _agg_stats([t.avg_cpu_power_watts for t in trimmed]),
        "input_tokens": _agg_stats([float(t.total_input_tokens) for t in trimmed]),
        "output_tokens": _agg_stats([float(t.total_output_tokens) for t in trimmed]),
        "total_tokens": _agg_stats([float(t.total_tokens) for t in trimmed]),
        "throughput_tokens_per_sec": _agg_stats([t.throughput_tokens_per_sec for t in trimmed]),
        "energy_per_token_joules": _agg_stats([t.energy_per_token_joules for t in trimmed]),
        "mbu_avg_pct": _agg_stats([t.query_mbu_avg_pct for t in trimmed]),
        "cost_usd": _agg_stats([t.total_cost_usd for t in trimmed]),
        "turns": _agg_stats([float(t.num_turns) for t in trimmed]),
        "tool_calls": _agg_stats([float(t.total_tool_calls) for t in trimmed]),
    }

    # Normalized efficiency
    n_resolved = sum(1 for t in trimmed if t.is_resolved is True)
    n_unresolved = sum(1 for t in trimmed if t.is_resolved is False)
    n_scored = n_resolved + n_unresolved
    n_accuracy = n_resolved / n_scored if n_scored > 0 else None

    n_gpu_energies = [t.total_gpu_energy_joules for t in trimmed if t.total_gpu_energy_joules is not None]
    n_total_gpu_energy = sum(n_gpu_energies) if n_gpu_energies else None
    n_cpu_energies = [t.total_cpu_energy_joules for t in trimmed if t.total_cpu_energy_joules is not None]
    n_total_cpu_energy = sum(n_cpu_energies) if n_cpu_energies else None
    n_avg_gpu_power = norm_stats["gpu_power_watts"]["avg"]
    n_avg_cpu_power = norm_stats["cpu_power_watts"]["avg"]

    n_ipj = None
    n_ipw = None
    if n_accuracy is not None and n_accuracy > 0:
        if n_total_gpu_energy is not None and n_total_gpu_energy > 0:
            n_ipj = n_accuracy / n_total_gpu_energy
        if n_avg_gpu_power is not None and n_avg_gpu_power > 0:
            n_ipw = n_accuracy / n_avg_gpu_power

    norm_eff = {
        "accuracy": n_accuracy,
        "total_gpu_energy_joules": n_total_gpu_energy,
        "total_cpu_energy_joules": n_total_cpu_energy,
        "avg_gpu_power_watts": n_avg_gpu_power,
        "avg_cpu_power_watts": n_avg_cpu_power,
        "ipj": n_ipj,
        "ipw": n_ipw,
    }

    return norm_stats, norm_eff, n, len(trimmed)

norm_stats_result, norm_eff_result, n_before, n_after = _compute_normalized(traces)

normalized_statistics = None
normalized_efficiency = None
if norm_stats_result is not None:
    normalized_statistics = {
        "_description": "Statistics recomputed after removing the top 5% and bottom 5% of queries by wall_clock_s",
        "_outliers_removed": {"top_pct": 5, "bottom_pct": 5, "queries_before": n_before, "queries_after": n_after},
        **norm_stats_result,
    }
    normalized_efficiency = {
        "_description": "Efficiency recomputed on the trimmed query set",
        "_outliers_removed": {"top_pct": 5, "bottom_pct": 5, "queries_before": n_before, "queries_after": n_after},
        **norm_eff_result,
    }
```

Then update the `summary` dict to include the new sections:

```python
summary = {
    "generated_at": time.time(),
    "config": config,
    "totals": {
        ...existing fields...,
        "accuracy": accuracy,
    },
    "efficiency": efficiency,
    "averages": { ...existing... },
    "statistics": stats,
    "normalized_statistics": normalized_statistics,
    "normalized_efficiency": normalized_efficiency,
}
```

**Step 4: Run tests to verify they pass**

Run: `cd intelligence-per-watt && python -m pytest src/ipw/tests/execution/test_exporters.py -v --no-header`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add intelligence-per-watt/src/ipw/execution/exporters.py intelligence-per-watt/src/ipw/tests/execution/test_exporters.py
git commit -m "feat: add efficiency, MBU, and normalized statistics to summary.json"
```

---

### Task 4: Add MBU Row and Resolved Count to Terminal Display

**Files:**
- Modify: `intelligence-per-watt/src/ipw/cli/_display.py:181-232` and `286-357`
- Test: `intelligence-per-watt/src/ipw/tests/cli/test_display.py`

**Step 1: Write the failing tests**

Add to `intelligence-per-watt/src/ipw/tests/cli/test_display.py`. Check the existing test file for patterns. Tests should verify:

```python
def test_compute_trace_metrics_includes_mbu():
    """MBU row is included when traces have MBU data."""
    traces = [
        QueryTrace(
            query_id="q0",
            workload_type="test",
            turns=[TurnTrace(turn_index=0, input_tokens=10, output_tokens=5, wall_clock_s=1.0)],
            total_wall_clock_s=1.0,
            completed=True,
            query_mbu_avg_pct=65.0,
        ),
    ]
    rows = compute_trace_metrics(traces)
    labels = [r.label for r in rows]
    assert "MBU" in labels


def test_efficiency_panel_shows_resolved_count():
    """Efficiency panel includes resolved count when traces have is_resolved."""
    # This test captures the Rich console output and checks for "Resolved" text.
    from io import StringIO
    from rich.console import Console
    console = Console(file=StringIO(), force_terminal=True, width=120)
    traces = [
        QueryTrace(
            query_id="q0", workload_type="test",
            turns=[TurnTrace(turn_index=0, gpu_energy_joules=10.0)],
            total_wall_clock_s=1.0, completed=True, is_resolved=True,
            query_gpu_energy_joules=10.0, query_gpu_power_avg_watts=100.0,
        ),
        QueryTrace(
            query_id="q1", workload_type="test",
            turns=[TurnTrace(turn_index=0, gpu_energy_joules=5.0)],
            total_wall_clock_s=0.5, completed=True, is_resolved=False,
            query_gpu_energy_joules=5.0, query_gpu_power_avg_watts=80.0,
        ),
    ]
    print_efficiency_panel(console, traces=traces)
    output = console.file.getvalue()
    assert "Resolved" in output
```

**Step 2: Run tests to verify they fail**

Run: `cd intelligence-per-watt && python -m pytest src/ipw/tests/cli/test_display.py -v -k "mbu or resolved" --no-header`
Expected: FAIL

**Step 3: Write minimal implementation**

In `_display.py`, in `compute_trace_metrics()` (around line 230), add the MBU row before the Completed row:

```python
_row("MBU", [t.query_mbu_avg_pct for t in traces], "%"),
```

In `print_efficiency_panel()`, after the accuracy line (around line 332), add resolved count:

```python
if traces:
    resolved = sum(1 for t in traces if t.is_resolved is True)
    total = len(traces)
    scored = sum(1 for t in traces if t.is_resolved is not None)
    if scored > 0:
        acc = resolved / scored
        lines.append(f"Accuracy:      [bold]{acc * 100:.1f}%[/bold]")
        lines.append(f"Resolved:      [bold]{resolved}/{total}[/bold]")
```

This replaces the existing `acc` computation block for traces (lines 324-327).

**Step 4: Run tests to verify they pass**

Run: `cd intelligence-per-watt && python -m pytest src/ipw/tests/cli/test_display.py -v --no-header`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add intelligence-per-watt/src/ipw/cli/_display.py intelligence-per-watt/src/ipw/tests/cli/test_display.py
git commit -m "feat: add MBU row and resolved count to terminal display"
```

---

### Task 5: Integrate Accuracy Scoring in `bench.py`

**Files:**
- Modify: `intelligence-per-watt/src/ipw/cli/bench.py:464-476`

**Context:** The `AgenticRunner._run_single_query()` already calls `self._dataset.score()` at line 414-417 for datasets that support it, setting `record.dataset_metadata["is_resolved"]`. This means `is_resolved` is already being populated on traces during the run for most datasets.

However, the scoring at line 414 only runs when `task_env is None` (i.e., non-TerminalBench datasets). TerminalBench sets `is_resolved` via its own `task_env.run_tests()` path. So **scoring is already integrated** for all datasets.

The gap is that `bench.py` doesn't pass `is_resolved` data through to the efficiency panel and doesn't ensure `summary.json` has the new fields. Since Tasks 3 and 4 handle the export and display side, and the runner already scores, this task is about verification.

**Step 1: Write a test verifying the end-to-end flow**

Add to `intelligence-per-watt/src/ipw/tests/execution/test_agentic_runner.py`:

```python
def test_is_resolved_populated_when_dataset_has_score():
    """AgenticRunner calls dataset.score() and populates is_resolved on traces."""
    # Mock a dataset with a score() method that returns (True, {})
    # Run the runner
    # Assert trace.is_resolved is True
```

Note: Adapt to match existing test patterns in the file.

**Step 2: Run test**

Run: `cd intelligence-per-watt && python -m pytest src/ipw/tests/execution/test_agentic_runner.py -v -k "is_resolved" --no-header`

**Step 3: If test passes, verify no code changes needed in bench.py**

The scoring is already integrated. The only change needed is to confirm `bench.py` doesn't filter out `is_resolved` when passing traces to the display functions. Read the bench.py display code (lines 798-809) — it already passes `traces` directly to `compute_trace_metrics()` and `print_efficiency_panel()`, which we updated in Task 4.

**Step 4: Commit**

```bash
git add intelligence-per-watt/src/ipw/tests/execution/test_agentic_runner.py
git commit -m "test: verify accuracy scoring integration in agentic runner"
```

---

### Task 6: Pass MBU Through in `bench.py` Telemetry Path

**Files:**
- Modify: `intelligence-per-watt/src/ipw/cli/bench.py:185-207`

**Context:** `bench.py` has its own `_compute_energy_metrics()` function that extracts energy/power from telemetry samples. This is used for the aggregate benchmark-level metrics (separate from per-query metrics computed in `AgenticRunner`). We need to also extract MBU here.

**Step 1: Modify `_compute_energy_metrics()` in bench.py**

After the existing power computation (around line 193), add:

```python
mbu_samples = [
    r.gpu_memory_bandwidth_utilization_pct for r in readings
    if getattr(r, 'gpu_memory_bandwidth_utilization_pct', None) is not None
    and r.gpu_memory_bandwidth_utilization_pct >= 0
]
```

Add to the return dict:

```python
"avg_mbu_pct": statistics.mean(mbu_samples) if mbu_samples else None,
"max_mbu_pct": max(mbu_samples) if mbu_samples else None,
```

**Step 2: Run existing tests**

Run: `cd intelligence-per-watt && python -m pytest src/ipw/tests/ -v --no-header -x`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add intelligence-per-watt/src/ipw/cli/bench.py
git commit -m "feat: extract MBU from telemetry in bench.py energy metrics"
```

---

### Task 7: Documentation — Document `ipw bench`

**Files:**
- Modify: `docs/user-guide/profiling.md`

**Step 1: Add `ipw bench` section**

Add a new section to `docs/user-guide/profiling.md` documenting:

- What `ipw bench` does (full pipeline: server start → warmup → telemetry → benchmark → accuracy eval → display → server stop)
- All CLI flags with descriptions:
  - `--agent` (required): Agent type (react, openhands, terminus, terminus-tb)
  - `--model`: HuggingFace model ID
  - `--preset`: Model preset name (alternative to --model)
  - `--dataset` (required): Dataset to benchmark
  - `--limit`: Max queries
  - `--output`: Output directory
  - `--client`: Model provider (vllm, openai, ollama)
  - `--vllm-url`: Override server URL
  - `--api-key`: API key
  - `--per-action`: Per-action energy breakdown
  - `--no-telemetry`: Disable energy telemetry
  - `--skip-warmup`: Skip warmup phase
  - `--auto-server`: Auto-manage inference server lifecycle
  - `--submodel`: Submodel specification
  - `--base-port`: Base port for auto-server
  - `--seed`: Random seed
- Example commands
- Output format: what's in `summary.json`, `traces.jsonl`
- Model presets: how to use `--preset`, where presets are defined

**Step 2: Add `ipw servers` section**

Add brief documentation of `ipw servers start|launch|stop|status` subcommands.

**Step 3: Commit**

```bash
git add docs/user-guide/profiling.md
git commit -m "docs: document ipw bench and ipw servers commands"
```

---

### Task 8: Documentation Fixes

**Files:**
- Create: `docs/includes/abbreviations.md`
- Modify: `mkdocs.yml` (nav label fix)

**Step 1: Create abbreviations file**

Create `docs/includes/abbreviations.md`:

```markdown
*[IPW]: Intelligence Per Watt
*[IPJ]: Intelligence Per Joule
*[MBU]: Memory Bandwidth Utilization
*[MFU]: Model FLOPs Utilization
*[TTFT]: Time To First Token
*[ITL]: Inter-Token Latency
*[NVML]: NVIDIA Management Library
*[RAPL]: Running Average Power Limit
*[gRPC]: Google Remote Procedure Call
*[vLLM]: Virtual Large Language Model server
```

**Step 2: Fix mkdocs.yml nav label**

Change "Telemetry" to "Benchmarking" in the navigation tabs section of `mkdocs.yml`.

**Step 3: Commit**

```bash
git add docs/includes/abbreviations.md mkdocs.yml
git commit -m "docs: add abbreviations file and fix nav label"
```

---

### Task 9: Final Verification

**Step 1: Run full test suite**

Run: `cd intelligence-per-watt && python -m pytest src/ipw/tests/ -v --no-header -x`
Expected: ALL PASS

**Step 2: Verify summary.json schema with a dry-run test**

Create a one-off test that generates a `summary.json` from synthetic traces and validates all expected keys exist:

```python
# Verify these top-level keys exist:
# config, totals, efficiency, averages, statistics, normalized_statistics, normalized_efficiency, generated_at
# Verify totals has: accuracy
# Verify efficiency has: accuracy, ipj, ipw, total_gpu_energy_joules, avg_gpu_power_watts
# Verify statistics has: mbu_avg_pct
# Verify normalized_statistics has: _outliers_removed, _description, wall_clock_s, ...
```

**Step 3: Run ruff linter**

Run: `cd intelligence-per-watt && ruff check src/ipw/`
Expected: No errors

**Step 4: Final commit if any fixes needed**

```bash
git add -A && git commit -m "fix: address linting issues from pipeline completion"
```
