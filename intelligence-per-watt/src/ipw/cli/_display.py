"""Rich display helpers for CLI output — banner, metrics tables, efficiency panels."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import TYPE_CHECKING, List, NamedTuple, Optional, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

if TYPE_CHECKING:
    from ..execution.trace import QueryTrace
    from ..execution.types import ProfilingRecord

BANNER = r"""
 _____ ______  _    _
|_   _|| ___ \| |  | |
  | |  | |_/ /| |  | |
  | |  |  __/ | |/\| |
 _| |_ | |    \  /\  /
 \___/ \_|     \/  \/
 Intelligence Per Watt
"""

# Module-level console with markup enabled for Rich renderables.
display_console = Console(highlight=False)


class MetricRow(NamedTuple):
    label: str
    avg: Optional[float]
    median: Optional[float]
    min: Optional[float]
    max: Optional[float]
    std: Optional[float]
    unit: str


# ---------------------------------------------------------------------------
# Value formatting
# ---------------------------------------------------------------------------

def _fmt(value: Optional[float], precision: int = 2) -> str:
    """Format a float with *precision* decimals, or return '—' for None/NaN."""
    if value is None:
        return "\u2014"
    try:
        if not isinstance(value, (int, float)):
            return "\u2014"
        import math
        if math.isnan(value) or math.isinf(value):
            return "\u2014"
        return f"{value:,.{precision}f}"
    except (TypeError, ValueError):
        return "\u2014"


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

def print_banner(console: Optional[Console] = None) -> None:
    con = console or display_console
    con.print(BANNER, style="bold cyan", highlight=False)
    con.print(Rule(style="cyan"))


# ---------------------------------------------------------------------------
# Config summary
# ---------------------------------------------------------------------------

def print_config_summary(
    console: Optional[Console] = None,
    *,
    config: Optional[dict] = None,
) -> None:
    """Print a compact 2-column Rich Table summarising the run configuration."""
    con = console or display_console
    if not config:
        return

    tbl = Table(title="Run Configuration", show_header=False, box=None, padding=(0, 2))
    tbl.add_column("Key", style="bold")
    tbl.add_column("Value")

    for key, value in config.items():
        if value is not None:
            tbl.add_row(key, str(value))

    con.print(tbl)
    con.print()


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

def compute_aggregate_stats(values: Sequence[Optional[float]]) -> dict:
    """Return {avg, median, min, max, std} filtering None values."""
    clean = [v for v in values if v is not None]
    if not clean:
        return {"avg": None, "median": None, "min": None, "max": None, "std": None}
    return {
        "avg": statistics.mean(clean),
        "median": statistics.median(clean),
        "min": min(clean),
        "max": max(clean),
        "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Profile metrics extraction
# ---------------------------------------------------------------------------

def _safe_get(obj, *attrs, scale: float = 1.0):
    """Safely traverse nested attributes, returning None on failure."""
    cur = obj
    for a in attrs:
        cur = getattr(cur, a, None)
        if cur is None:
            return None
    try:
        return float(cur) * scale
    except (TypeError, ValueError):
        return None


def compute_profile_metrics(
    records: List["ProfilingRecord"],
    model: str,
) -> List[MetricRow]:
    """Extract per-query values from ProfilingRecord list, return MetricRow list."""
    if not records:
        return []

    # Find the model key in model_metrics (usually the model name).
    def _metrics(rec):
        if model in rec.model_metrics:
            return rec.model_metrics[model]
        # Fall back to the first entry if only one exists.
        if len(rec.model_metrics) == 1:
            return next(iter(rec.model_metrics.values()))
        return None

    def _collect(extractor):
        vals = []
        for rec in records:
            m = _metrics(rec)
            if m is not None:
                vals.append(extractor(m))
        return vals

    def _row(label, values, unit):
        s = compute_aggregate_stats(values)
        return MetricRow(label, s["avg"], s["median"], s["min"], s["max"], s["std"], unit)

    rows = [
        _row("GPU Energy", _collect(lambda m: _safe_get(m, "energy_metrics", "per_query_joules")), "J"),
        _row("CPU Energy", _collect(lambda m: _safe_get(m, "energy_metrics", "cpu_per_query_joules")), "J"),
        _row("GPU Power", _collect(lambda m: _safe_get(m, "power_metrics", "gpu", "per_query_watts", "avg")), "W"),
        _row("Latency", _collect(lambda m: _safe_get(m, "latency_metrics", "total_query_seconds")), "s"),
        _row("TTFT", _collect(lambda m: _safe_get(m, "latency_metrics", "time_to_first_token_seconds", scale=1000.0)), "ms"),
        _row("Throughput", _collect(lambda m: _safe_get(m, "latency_metrics", "throughput_tokens_per_sec")), "tok/s"),
        _row("Input Tokens", _collect(lambda m: _safe_get(m, "token_metrics", "input")), ""),
        _row("Output Tokens", _collect(lambda m: _safe_get(m, "token_metrics", "output")), ""),
        _row("Energy/Token", _collect(lambda m: _safe_get(m, "energy_metrics", "energy_per_output_token_joules")), "J"),
        _row("ITL Median", _collect(lambda m: _safe_get(m, "latency_metrics", "median_itl_ms")), "ms"),
        _row("Throughput/Watt", _collect(lambda m: _safe_get(m, "efficiency", "throughput_per_watt")), ""),
    ]
    return rows


# ---------------------------------------------------------------------------
# Trace metrics extraction
# ---------------------------------------------------------------------------

def compute_trace_metrics(traces: List["QueryTrace"]) -> List[MetricRow]:
    """Extract per-query values from QueryTrace list, return MetricRow list."""
    if not traces:
        return []

    def _row(label, values, unit):
        s = compute_aggregate_stats(values)
        return MetricRow(label, s["avg"], s["median"], s["min"], s["max"], s["std"], unit)

    completed_count = sum(1 for t in traces if t.completed)
    total = len(traces)

    rows = [
        _row("Wall Clock", [t.total_wall_clock_s for t in traces], "s"),
        _row("GPU Energy", [t.total_gpu_energy_joules for t in traces], "J"),
        _row("CPU Energy", [t.total_cpu_energy_joules for t in traces], "J"),
        _row("Input Tokens", [float(t.total_input_tokens) for t in traces], ""),
        _row("Output Tokens", [float(t.total_output_tokens) for t in traces], ""),
        _row("Turns", [float(t.num_turns) for t in traces], ""),
        _row("Tool Calls", [float(t.total_tool_calls) for t in traces], ""),
        MetricRow(
            "Completed",
            float(completed_count),
            None,
            None,
            None,
            None,
            f"{completed_count}/{total}",
        ),
    ]
    return rows


# ---------------------------------------------------------------------------
# Metrics table
# ---------------------------------------------------------------------------

def print_metrics_table(
    console: Optional[Console] = None,
    *,
    rows: List[MetricRow],
    title: str = "Aggregate Metrics",
) -> None:
    """Render a Rich Table with columns: Metric | Avg | Median | Min | Max | Std."""
    con = console or display_console
    if not rows:
        return

    # Filter out rows where all numeric values are None.
    visible = [
        r for r in rows
        if not all(v is None for v in (r.avg, r.median, r.min, r.max, r.std))
    ]
    if not visible:
        return

    tbl = Table(title=title, show_lines=False)
    tbl.add_column("Metric", style="bold")
    tbl.add_column("Avg", justify="right")
    tbl.add_column("Median", justify="right")
    tbl.add_column("Min", justify="right")
    tbl.add_column("Max", justify="right")
    tbl.add_column("Std", justify="right")
    tbl.add_column("Unit", style="dim")

    for r in visible:
        tbl.add_row(
            r.label,
            _fmt(r.avg),
            _fmt(r.median),
            _fmt(r.min),
            _fmt(r.max),
            _fmt(r.std),
            r.unit,
        )

    con.print()
    con.print(tbl)


# ---------------------------------------------------------------------------
# Efficiency panel (IPJ / IPW)
# ---------------------------------------------------------------------------

def print_efficiency_panel(
    console: Optional[Console] = None,
    *,
    records: Optional[List["ProfilingRecord"]] = None,
    traces: Optional[List["QueryTrace"]] = None,
    model: Optional[str] = None,
    accuracy: Optional[float] = None,
) -> None:
    """Display aggregate IPJ / IPW in a Rich Panel."""
    con = console or display_console
    total_energy: float = 0.0
    avg_power: float = 0.0
    acc: Optional[float] = accuracy

    if records and model:
        energies = []
        powers = []
        for rec in records:
            mm = rec.model_metrics.get(model)
            if mm is None and len(rec.model_metrics) == 1:
                mm = next(iter(rec.model_metrics.values()))
            if mm is None:
                continue
            e = mm.energy_metrics.per_query_joules
            if e is not None:
                energies.append(e)
            p = mm.power_metrics.gpu.per_query_watts.avg
            if p is not None:
                powers.append(p)
        total_energy = sum(energies) if energies else 0.0
        avg_power = statistics.mean(powers) if powers else 0.0

    elif traces:
        energies = [t.total_gpu_energy_joules for t in traces if t.total_gpu_energy_joules is not None]
        total_energy = sum(energies) if energies else 0.0
        # Derive power from energy / time as a proxy
        total_time = sum(t.total_wall_clock_s for t in traces if t.total_wall_clock_s > 0)
        avg_power = total_energy / total_time if total_time > 0 else 0.0
        if acc is None:
            completed = sum(1 for t in traces if t.completed)
            total = len(traces)
            acc = completed / total if total > 0 else 0.0

    lines: list[str] = []
    if acc is not None and total_energy > 0:
        ipj = acc / total_energy
        lines.append(f"[bold green]IPJ[/bold green]  (Intelligence Per Joule):  [bold]{ipj:.4f}[/bold]")
    elif total_energy > 0:
        lines.append(f"[dim]IPJ  (Intelligence Per Joule):  N/A (no accuracy data)[/dim]")

    if acc is not None and avg_power > 0:
        ipw = acc / avg_power
        lines.append(f"[bold green]IPW[/bold green]  (Intelligence Per Watt):   [bold]{ipw:.4f}[/bold]")
    elif avg_power > 0:
        lines.append(f"[dim]IPW  (Intelligence Per Watt):   N/A (no accuracy data)[/dim]")

    if not lines:
        return

    con.print()
    con.print(Panel("\n".join(lines), title="Efficiency", border_style="green"))


# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------

def print_output_path(console: Optional[Console] = None, *, path: Optional[Path] = None) -> None:
    """Print the output directory in a styled display."""
    con = console or display_console
    if path is None:
        return
    con.print()
    con.print(Panel(str(path), title="Output Directory", border_style="blue"))


__all__ = [
    "BANNER",
    "MetricRow",
    "compute_aggregate_stats",
    "compute_profile_metrics",
    "compute_trace_metrics",
    "display_console",
    "print_banner",
    "print_config_summary",
    "print_efficiency_panel",
    "print_metrics_table",
    "print_output_path",
    "_fmt",
]
