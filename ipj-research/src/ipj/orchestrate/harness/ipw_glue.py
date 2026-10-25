"""Glue to the IPW MCP layer: worker server factory, tool surface, and telemetry energy probe."""

from __future__ import annotations

import math
import os
from typing import Any

from ipj.energy.capture import DEFAULT_ENERGY_COVERAGE_POLICY, EnergyCoveragePolicy
from ipj.metrics import (
    UNKNOWN_JOULES_PROVENANCE,
    accelerator_joules_are_complete,
    resolve_joules_provenance,
)

from .dispatch import AcceleratorEnergyMeasurement, EnergyProbe, WorkerResult
from .tools import ToolOutcome, ToolSurface, bash_tool_infos
from .workers import WorkerBinding


class _MCPWorkerAdapter:
    """Adapts an IPW BaseMCPServer (returns MCPToolResult) to our WorkerServer protocol."""

    def __init__(self, server: Any) -> None:
        self._server = server

    def execute(self, prompt: str, **params: Any) -> WorkerResult:
        res = self._server.execute(prompt, **params)
        usage = getattr(res, "usage", {}) or {}
        return WorkerResult(
            content=getattr(res, "content", ""),
            tokens_in=usage.get("prompt_tokens"),
            tokens_out=usage.get("completion_tokens"),
            dollars=getattr(res, "cost_usd", None),
            latency_s=getattr(res, "latency_seconds", None),
        )


def mcp_server_factory(binding: WorkerBinding) -> _MCPWorkerAdapter:
    provider = binding.provider
    if provider == "openai":
        from ipw.agents.mcp.openai_server import OpenAIMCPServer

        server = OpenAIMCPServer(model_name=binding.model_id, api_key=os.getenv("OPENAI_API_KEY"))
    elif provider == "anthropic":
        from ipw.agents.mcp.anthropic_server import AnthropicMCPServer

        server = AnthropicMCPServer(model_name=binding.model_id, api_key=os.getenv("ANTHROPIC_API_KEY"))
    elif provider == "gemini":
        from ipw.agents.mcp.gemini_server import GeminiMCPServer

        server = GeminiMCPServer(model_name=binding.model_id, api_key=os.getenv("GEMINI_API_KEY"))
    elif provider == "openrouter":
        from ipw.agents.mcp.openrouter_server import OpenRouterMCPServer

        server = OpenRouterMCPServer(model_name=binding.model_id, api_key=os.getenv("OPENROUTER_API_KEY"))
    elif provider == "vllm":
        from ipw.agents.mcp.vllm_server import VLLMMCPServer

        server = VLLMMCPServer(model_name=binding.model_id, vllm_url=binding.base_url or "http://localhost:8000")
    else:
        raise ValueError(f"unsupported provider {provider!r}")
    return _MCPWorkerAdapter(server)


def _derive_success(res: Any) -> bool:
    """Derive success from an IPW MCPToolResult instead of assuming it.

    ShellServer/CodeInterpreterServer signal failure via metadata: an ``error`` key on
    timeouts/exceptions/guard rejections, or a non-zero ``return_code``. Absent any signal we
    default to success.
    """
    meta = getattr(res, "metadata", {}) or {}
    if meta.get("error") is not None:
        return False
    if "return_code" in meta:
        return meta["return_code"] == 0
    return True


class _ShellBackend:
    def __init__(self, server: Any) -> None:
        self._server = server

    def run(self, **kwargs: Any) -> ToolOutcome:
        res = self._server.execute(kwargs.get("command", ""))
        return ToolOutcome(
            content=getattr(res, "content", ""),
            success=_derive_success(res),
            dollars=getattr(res, "cost_usd", None),
            latency_s=getattr(res, "latency_seconds", None),
        )


def build_tool_surface() -> ToolSurface:
    from ipw.agents.mcp.tool_server import CodeInterpreterServer, ShellServer

    backends = {
        "bash": _ShellBackend(ShellServer()),  # real /bin/bash shell
        "code_interpreter": _ShellBackend(CodeInterpreterServer()),  # Python interpreter
    }
    return ToolSurface(backends, bash_tool_infos())


def telemetry_energy_probe(
    session: Any,
    *,
    provenance: str = UNKNOWN_JOULES_PROVENANCE,
    accelerator_device_coverage_complete: bool = False,
    coverage_policy: EnergyCoveragePolicy = DEFAULT_ENERGY_COVERAGE_POLICY,
) -> EnergyProbe:
    """Build a cumulative-energy probe over epoch timestamps.

    ``TelemetryReading.energy_joules`` is a cumulative device counter, so the
    measurement is the last-first delta, never the sum of readings. A complete
    measurement also requires continuous boundary/sample coverage and an
    explicit dedicated-measurement provenance declaration.
    """

    def _probe(start_ts: float, end_ts: float) -> AcceleratorEnergyMeasurement:
        try:
            if not math.isfinite(start_ts) or not math.isfinite(end_ts) or end_ts <= start_ts:
                return AcceleratorEnergyMeasurement(None)
            points: list[tuple[float, float]] = []
            for sample in session.window(start_ts, end_ts):
                timestamp = float(sample.timestamp)
                energy_joules = float(sample.reading.energy_joules)
                if (
                    not math.isfinite(timestamp)
                    or not math.isfinite(energy_joules)
                    or energy_joules < 0
                    or timestamp < start_ts
                    or timestamp > end_ts
                ):
                    return AcceleratorEnergyMeasurement(None)
                points.append((timestamp, energy_joules))
            points.sort(key=lambda point: point[0])
            times = sorted({timestamp for timestamp, _ in points})
            if len(times) < coverage_policy.minimum_distinct_samples:
                return AcceleratorEnergyMeasurement(None)
            if times[0] - start_ts > coverage_policy.max_boundary_gap_s:
                return AcceleratorEnergyMeasurement(None)
            if end_ts - times[-1] > coverage_policy.max_boundary_gap_s:
                return AcceleratorEnergyMeasurement(None)
            if any(
                later - earlier > coverage_policy.max_sample_gap_s
                for earlier, later in zip(times, times[1:])
            ):
                return AcceleratorEnergyMeasurement(None)
            if any(
                later_energy < earlier_energy
                for (_, earlier_energy), (_, later_energy) in zip(points, points[1:])
            ):
                return AcceleratorEnergyMeasurement(None)
            delta = points[-1][1] - points[0][1]
            if not math.isfinite(delta) or delta < 0:
                return AcceleratorEnergyMeasurement(None)
            resolved_provenance = resolve_joules_provenance(delta, provenance)
            return AcceleratorEnergyMeasurement(
                delta,
                provenance=resolved_provenance,
                complete=accelerator_joules_are_complete(
                    delta,
                    resolved_provenance,
                    accelerator_device_coverage_complete,
                ),
                accelerator_device_coverage_complete=(
                    accelerator_device_coverage_complete
                ),
            )
        except Exception:  # noqa: BLE001 - backend failure means unavailable energy
            return AcceleratorEnergyMeasurement(None)

    return _probe
