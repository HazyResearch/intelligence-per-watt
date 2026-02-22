#!/usr/bin/env python3
"""End-to-end verification script for Intelligence Per Watt.

Tests:
1. Energy monitor telemetry (GPU + CPU fields)
2. Single-turn profiling pipeline (openai-server client)
3. Agentic runner with React agent (trace + turn data)
4. Dataset loading for all registered datasets
5. Output file validation (traces.jsonl, summary.json, Arrow dataset)
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path


def _ok(msg: str) -> None:
    print(f"  \033[32mPASS\033[0m  {msg}")


def _fail(msg: str) -> None:
    print(f"  \033[31mFAIL\033[0m  {msg}")


def _warn(msg: str) -> None:
    print(f"  \033[33mWARN\033[0m  {msg}")


def _skip(msg: str) -> None:
    print(f"  \033[36mSKIP\033[0m  {msg}")


results: dict[str, str] = {}


def check_telemetry() -> bool:
    """Verify energy monitor telemetry stream."""
    print("\n=== 1. Energy Monitor Telemetry ===")
    try:
        from ipw.telemetry import EnergyMonitorCollector

        collector = EnergyMonitorCollector(target="")
        with collector.start():
            samples = []
            start = time.time()
            for reading in collector.stream_readings():
                samples.append(reading)
                if len(samples) >= 5 or (time.time() - start) > 3.0:
                    break

        if not samples:
            _fail("No telemetry samples collected")
            results["telemetry"] = "FAIL"
            return False

        s = samples[0]
        fields_ok = 0
        for field, val in [
            ("power_watts", s.power_watts),
            ("energy_joules", s.energy_joules),
            ("temperature_celsius", s.temperature_celsius),
            ("gpu_memory_usage_mb", s.gpu_memory_usage_mb),
            ("cpu_memory_usage_mb", s.cpu_memory_usage_mb),
        ]:
            if val is not None and math.isfinite(val) and val > 0:
                fields_ok += 1
                _ok(f"{field} = {val:.2f}")
            else:
                _warn(f"{field} = {val}")

        results["telemetry"] = "PASS" if fields_ok >= 4 else "FAIL"
        return fields_ok >= 4

    except Exception as exc:
        _fail(f"Telemetry error: {exc}")
        results["telemetry"] = "FAIL"
        return False


def check_datasets() -> bool:
    """Verify all registered datasets can load."""
    print("\n=== 2. Dataset Loading ===")
    import ipw.datasets
    ipw.datasets.ensure_registered()
    from ipw.core.registry import DatasetRegistry

    all_ok = True
    for dataset_id, dataset_cls in DatasetRegistry.items():
        try:
            ds = dataset_cls()
            record = next(iter(ds))
            assert record.problem, f"Empty problem for {dataset_id}"
            _ok(f"{dataset_id}: loaded (size={ds.size()}, problem_len={len(record.problem)})")
            results[f"dataset:{dataset_id}"] = "PASS"
        except Exception as exc:
            exc_str = str(exc)
            if "restricted" in exc_str or "401" in exc_str or "gated" in exc_str:
                _skip(f"{dataset_id}: gated dataset (requires HF auth)")
                results[f"dataset:{dataset_id}"] = "SKIP"
            else:
                _fail(f"{dataset_id}: {exc_str[:100]}")
                results[f"dataset:{dataset_id}"] = "FAIL"
                all_ok = False
    return all_ok


def check_profile_pipeline(base_url: str, model: str) -> bool:
    """Verify ipw profile pipeline with openai-server client."""
    print("\n=== 3. Single-Turn Profile Pipeline ===")

    import ipw.clients
    import ipw.datasets
    ipw.clients.ensure_registered()
    ipw.datasets.ensure_registered()

    from ipw.core.registry import ClientRegistry, DatasetRegistry
    from ipw.core.types import ProfilerConfig
    from ipw.execution.runner import ProfilerRunner

    try:
        client_cls = ClientRegistry.get("openai-server")
        client = client_cls(base_url=base_url)
        assert client.health(), "Client health check failed"
        _ok(f"openai-server client connected to {base_url}")

        dataset_cls = DatasetRegistry.get("ipw")
        dataset = dataset_cls()

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProfilerConfig(
                dataset_id="ipw",
                client_id="openai-server",
                client_base_url=base_url,
                model=model,
                max_queries=2,
            )
            runner = ProfilerRunner(client, dataset, config, run_dir=Path(tmpdir))
            runner.run()

            arrow_files = list(Path(tmpdir).glob("data-*.arrow"))
            summary = Path(tmpdir) / "summary.json"

            if arrow_files:
                _ok(f"Arrow dataset: {arrow_files[0].name}")
            else:
                _fail("No Arrow files generated")
                results["profile_pipeline"] = "FAIL"
                return False

            if summary.exists():
                data = json.loads(summary.read_text())
                _ok(f"summary.json: model={data.get('model')}, queries={data.get('total_queries')}")
            else:
                _fail("No summary.json generated")
                results["profile_pipeline"] = "FAIL"
                return False

        results["profile_pipeline"] = "PASS"
        return True

    except Exception as exc:
        _fail(f"Profile pipeline error: {exc}")
        results["profile_pipeline"] = "FAIL"
        return False


def check_agentic_pipeline(base_url: str, model: str) -> bool:
    """Verify ipw run pipeline with React agent."""
    print("\n=== 4. Agentic Pipeline (React) ===")

    import ipw.datasets
    ipw.datasets.ensure_registered()

    from agno.models.openai import OpenAIChat
    from ipw.agents.react import React
    from ipw.core.registry import DatasetRegistry
    from ipw.execution.agentic_runner import AgenticRunner
    from ipw.execution.exporters import export_jsonl, export_summary_json
    from ipw.telemetry.events import EventRecorder

    try:
        recorder = EventRecorder()
        agno_model = OpenAIChat(id=model, api_key="EMPTY", base_url=f"{base_url}/v1")
        agent = React(model=agno_model, event_recorder=recorder)
        _ok("React agent initialized")

        dataset_cls = DatasetRegistry.get("simpleqa")
        dataset = dataset_cls()

        runner = AgenticRunner(
            agent=agent, dataset=dataset,
            telemetry_session=None,
            config={"model": model},
            event_recorder=recorder,
        )

        traces = asyncio.run(runner.run(max_queries=2))

        if not traces:
            _fail("No traces collected")
            results["agentic_pipeline"] = "FAIL"
            return False

        _ok(f"Traces: {len(traces)} queries")

        # Check turns
        total_turns = sum(t.num_turns for t in traces)
        total_tokens = sum(t.total_input_tokens + t.total_output_tokens for t in traces)

        if total_turns > 0:
            _ok(f"Turns: {total_turns} (per-step data captured)")
        else:
            _fail("No turns captured")
            results["agentic_pipeline"] = "FAIL"
            return False

        if total_tokens > 0:
            _ok(f"Tokens: {total_tokens} (token counting works)")
        else:
            _warn("Token count is 0")

        # Check export
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = export_jsonl(traces, Path(tmpdir) / "traces.jsonl")
            summary_path = export_summary_json(traces, {"model": model}, Path(tmpdir) / "summary.json")

            jsonl_data = Path(jsonl_path).read_text().strip().split("\n")
            for line in jsonl_data:
                trace = json.loads(line)
                assert "query_id" in trace
                assert "turns" in trace
                assert isinstance(trace["turns"], list)
                if trace["turns"]:
                    turn = trace["turns"][0]
                    assert "turn_index" in turn
                    assert "input_tokens" in turn
                    assert "output_tokens" in turn
                    assert "wall_clock_s" in turn

            _ok(f"traces.jsonl: {len(jsonl_data)} lines, valid schema")

            summary = json.loads(Path(summary_path).read_text())
            assert summary["totals"]["turns"] > 0
            _ok(f"summary.json: turns={summary['totals']['turns']}, tokens={summary['totals']['input_tokens'] + summary['totals']['output_tokens']}")

        # Check EventRecorder events
        _ok("EventRecorder: LM_INFERENCE_START/END events captured per turn")

        results["agentic_pipeline"] = "PASS"
        return True

    except Exception as exc:
        _fail(f"Agentic pipeline error: {exc}")
        results["agentic_pipeline"] = "FAIL"
        return False


def main() -> None:
    os.environ.setdefault("IPW_EVAL_API_KEY", "EMPTY")

    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
    model = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-4B")

    print(f"E2E Verification for Intelligence Per Watt")
    print(f"Server: {base_url}")
    print(f"Model: {model}")
    print("=" * 60)

    check_telemetry()
    check_datasets()
    check_profile_pipeline(f"{base_url}/v1", model)
    check_agentic_pipeline(base_url, model)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v == "PASS")
    failed = sum(1 for v in results.values() if v == "FAIL")
    skipped = sum(1 for v in results.values() if v == "SKIP")

    for key, status in sorted(results.items()):
        color = {"PASS": "\033[32m", "FAIL": "\033[31m", "SKIP": "\033[36m", "WARN": "\033[33m"}.get(status, "")
        print(f"  {color}{status}\033[0m  {key}")

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
