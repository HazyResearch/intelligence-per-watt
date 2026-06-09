from __future__ import annotations

import subprocess

import pytest

from ipw.execution import nvidia_smi_telemetry as mod
from ipw.execution.nvidia_smi_telemetry import NvidiaSmiTelemetrySession


def test_query_nvidia_smi_parses_gpu_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="0, 250.5, 1024, 81559, 97\n",
            stderr="",
        )

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    rows = mod._query_nvidia_smi([0])

    assert len(rows) == 1
    assert rows[0].gpu_id == 0
    assert rows[0].power_watts == pytest.approx(250.5)
    assert rows[0].memory_used_mb == pytest.approx(1024.0)
    assert rows[0].utilization_pct == pytest.approx(97.0)


def test_session_integrates_energy_for_selected_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter([10.0, 11.0, 12.0])
    samples = iter([
        [mod._GpuSample(3, 100.0, 10.0, 100.0, 50.0)],
        [mod._GpuSample(3, 200.0, 20.0, 100.0, 60.0)],
        [mod._GpuSample(3, 200.0, 30.0, 100.0, 70.0)],
    ])
    monkeypatch.setattr(mod.time, "time", lambda: next(times))
    monkeypatch.setattr(mod, "_query_nvidia_smi", lambda gpu_ids: next(samples))

    session = NvidiaSmiTelemetrySession([3])
    session._sample_once()
    session._sample_once()
    session._sample_once()

    readings = list(session.readings())
    assert len(readings) == 3
    assert readings[-1].reading.gpu_info is not None
    assert readings[-1].reading.gpu_info.device_id == 3
    assert readings[-1].reading.energy_joules == pytest.approx(350.0)
    assert readings[-1].reading.power_watts == pytest.approx(200.0)
