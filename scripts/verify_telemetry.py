#!/usr/bin/env python3
"""Quick verification script for NVIDIA GPU+CPU telemetry fields."""

from __future__ import annotations

import math
import sys
import time

from ipw.telemetry import EnergyMonitorCollector


def main() -> None:
    collector = EnergyMonitorCollector(target="")

    print("Connecting to energy monitor...")
    with collector.start():
        print(f"Collector: {collector.collector_name}")
        print("Collecting 10 samples (~500ms)...\n")

        samples = []
        start = time.time()
        for reading in collector.stream_readings():
            samples.append(reading)
            if len(samples) >= 10 or (time.time() - start) > 5.0:
                break

    if not samples:
        print("FAIL: No samples collected!")
        sys.exit(1)

    print(f"Collected {len(samples)} samples\n")

    # Check GPU fields
    checks = {
        "power_watts": [],
        "energy_joules": [],
        "temperature_celsius": [],
        "gpu_memory_usage_mb": [],
        "gpu_compute_utilization_pct": [],
        "cpu_energy_joules": [],
        "cpu_memory_usage_mb": [],
    }

    for s in samples:
        checks["power_watts"].append(s.power_watts)
        checks["energy_joules"].append(s.energy_joules)
        checks["temperature_celsius"].append(s.temperature_celsius)
        checks["gpu_memory_usage_mb"].append(s.gpu_memory_usage_mb)
        checks["gpu_compute_utilization_pct"].append(s.gpu_compute_utilization_pct)
        checks["cpu_energy_joules"].append(s.cpu_energy_joules)
        checks["cpu_memory_usage_mb"].append(s.cpu_memory_usage_mb)

    passed = 0
    failed = 0

    for field, values in checks.items():
        valid = [v for v in values if v is not None and math.isfinite(v) and v >= 0]
        positive = [v for v in valid if v > 0]

        if field == "gpu_compute_utilization_pct":
            # GPU util can be 0 when idle, or None on some NVML configs
            status = "PASS" if len(valid) > 0 else "WARN (utilization not exposed)"
        elif field in ("cpu_energy_joules",):
            # RAPL may not be available
            status = "PASS" if len(valid) > 0 else "WARN (RAPL may not be available)"
        else:
            status = "PASS" if len(positive) > 0 else "FAIL"

        if status == "PASS":
            passed += 1
        elif "FAIL" in status:
            failed += 1

        sample_vals = values[:3]
        print(f"  {field:35s}: {status}  (samples: {sample_vals})")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")

    # Print one full sample for visual inspection
    s = samples[0]
    print(f"\nFull first sample:")
    print(f"  timestamp_nanos:           {s.timestamp_nanos}")
    print(f"  power_watts:               {s.power_watts}")
    print(f"  energy_joules:             {s.energy_joules}")
    print(f"  temperature_celsius:       {s.temperature_celsius}")
    print(f"  gpu_memory_usage_mb:       {s.gpu_memory_usage_mb}")
    print(f"  gpu_memory_total_mb:       {s.gpu_memory_total_mb}")
    print(f"  gpu_compute_utilization:   {s.gpu_compute_utilization_pct}")
    print(f"  cpu_energy_joules:         {s.cpu_energy_joules}")
    print(f"  cpu_memory_usage_mb:       {s.cpu_memory_usage_mb}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
