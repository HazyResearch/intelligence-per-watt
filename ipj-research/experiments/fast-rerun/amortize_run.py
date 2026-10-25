#!/usr/bin/env python
"""Turn a finished harbor run (jobs dir of per-task result.json) + a board-power time series into
the MANDATORY central-percentile amortized metrics (total/P90/P80/P50). Run where `ipj` imports.

Harbor's per-task result.json carries `started_at`/`finished_at` (ISO), `agent_result.n_input_tokens`/
`n_output_tokens`, and the verifier reward — exactly the per-query event log the methodology needs.

Usage:
  python amortize_run.py --jobs-dir runs/.../jobs --power power.txt \
      [--gpu-hourly-rate 1.10 --n-gpus 1] [--price-in 0 --price-out 0] [--out metrics.json]
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
from datetime import datetime

from ipj.efficiency.cost import H100_HOURLY_USD
from ipj.percentile_amortize import QueryEvent, amortize_all


def _epoch(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def load_events(jobs_dir: str) -> list[QueryEvent]:
    pats = [os.path.join(jobs_dir, "*", "*__*", "result.json"),
            os.path.join(jobs_dir, "*__*", "result.json")]
    seen, events = set(), []
    for pat in pats:
        for f in glob.glob(pat):
            if f in seen:
                continue
            seen.add(f)
            try:
                d = json.load(open(f))
            except (json.JSONDecodeError, OSError):
                continue
            if not d.get("started_at") or not d.get("finished_at"):
                continue
            ar = d.get("agent_result") or {}
            reward = ((d.get("verifier_result") or {}).get("rewards") or {}).get("reward")
            events.append(QueryEvent(
                start=_epoch(d["started_at"]), end=_epoch(d["finished_at"]),
                tokens_in=int(ar.get("n_input_tokens") or 0),
                tokens_out=int(ar.get("n_output_tokens") or 0),
                resolved=(reward is not None and reward >= 1.0),
            ))
    return events


def load_power(path: str) -> list[tuple[float, float]]:
    s = []
    for ln in open(path):
        p = ln.split()
        if len(p) >= 2:
            try:
                s.append((float(p[0]), float(p[1])))
            except ValueError:
                pass
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs-dir", required=True)
    ap.add_argument("--power", required=True)
    ap.add_argument("--gpu-hourly-rate", type=float, default=H100_HOURLY_USD,
                    help="US$/GPU-hr for local amortized cost (default: the decided $3.00 H100 reference)")
    ap.add_argument("--n-gpus", type=int, default=1)
    ap.add_argument("--price-in", type=float, default=0.0)
    ap.add_argument("--price-out", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    events = load_events(a.jobs_dir)
    power = load_power(a.power)
    n = len(events)
    resolved = sum(e.resolved for e in events)
    acc = resolved / n if n else 0.0
    print(f"queries={n} resolved={resolved} accuracy(total-N)={acc:.3f} power_samples={len(power)}")
    res = amortize_all(events, power, gpu_hourly_rate=a.gpu_hourly_rate, n_gpus=a.n_gpus,
                       price_in=a.price_in, price_out=a.price_out)
    for p, m in res.items():
        dpq = f"{m.dollars_per_query:.4f}" if m.dollars_per_query is not None else "n/a"
        print(f"  P{p}: n={m.n_queries} E/q={m.energy_per_query_j:.0f}J avgP={m.avg_power_w:.0f}W "
              f"$/q={dpq} lat(mean/p50/p90)={m.latency_mean_s:.1f}/{m.latency_p50_s:.1f}/{m.latency_p90_s:.1f}s "
              f"in/q={m.tokens_in_per_query:.0f} out/q={m.tokens_out_per_query:.0f} dec={m.decode_throughput_tok_s:.1f}tok/s")
    if a.out:
        json.dump({"n": n, "resolved": resolved, "accuracy": acc,
                   "windows": {p: dataclasses.asdict(m) for p, m in res.items()}}, open(a.out, "w"), indent=2)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
