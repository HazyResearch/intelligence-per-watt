#!/usr/bin/env python3
"""Self-contained (stdlib-only) central-percentile amortizer for a matx fast-serving run.

Parses a harbor jobs dir (per-task result.json: started_at/finished_at/tokens/reward) + a board-power
time series (power.txt: "<epoch> <watts>" per line) and reports each cost metric amortized over the
central-percentile windows total/P90/P80/P50 (P50 = steady-state headline) — the MANDATORY IPJ
methodology (ipj/experiments/INFERENCE_CONFIG.md).

It has NO `ipj` dependency on purpose, so the run-script can drop it onto a bare compute node and have
the job self-amortize (its metrics land in the job's stdout + a metrics.json). It mirrors the canonical
ipj/src/ipj/percentile_amortize.py; test_amortize_standalone.py pins the numerics.

Window definition (confirmed): order tasks by COMPLETION time; for P_X drop the first/last (100-X)/2%;
window energy = trapezoid integral of board power over [earliest start, latest completion] of the
central tasks, amortized over the X%*N central tasks. Accuracy is over total N (not trimmed).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime

WINDOWS = (100, 90, 80, 50)


def trapezoid_joules(samples: list[tuple[float, float]], start: float, end: float) -> float:
    """Trapezoidal integral of watts over [start, end] -> joules. samples = sorted (epoch, watts)."""
    pts = [(t, w) for t, w in samples if start <= t <= end]
    if len(pts) < 2:
        if samples:                       # fallback: mean power * duration
            return (sum(w for _, w in samples) / len(samples)) * max(0.0, end - start)
        return 0.0
    e = 0.0
    for i in range(1, len(pts)):
        e += 0.5 * (pts[i][1] + pts[i - 1][1]) * (pts[i][0] - pts[i - 1][0])
    return e


def _epoch(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def load_events(jobs_dir: str) -> list[dict]:
    seen, events = set(), []
    for pat in (os.path.join(jobs_dir, "*", "*__*", "result.json"),
                os.path.join(jobs_dir, "*__*", "result.json")):
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
            events.append(dict(
                start=_epoch(d["started_at"]), end=_epoch(d["finished_at"]),
                tin=int(ar.get("n_input_tokens") or 0), tout=int(ar.get("n_output_tokens") or 0),
                resolved=(reward is not None and reward >= 1.0)))
    return events


def load_power(path: str) -> list[tuple[float, float]]:
    s = []
    try:
        for ln in open(path):
            p = ln.split()
            if len(p) >= 2:
                try:
                    s.append((float(p[0]), float(p[1])))
                except ValueError:
                    pass
    except OSError:
        pass
    return sorted(s)


def _pct(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    return sorted_vals[min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))]


def window(events, power, pct, gpu_hourly_rate=None, n_gpus=1):
    evs = sorted(events, key=lambda e: e["end"])
    n = len(evs)
    trim = (100 - pct) / 200.0
    lo = int(n * trim)
    central = evs[lo:n - lo] if pct < 100 and lo > 0 else evs
    nc = len(central)
    if nc == 0:
        return None
    ws = min(e["start"] for e in central)
    we = max(e["end"] for e in central)
    dur = we - ws
    energy = trapezoid_joules(power, ws, we)
    lats = sorted(e["end"] - e["start"] for e in central)
    tin = sum(e["tin"] for e in central)
    tout = sum(e["tout"] for e in central)
    dollars = dpq = None
    if gpu_hourly_rate is not None:
        dollars = (dur / 3600.0) * gpu_hourly_rate * n_gpus
        dpq = dollars / nc
    return dict(
        percentile=pct, n=nc, window_s=dur, energy_j=energy, energy_per_query_j=energy / nc,
        avg_power_w=(energy / dur if dur > 0 else 0.0),
        tokens_in_per_query=tin / nc, tokens_out_per_query=tout / nc,
        decode_throughput_tok_s=(tout / dur if dur > 0 else 0.0),
        latency_mean_s=sum(lats) / nc, latency_p50_s=_pct(lats, 0.5), latency_p90_s=_pct(lats, 0.9),
        dollars=dollars, dollars_per_query=dpq)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs-dir", required=True)
    ap.add_argument("--power", required=True)
    ap.add_argument("--gpu-hourly-rate", type=float, default=None)
    ap.add_argument("--n-gpus", type=int, default=1)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    events = load_events(a.jobs_dir)
    power = load_power(a.power)
    n = len(events)
    resolved = sum(e["resolved"] for e in events)
    acc = resolved / n if n else 0.0
    print(f"[amortize] label={a.label} queries={n} resolved={resolved} "
          f"accuracy(total-N)={acc:.3f} power_samples={len(power)}")
    wins = {}
    for p in WINDOWS:
        m = window(events, power, p, a.gpu_hourly_rate, a.n_gpus)
        if not m:
            continue
        wins[p] = m
        dpq = f"{m['dollars_per_query']:.4f}" if m["dollars_per_query"] is not None else "n/a"
        print(f"  P{p}: n={m['n']} E/q={m['energy_per_query_j']:.0f}J avgP={m['avg_power_w']:.0f}W "
              f"$/q={dpq} lat(mean/p50/p90)={m['latency_mean_s']:.1f}/{m['latency_p50_s']:.1f}/"
              f"{m['latency_p90_s']:.1f}s in/q={m['tokens_in_per_query']:.0f} "
              f"out/q={m['tokens_out_per_query']:.0f} dec={m['decode_throughput_tok_s']:.1f}tok/s")
    if a.out:
        json.dump(dict(label=a.label, n=n, resolved=resolved, accuracy=acc, windows=wins),
                  open(a.out, "w"), indent=2)
        print(f"[amortize] wrote {a.out}")


if __name__ == "__main__":
    main()
