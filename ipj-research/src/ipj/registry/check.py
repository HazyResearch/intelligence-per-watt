from __future__ import annotations

from dataclasses import dataclass, field

from .record import RunRecord


@dataclass(slots=True)
class Verdict:
    tier: str  # exact | accuracy | near | none
    matches: list[dict] = field(default_factory=list)
    message: str = ""


def _authoritative(rows: list[dict]) -> list[dict]:
    return [
        r for r in rows
        if not r.get("git_dirty", False) and r.get("status", "ok") == "ok"
    ]


def check_spec(rec: RunRecord, index: list[dict]) -> Verdict:
    rows = _authoritative(index)
    exact = [r for r in rows if r.get("energy_sig") == rec.energy_sig]
    if exact:
        r = exact[0]
        acc = r.get("metrics", {}).get("accuracy")
        return Verdict("exact", exact,
                       f"EXACT match: run @ {r.get('energy_sig')} (acc={acc}, {r.get('run_dir')}). Skip. "
                       "NOTE: sampling & serve config (temperature, top_p, seed, etc.) are not captured — "
                       "verify your run config matches before skipping.")
    acc_only = [r for r in rows if r.get("accuracy_sig") == rec.accuracy_sig]
    if acc_only:
        r = acc_only[0]
        acc = r.get("metrics", {}).get("accuracy")
        hw = r.get("hardware", {}).get("gpu")
        return Verdict("accuracy", acc_only,
                       f"ACCURACY known (acc={acc} on {hw}); energy on your HW is new. "
                       "NOTE: sampling & serve config (temperature, top_p, seed, etc.) are not captured — "
                       "verify your run config matches before reusing these results.")
    near = [r for r in rows
            if r.get("model", {}).get("id") == rec.model.id
            and r.get("benchmark", {}).get("name") == rec.benchmark.name]
    if near:
        return Verdict("near", near,
                       f"{len(near)} near-match(es): same model+benchmark, differing config — review the diff.")
    return Verdict("none", [], "No prior run found.")
