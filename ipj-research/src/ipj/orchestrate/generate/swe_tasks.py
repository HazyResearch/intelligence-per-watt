"""Load ADP *verifiable* SWE tasks and join their source datasets for the runnable Docker env.

ADP (``neulab/agent-data-collection``) records are TRAJECTORY records -- ``instance_id`` + outcome +
``messages`` + a ``patch`` -- NOT task specs. To run a task we need a *triple* per source:

  (i)   an **ADP config loader** yielding (normalized) instance ids;
  (ii)  a **join** to a source task-spec dataset recovering problem/repo/base_commit/tests;
  (iii) a **Docker image-ref scheme**.

Only ``swe-gym`` has a proven triple. ``swe-smith`` and ``nebius`` are wired behind the same
``SweSource`` interface but every concrete join/image guess is isolated in a small, clearly-marked,
independently-testable function flagged ``[UNVERIFIED]`` (a sibling audit resolves them). Nothing
unverified is hardcoded deep in the loaders -- swap the per-source function and the loaders adapt.
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass
from typing import Any, Callable

_log = logging.getLogger(__name__)

_RUN_SUFFIX = re.compile(r"_\d+$")

# `datasets` is an optional heavy dep (not installed in the lite test env); import it lazily so this
# module -- and its de-dup / sampling / image-ref logic -- imports and unit-tests without it.
LoadDataset = Callable[..., Any]


def _default_load_dataset() -> LoadDataset:
    from datasets import load_dataset

    return load_dataset


def _default_log(msg: str) -> None:
    _log.info(msg)


# ======================================================================================
# SweTask -- the runnable, verifier-ready task spec (unchanged public shape)
# ======================================================================================


@dataclass(slots=True, frozen=True)
class SweTask:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    test_patch: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    image_ref: str


def _as_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return [str(x) for x in parsed] if isinstance(parsed, list) else [value]
        except json.JSONDecodeError:
            return [value]
    return []


def _swe_bench_task(rec: dict, image_ref: str) -> SweTask:
    """Map a SWE-bench-schema source record (+ a resolved image ref) into a SweTask."""
    return SweTask(
        instance_id=str(rec["instance_id"]),
        repo=str(rec.get("repo", "")),
        base_commit=str(rec.get("base_commit", "")),
        problem_statement=str(rec.get("problem_statement", "")),
        test_patch=str(rec.get("test_patch", "")),
        fail_to_pass=_as_list(rec.get("FAIL_TO_PASS")),
        pass_to_pass=_as_list(rec.get("PASS_TO_PASS")),
        image_ref=image_ref,
    )


# ======================================================================================
# Per-source concrete schemes -- each a small, independently-testable function
# ======================================================================================

# --- swe-gym (VERIFIED) ---------------------------------------------------------------


def normalize_adp_instance_id(adp_id: str) -> str:
    """ADP swe-gym ids carry a run suffix ('inst_0'); the SWE-Gym source id has none."""
    return _RUN_SUFFIX.sub("", adp_id)


def swe_gym_image_ref(instance_id: str) -> str:
    """Docker image ref for a SWE-Gym instance (VERIFIED against docker.io).

    SWE-Gym's eval images live in the ``xingyaoww`` namespace and normalize the ``__`` separator to
    ``_s_`` (then lowercase): getmoto__moto-7365 -> xingyaoww/sweb.eval.x86_64.getmoto_s_moto-7365.
    This differs from SWE-bench mainline (``swebench`` namespace, ``_1776_`` separator), which does
    not host the SWE-Gym repos. One image per instance -> a per-instance resolver is correct here.
    """
    normalized = instance_id.replace("__", "_s_").lower()
    return f"docker.io/xingyaoww/sweb.eval.x86_64.{normalized}:latest"


def _record_to_swe_task(rec: dict) -> SweTask:
    """swe-gym source record -> SweTask (kept as a named function for back-compat + direct tests)."""
    return _swe_bench_task(rec, swe_gym_image_ref(str(rec["instance_id"])))


# --- swe-smith ([UNVERIFIED] -- sibling audit resolves the source join + image namespace) ----------


def swe_smith_image_ref(rec: dict) -> str:
    """Image ref for a swe-smith bug. [UNVERIFIED namespace/field name.]

    swe-smith bugs are SYNTHETIC and share a **per-repo-environment** image: one image serves MANY
    bugs, so the ref is NOT derivable from the instance id (a swe-gym-style per-instance resolver
    will NOT transfer). We therefore read it straight from the source record's environment-image
    field (``image_name`` in the SWE-bench-family schema); "" if absent (=> treat as build-required
    / not-pullable). The exact field name + namespace are pending the sibling audit.
    """
    for key in ("image_name", "image", "docker_image", "environment"):
        val = rec.get(key)
        if val:
            return str(val)
    return ""


def _record_to_swe_smith_task(rec: dict) -> SweTask:
    return _swe_bench_task(rec, swe_smith_image_ref(rec))


# --- nebius ([UNVERIFIED] -- likely source SWE-bench-Extra; prebuilt images usually absent) --------


def nebius_image_ref(instance_id: str) -> str:  # noqa: ARG001 - id kept for a symmetric interface
    """Image ref for a nebius (SWE-bench-Extra) instance. [UNVERIFIED.]

    SWE-bench-Extra rarely publishes prebuilt eval images, so these are typically BUILD-REQUIRED.
    Default returns "" (=> not-pullable; Gate 1's image audit / a build step handles them). If the
    audit finds a hosted namespace, replace this body with the concrete scheme.
    """
    return ""


def _record_to_nebius_task(rec: dict) -> SweTask:
    return _swe_bench_task(rec, nebius_image_ref(str(rec["instance_id"])))


# ======================================================================================
# SweSource registry -- the pluggable per-source triple
# ======================================================================================


@dataclass(slots=True, frozen=True)
class SweSource:
    """A verifiable ADP config: its trajectory config, its source task-spec join, its image scheme.

    ``normalize_id``   : ADP instance id -> unique task id (strip run-index / traj-hash suffix).
    ``record_to_task`` : source task-spec record -> SweTask (embeds this source's image-ref scheme).
    ``image_ref``      : source task-spec record -> image ref str ("" when build-required/unknown).
    ``verified``       : True only for the proven swe-gym triple; the others are gated.
    """

    name: str
    adp_config_id: str
    adp_split: str
    source_repo: str
    source_split: str
    normalize_id: Callable[[str], str]
    record_to_task: Callable[[dict], SweTask]
    image_ref: Callable[[dict], str]
    verified: bool
    notes: str


SOURCES: dict[str, SweSource] = {
    "swe-gym": SweSource(
        name="swe-gym",
        adp_config_id="swe-gym_openhands_sampled_trajectories",
        adp_split="raw",
        source_repo="SWE-Gym/SWE-Gym",
        source_split="train",
        normalize_id=normalize_adp_instance_id,
        record_to_task=_record_to_swe_task,
        image_ref=lambda rec: swe_gym_image_ref(str(rec["instance_id"])),
        verified=True,
        notes="Proven triple (EXP pilot): xingyaoww per-instance images, SWE-Gym/SWE-Gym source.",
    ),
    "swe-smith": SweSource(
        name="swe-smith",
        adp_config_id="swe-smith",
        adp_split="raw",
        source_repo="SWE-bench/SWE-smith",  # [UNVERIFIED] sibling audit resolves the true source join
        source_split="train",
        normalize_id=normalize_adp_instance_id,  # [UNVERIFIED] suffix scheme (run-index assumed)
        record_to_task=_record_to_swe_smith_task,
        image_ref=swe_smith_image_ref,
        verified=False,
        notes="[UNVERIFIED] per-repo-environment images (one image <-> many synthetic bugs); read "
        "the env image from the source record, not the instance id.",
    ),
    "nebius": SweSource(
        name="nebius",
        adp_config_id="nebius_SWE-agent-trajectories",
        adp_split="raw",
        source_repo="nebius/SWE-bench-extra",  # [UNVERIFIED] sibling audit resolves the true source join
        source_split="train",
        normalize_id=normalize_adp_instance_id,  # [UNVERIFIED] suffix scheme (run-index assumed)
        record_to_task=_record_to_nebius_task,
        image_ref=lambda rec: nebius_image_ref(str(rec["instance_id"])),
        verified=False,
        notes="[UNVERIFIED] source likely SWE-bench-Extra; prebuilt images usually absent "
        "(build-required) -> image_ref defaults to '' until the audit finds a hosted namespace.",
    ),
}


# ======================================================================================
# ADP config loading + source join (datasets.load_dataset injected -> testable offline)
# ======================================================================================


def adp_instance_ids(source: SweSource, n: int, *, load_dataset: LoadDataset | None = None) -> set[str]:
    """Up to ``n`` unique (normalized) instance ids from ``source``'s ADP config."""
    load = load_dataset if load_dataset is not None else _default_load_dataset()
    ds = load("neulab/agent-data-collection", source.adp_config_id, split=source.adp_split, streaming=True)
    out: set[str] = set()
    for rec in ds:
        out.add(source.normalize_id(str(rec["instance_id"])))
        if len(out) >= n:
            break
    return out


def load_source_tasks(
    source: SweSource,
    n: int,
    *,
    instance_filter: set[str] | None = None,
    load_dataset: LoadDataset | None = None,
) -> list[SweTask]:
    """Up to ``n`` runnable SweTasks from ``source``'s task-spec dataset, restricted to a filter set."""
    load = load_dataset if load_dataset is not None else _default_load_dataset()
    ds = load(source.source_repo, split=source.source_split, streaming=True)
    tasks: list[SweTask] = []
    for rec in ds:
        if instance_filter is not None and str(rec["instance_id"]) not in instance_filter:
            continue
        tasks.append(source.record_to_task(rec))
        if len(tasks) >= n:
            break
    return tasks


# --- swe-gym back-compat wrappers (unchanged call sites in gen_smoke.py) ---------------


def adp_swe_gym_instance_ids(n: int, *, load_dataset: LoadDataset | None = None) -> set[str]:
    """Normalized instance ids from ADP's swe-gym slice (the ADP selection layer)."""
    return adp_instance_ids(SOURCES["swe-gym"], n, load_dataset=load_dataset)


def load_swe_gym_tasks(
    n: int, *, instance_filter: set[str] | None = None, load_dataset: LoadDataset | None = None
) -> list[SweTask]:
    """Runnable SWE-Gym source tasks (SWE-bench format), optionally restricted to instance_filter."""
    return load_source_tasks(SOURCES["swe-gym"], n, instance_filter=instance_filter, load_dataset=load_dataset)


# ======================================================================================
# De-dup + unique counts per source
# ======================================================================================


def dedup_instance_ids(raw_ids: Any, normalize: Callable[[str], str]) -> set[str]:
    """Collapse run-index / traj-hash suffixed ids into unique task ids via ``normalize``."""
    return {normalize(str(i)) for i in raw_ids}


def unique_instance_ids(
    source: SweSource, *, load_dataset: LoadDataset | None = None, limit: int | None = None
) -> set[str]:
    """The full set of unique (normalized) instance ids in ``source``'s ADP config (network on default)."""
    load = load_dataset if load_dataset is not None else _default_load_dataset()
    ds = load("neulab/agent-data-collection", source.adp_config_id, split=source.adp_split, streaming=True)
    seen: set[str] = set()
    for i, rec in enumerate(ds):
        seen.add(source.normalize_id(str(rec["instance_id"])))
        if limit is not None and i + 1 >= limit:
            break
    return seen


def unique_counts_by_source(
    sources: dict[str, SweSource] | None = None,
    *,
    load_dataset: LoadDataset | None = None,
    limit: int | None = None,
) -> dict[str, int]:
    """Unique-task count per source after de-dup (the number that sizes the harvest, not traj totals)."""
    srcs = sources if sources is not None else SOURCES
    return {
        name: len(unique_instance_ids(src, load_dataset=load_dataset, limit=limit))
        for name, src in srcs.items()
    }


# ======================================================================================
# Stratified diverse sampling + image-pullable filtering
# ======================================================================================


def _proportional_quotas(tasks_by_source: dict[str, list[SweTask]], total: int | None) -> dict[str, int]:
    """Split ``total`` across sources proportional to availability (largest-remainder; name tie-break)."""
    avail = {name: len(v) for name, v in tasks_by_source.items()}
    grand = sum(avail.values())
    if total is None or total >= grand:
        return dict(avail)
    if grand == 0:
        return {name: 0 for name in avail}
    exact = {name: total * avail[name] / grand for name in avail}
    floors = {name: int(exact[name]) for name in avail}
    remainder = total - sum(floors.values())
    order = sorted(avail, key=lambda name: (-(exact[name] - floors[name]), name))
    for name in order[:remainder]:
        floors[name] += 1
    return floors


def stratified_sample(
    tasks_by_source: dict[str, list[SweTask]],
    *,
    total: int | None = None,
    quotas: dict[str, int] | None = None,
    seed: int = 0,
    log: Callable[[str], None] | None = None,
) -> list[SweTask]:
    """Deterministic diverse sample across sources.

    Provide explicit per-source ``quotas`` OR a ``total`` (split proportionally). Each source is
    shuffled under a per-source seed derived from ``seed`` (so selection is stable + reproducible and
    independent of upstream record order). A source with fewer tasks than its quota is capped and the
    shortfall is logged -- never silently truncated.
    """
    log = log if log is not None else _default_log
    if quotas is None:
        quotas = _proportional_quotas(tasks_by_source, total)

    selected: list[SweTask] = []
    for name in sorted(tasks_by_source):
        pool = sorted(tasks_by_source[name], key=lambda t: t.instance_id)
        random.Random(f"{seed}:{name}").shuffle(pool)
        want = quotas.get(name, 0)
        if want > len(pool):
            log(
                f"stratified_sample: source {name!r} short of quota "
                f"(wanted {want}, have {len(pool)}); took {len(pool)}"
            )
            want = len(pool)
        selected.extend(pool[:want])
    return selected


def filter_pullable(
    tasks: list[SweTask],
    is_pullable: Callable[[str], bool],
    *,
    log: Callable[[str], None] | None = None,
) -> list[SweTask]:
    """Keep tasks whose image is pullable per the injected predicate; log (never drop silently) the rest.

    ``is_pullable`` is injected so this is testable without Docker; in production pass a predicate
    backed by ``docker manifest inspect`` (see gen_smoke). An empty image_ref (build-required) is
    dropped without even consulting the predicate.
    """
    log = log if log is not None else _default_log
    kept: list[SweTask] = []
    for t in tasks:
        if t.image_ref and is_pullable(t.image_ref):
            kept.append(t)
        else:
            reason = "no image (build-required)" if not t.image_ref else "image not pullable"
            log(f"filter_pullable: drop {t.instance_id} ({reason}: {t.image_ref!r})")
    return kept
