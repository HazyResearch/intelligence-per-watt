from __future__ import annotations

import pytest

from ipj.orchestrate.generate.swe_tasks import (
    SOURCES,
    SweSource,
    SweTask,
    _record_to_swe_task,
    adp_instance_ids,
    dedup_instance_ids,
    filter_pullable,
    load_source_tasks,
    load_swe_gym_tasks,
    nebius_image_ref,
    normalize_adp_instance_id,
    stratified_sample,
    swe_gym_image_ref,
    swe_smith_image_ref,
    unique_counts_by_source,
)

# --------------------------------------------------------------------------------------
# swe-gym (verified path) -- back-compatible surface
# --------------------------------------------------------------------------------------


def test_normalize_strips_adp_run_suffix():
    assert normalize_adp_instance_id("getmoto__moto-5321_0") == "getmoto__moto-5321"
    assert normalize_adp_instance_id("getmoto__moto-5321") == "getmoto__moto-5321"
    assert normalize_adp_instance_id("astropy__astropy-1234_12") == "astropy__astropy-1234"


def test_image_ref_is_deterministic_and_instance_scoped():
    ref = swe_gym_image_ref("getmoto__moto-7365")
    assert "xingyaoww/sweb.eval.x86_64.getmoto_s_moto-7365" in ref  # __ -> _s_, xingyaoww namespace
    assert "getmoto__moto-7365" not in ref  # the raw '__' form must be normalized away
    assert ref == swe_gym_image_ref("getmoto__moto-7365")  # deterministic


def test_record_to_swe_task_maps_swe_bench_fields():
    rec = {
        "instance_id": "getmoto__moto-7365",
        "repo": "getmoto/moto",
        "base_commit": "7f6c9cb1",
        "problem_statement": "fix the bug",
        "test_patch": "diff --git a/t b/t",
        "FAIL_TO_PASS": ["tests/test_x.py::test_a"],
        "PASS_TO_PASS": ["tests/test_x.py::test_b"],
        "version": "5.0",
    }
    t = _record_to_swe_task(rec)
    assert isinstance(t, SweTask)
    assert t.instance_id == "getmoto__moto-7365"
    assert t.repo == "getmoto/moto" and t.base_commit == "7f6c9cb1"
    assert t.fail_to_pass == ["tests/test_x.py::test_a"]
    assert t.pass_to_pass == ["tests/test_x.py::test_b"]
    assert t.image_ref  # populated


def test_record_to_swe_task_parses_json_string_test_lists():
    # SWE-Gym sometimes stores FAIL_TO_PASS/PASS_TO_PASS as JSON strings
    rec = {
        "instance_id": "a__b-1", "repo": "a/b", "base_commit": "c", "problem_statement": "p",
        "test_patch": "d", "FAIL_TO_PASS": '["t::a"]', "PASS_TO_PASS": '["t::b"]',
    }
    t = _record_to_swe_task(rec)
    assert t.fail_to_pass == ["t::a"] and t.pass_to_pass == ["t::b"]


# --------------------------------------------------------------------------------------
# Pluggable per-source registry: the three verifiable ADP configs
# --------------------------------------------------------------------------------------


def test_registry_declares_the_three_verifiable_configs_with_exact_ids():
    assert set(SOURCES) == {"swe-gym", "swe-smith", "nebius"}
    ids = {s.name: s.adp_config_id for s in SOURCES.values()}
    assert ids["swe-gym"] == "swe-gym_openhands_sampled_trajectories"
    assert ids["swe-smith"] == "swe-smith"
    assert ids["nebius"] == "nebius_SWE-agent-trajectories"
    # swe-gym is the only proven source; the others are explicitly flagged unverified.
    assert SOURCES["swe-gym"].verified is True
    assert SOURCES["swe-smith"].verified is False
    assert SOURCES["nebius"].verified is False


def test_swe_gym_source_maps_record_and_image():
    src = SOURCES["swe-gym"]
    rec = {"instance_id": "getmoto__moto-7365", "repo": "getmoto/moto", "base_commit": "abc",
           "problem_statement": "p", "test_patch": "d", "FAIL_TO_PASS": ["t::a"], "PASS_TO_PASS": []}
    t = src.record_to_task(rec)
    assert t.instance_id == "getmoto__moto-7365" and t.repo == "getmoto/moto"
    assert "xingyaoww/sweb.eval.x86_64.getmoto_s_moto-7365" in t.image_ref


def test_swe_smith_image_ref_is_per_repo_environment_from_record_field():
    # swe-smith bugs are synthetic: one env image serves MANY bugs, so the image is NOT derivable
    # from the instance id -- it is read from the source record's environment field.
    rec = {"instance_id": "pallets__flask.abc123.func_pm_1", "repo": "pallets/flask",
           "image_name": "jyangballin/swesmith.x86_64.pallets_s_flask:latest",
           "problem_statement": "p", "FAIL_TO_PASS": ["t::a"], "PASS_TO_PASS": []}
    assert swe_smith_image_ref(rec) == "jyangballin/swesmith.x86_64.pallets_s_flask:latest"
    # A record with no environment image field yields "" (build-required / unresolved).
    assert swe_smith_image_ref({"instance_id": "x", "repo": "r"}) == ""
    # The mapped SweTask carries the env image straight through.
    t = SOURCES["swe-smith"].record_to_task(rec)
    assert t.image_ref == "jyangballin/swesmith.x86_64.pallets_s_flask:latest"
    assert t.instance_id == "pallets__flask.abc123.func_pm_1"


def test_nebius_images_are_build_required_by_default():
    # SWE-bench-Extra rarely ships prebuilt images: default resolver returns "" (build-required).
    assert nebius_image_ref("nebius__proj-42") == ""
    rec = {"instance_id": "nebius__proj-42", "repo": "n/proj", "base_commit": "c",
           "problem_statement": "p", "test_patch": "d", "FAIL_TO_PASS": ["t::a"], "PASS_TO_PASS": []}
    t = SOURCES["nebius"].record_to_task(rec)
    assert t.image_ref == "" and t.repo == "n/proj"


# --------------------------------------------------------------------------------------
# ADP loading + join (datasets.load_dataset injected -> no network)
# --------------------------------------------------------------------------------------


def _fake_loader(by_call):
    """Build a fake load_dataset. `by_call` maps a match-key -> list[record]."""

    def _load(path, *args, **kwargs):
        # ADP config load: load_dataset("neulab/agent-data-collection", <config>, split=..., streaming=True)
        if path == "neulab/agent-data-collection":
            return list(by_call[args[0]])
        return list(by_call[path])

    return _load


def test_adp_instance_ids_normalizes_and_dedups_via_injected_loader():
    adp = [
        {"instance_id": "getmoto__moto-1_0"},
        {"instance_id": "getmoto__moto-1_1"},   # same task, different run index
        {"instance_id": "astropy__astropy-9_3"},
    ]
    loader = _fake_loader({"swe-gym_openhands_sampled_trajectories": adp})
    ids = adp_instance_ids(SOURCES["swe-gym"], n=100, load_dataset=loader)
    assert ids == {"getmoto__moto-1", "astropy__astropy-9"}


def test_load_source_tasks_joins_source_and_filters_instances():
    source_records = [
        {"instance_id": "getmoto__moto-1", "repo": "getmoto/moto", "base_commit": "c1",
         "problem_statement": "p1", "test_patch": "d1", "FAIL_TO_PASS": ["t::a"], "PASS_TO_PASS": []},
        {"instance_id": "other__repo-2", "repo": "o/r", "base_commit": "c2",
         "problem_statement": "p2", "test_patch": "d2", "FAIL_TO_PASS": ["t::b"], "PASS_TO_PASS": []},
    ]
    loader = _fake_loader({"SWE-Gym/SWE-Gym": source_records})
    tasks = load_source_tasks(SOURCES["swe-gym"], n=100, instance_filter={"getmoto__moto-1"}, load_dataset=loader)
    assert [t.instance_id for t in tasks] == ["getmoto__moto-1"]
    assert tasks[0].base_commit == "c1" and tasks[0].fail_to_pass == ["t::a"]


def test_load_swe_gym_tasks_backcompat_delegates(monkeypatch):
    source_records = [
        {"instance_id": "getmoto__moto-1", "repo": "getmoto/moto", "base_commit": "c1",
         "problem_statement": "p1", "test_patch": "d1", "FAIL_TO_PASS": ["t::a"], "PASS_TO_PASS": []},
    ]
    loader = _fake_loader({"SWE-Gym/SWE-Gym": source_records})
    tasks = load_swe_gym_tasks(5, load_dataset=loader)
    assert [t.instance_id for t in tasks] == ["getmoto__moto-1"]


# --------------------------------------------------------------------------------------
# De-dup counts per source
# --------------------------------------------------------------------------------------


def test_dedup_instance_ids_collapses_run_suffixes():
    raw = ["a__b-1_0", "a__b-1_1", "a__b-1_2", "c__d-9_0"]
    assert dedup_instance_ids(raw, normalize_adp_instance_id) == {"a__b-1", "c__d-9"}


def test_unique_counts_by_source_reports_per_source_unique():
    swe_gym_adp = [{"instance_id": "a__b-1_0"}, {"instance_id": "a__b-1_1"}, {"instance_id": "c__d-2_0"}]
    swe_smith_adp = [{"instance_id": "x__y.h1_0"}, {"instance_id": "x__y.h1_0"}]  # exact dup
    nebius_adp = [{"instance_id": "n__p-1_0"}]
    loader = _fake_loader({
        "swe-gym_openhands_sampled_trajectories": swe_gym_adp,
        "swe-smith": swe_smith_adp,
        "nebius_SWE-agent-trajectories": nebius_adp,
    })
    counts = unique_counts_by_source(load_dataset=loader)
    assert counts == {"swe-gym": 2, "swe-smith": 1, "nebius": 1}


# --------------------------------------------------------------------------------------
# Stratified diverse sampling + pullable filtering
# --------------------------------------------------------------------------------------


def _mktask(iid, image="img"):
    return SweTask(iid, "r", "c", "p", "d", ["t::a"], [], image)


def test_stratified_sample_hits_explicit_quotas_and_is_deterministic():
    by_source = {
        "swe-gym": [_mktask(f"g{i}") for i in range(10)],
        "swe-smith": [_mktask(f"s{i}") for i in range(10)],
        "nebius": [_mktask(f"n{i}") for i in range(10)],
    }
    quotas = {"swe-gym": 3, "swe-smith": 2, "nebius": 1}
    a = stratified_sample(by_source, quotas=quotas, seed=7)
    b = stratified_sample(by_source, quotas=quotas, seed=7)
    assert [t.instance_id for t in a] == [t.instance_id for t in b]  # deterministic
    per = {p: sum(1 for t in a if t.instance_id.startswith(p[0])) for p in ("g", "s", "n")}
    assert per == {"g": 3, "s": 2, "n": 1}
    # A different seed keeps the quota counts but reshuffles which items land in the sample.
    c = stratified_sample(by_source, quotas=quotas, seed=8)
    assert sum(1 for t in c if t.instance_id.startswith("g")) == 3
    assert [t.instance_id for t in a] != [t.instance_id for t in c]


def test_stratified_sample_proportional_total():
    by_source = {
        "swe-gym": [_mktask(f"g{i}") for i in range(80)],
        "swe-smith": [_mktask(f"s{i}") for i in range(20)],
    }
    sel = stratified_sample(by_source, total=10, seed=1)
    counts = {"g": 0, "s": 0}
    for t in sel:
        counts[t.instance_id[0]] += 1
    assert sum(counts.values()) == 10
    assert counts == {"g": 8, "s": 2}  # 80:20 -> 8:2


def test_stratified_sample_logs_shortfall(caplog):
    logged: list[str] = []
    by_source = {"swe-gym": [_mktask("g0")], "swe-smith": [_mktask("s0"), _mktask("s1")]}
    sel = stratified_sample(by_source, quotas={"swe-gym": 5, "swe-smith": 1}, seed=0, log=logged.append)
    assert len([t for t in sel if t.instance_id.startswith("g")]) == 1  # capped to available
    assert any("swe-gym" in m and "short" in m for m in logged)


def test_filter_pullable_uses_injected_predicate_and_logs_drops():
    tasks = [_mktask("a", "img-a"), _mktask("b", ""), _mktask("c", "img-c")]
    logged: list[str] = []

    def is_pullable(ref: str) -> bool:
        return ref == "img-a"

    kept = filter_pullable(tasks, is_pullable, log=logged.append)
    assert [t.instance_id for t in kept] == ["a"]
    assert any("b" in m for m in logged) and any("c" in m for m in logged)  # both drops logged


def test_swe_source_is_frozen_dataclass():
    with pytest.raises(Exception):
        SOURCES["swe-gym"].name = "x"  # frozen
    assert isinstance(SOURCES["swe-gym"], SweSource)
