"""Tests for the Apple Foundation Models client.

Split in two: the helpers in ``ipw.clients._afm_support`` carry no SDK import and
run everywhere (including Linux CI), while the live tests are double-gated on the
SDK being importable and on ``IPW_AFM_LIVE`` being set, mirroring
``test_mlx_client.py``. The SDK only installs on an Apple Silicon Mac with a full
Xcode, so CI must skip rather than fail.
"""

from __future__ import annotations

import os

import pytest

from ipw.clients._afm_support import (
    MODEL_LABELS,
    SnapshotAccumulator,
    build_options_kwargs,
    coerce_scalar,
    parse_sampling_spec,
    snapshot_deltas,
    validate_model_label,
)


class TestSnapshotAccumulator:
    """``stream_response`` yields the full text so far, not the new piece.

    Appending each chunk -- the pattern that is correct for
    ``mlx_lm.stream_generate`` -- would duplicate text quadratically here, so
    these are the highest-value tests in the file.
    """

    def test_append_only_stream_yields_deltas(self) -> None:
        snapshots = ["Three", "Three primary", "Three primary colors"]
        assert snapshot_deltas(snapshots) == ["Three", " primary", " colors"]

    def test_reassembled_deltas_equal_final_snapshot(self) -> None:
        snapshots = ["a", "ab", "abc", "abcd"]
        assert "".join(snapshot_deltas(snapshots)) == snapshots[-1]

    def test_final_text_is_the_last_snapshot(self) -> None:
        accumulator = SnapshotAccumulator()
        for snapshot in ["Red", "Red and blue"]:
            accumulator.add(snapshot)
        assert accumulator.text == "Red and blue"

    def test_repeated_snapshot_yields_no_delta(self) -> None:
        accumulator = SnapshotAccumulator()
        assert accumulator.add("same") == "same"
        assert accumulator.add("same") == ""

    def test_revised_snapshot_is_reported_whole(self) -> None:
        # Guided generation can replace a snapshot rather than extend it.
        accumulator = SnapshotAccumulator()
        accumulator.add('{"color": "re')
        assert accumulator.add('{"colour": "red"}') == '{"colour": "red"}'
        assert accumulator.text == '{"colour": "red"}'

    def test_empty_stream(self) -> None:
        assert snapshot_deltas([]) == []

    def test_none_snapshot_ignored(self) -> None:
        accumulator = SnapshotAccumulator()
        assert accumulator.add(None) == ""
        assert accumulator.text == ""

    def test_empty_first_snapshot_yields_no_delta(self) -> None:
        assert snapshot_deltas(["", "hi"]) == ["hi"]


class TestValidateModelLabel:
    @pytest.mark.parametrize("label", MODEL_LABELS)
    def test_accepts_known_labels(self, label: str) -> None:
        assert validate_model_label(label) == label

    def test_normalizes_case_and_whitespace(self) -> None:
        assert validate_model_label("  AFM-3-Core-Advanced ") == "afm-3-core-advanced"

    @pytest.mark.parametrize("label", ["afm-4", "afm3", "gpt-4", "", "afm-3-pro"])
    def test_rejects_unknown_labels(self, label: str) -> None:
        with pytest.raises(ValueError, match="Unknown AFM model label"):
            validate_model_label(label)


class TestBuildOptionsKwargs:
    def test_max_tokens_maps_to_maximum_response_tokens(self) -> None:
        assert build_options_kwargs({"max_tokens": 128}) == {
            "maximum_response_tokens": 128
        }

    def test_client_param_strings_are_coerced(self) -> None:
        # --client-param values always arrive as strings.
        options = build_options_kwargs({"max_tokens": "256", "temperature": "0.7"})
        assert options == {"maximum_response_tokens": 256, "temperature": 0.7}
        assert isinstance(options["maximum_response_tokens"], int)

    def test_unset_values_are_omitted_so_sdk_defaults_apply(self) -> None:
        assert build_options_kwargs({}) == {}
        assert build_options_kwargs({"max_tokens": None, "temperature": None}) == {}

    def test_unrelated_params_ignored(self) -> None:
        assert build_options_kwargs({"instructions": "hi", "opts": "x"}) == {}


class TestParseSamplingSpec:
    def test_defaults_to_greedy_for_reproducibility(self) -> None:
        assert parse_sampling_spec({}) == ("greedy", {})

    def test_random_collects_its_fields(self) -> None:
        mode, kwargs = parse_sampling_spec(
            {"sampling": "random", "top": "50", "seed": "7"}
        )
        assert mode == "random"
        assert kwargs == {"top": 50, "seed": 7}

    def test_greedy_ignores_random_fields(self) -> None:
        assert parse_sampling_spec({"sampling": "greedy", "top": "50"}) == ("greedy", {})

    def test_rejects_unknown_mode(self) -> None:
        with pytest.raises(ValueError, match="Unknown AFM sampling mode"):
            parse_sampling_spec({"sampling": "beam"})


class TestCoerceScalar:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("128", 128),
            ("0.7", 0.7),
            ("1e-3", 0.001),
            ("true", True),
            ("False", False),
            ("afm-3", "afm-3"),
            (42, 42),
            (None, None),
        ],
    )
    def test_coercion(self, raw: object, expected: object) -> None:
        assert coerce_scalar(raw) == expected


# --------------------------------------------------------------------------
# Live tests: require the SDK plus an opt-in env var so CI never runs them.
# --------------------------------------------------------------------------

pytest.importorskip(
    "apple_fm_sdk",
    reason="AFM tests require the optional 'afm' extra (Apple Silicon, macOS 26+, full Xcode)",
)

if not os.environ.get("IPW_AFM_LIVE"):
    pytest.skip(
        "Set IPW_AFM_LIVE=1 to run live Apple Foundation Models tests",
        allow_module_level=True,
    )

from ipw.clients.afm import AFMClient  # noqa: E402

MODEL = os.environ.get("IPW_AFM_MODEL", "afm-3-core-advanced")


@pytest.fixture(scope="module")
def client():
    """One prepared client for the module; loading assets is slow (~10s cold)."""
    instance = AFMClient()
    if not instance.health():
        instance.close()
        pytest.skip("Apple Foundation Models is unavailable on this host")
    instance.prepare(MODEL)
    try:
        yield instance
    finally:
        instance.close()


@pytest.mark.apple
class TestAFMClientLive:
    def test_metadata(self, client: AFMClient) -> None:
        assert client.base_url == "in-process"
        assert client.list_models() == list(MODEL_LABELS)
        info = client.describe()
        assert info["model_label"] == MODEL
        # 4096 today; assert the shape, not the constant.
        assert isinstance(info["afm_context_size"], int)
        assert info["afm_context_size"] > 0
        assert info["afm_sdk_version"]
        # Must name the part ("Apple M1 Pro"), not platform.processor()'s bare
        # "arm" -- attributing a run to a device is the whole point of the field.
        assert info["host_chip"].startswith("Apple ")
        assert info["host_chip"] not in ("arm", "arm64")

    def test_generation_populates_response(self, client: AFMClient) -> None:
        response = client.stream_chat_completion(
            MODEL, "Name three primary colors.", max_tokens=48
        )

        assert response.content.strip()
        assert response.usage.prompt_tokens and response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens and response.usage.completion_tokens > 0
        assert response.usage.total_tokens == (
            response.usage.prompt_tokens + response.usage.completion_tokens
        )
        # Time-to-first-*chunk*, since snapshots batch several tokens.
        assert response.time_to_first_token_ms > 0
        assert response.token_timestamps
        # Wall-clock bounds must be set so phase energy split is accurate.
        assert response.request_end_time > response.request_start_time > 0

    def test_no_text_duplication_across_snapshots(self, client: AFMClient) -> None:
        """Mishandling cumulative snapshots as deltas repeats the text's prefix."""
        response = client.stream_chat_completion(
            MODEL, "Count from one to five.", max_tokens=48
        )
        content = response.content
        assert len(content) > 20, "need a non-trivial response to detect duplication"
        # The first quarter of a correctly assembled response should not recur
        # later in it; under the append-every-chunk bug it always does.
        prefix = content[: len(content) // 4]
        assert prefix not in content[len(prefix) :]

    def test_max_tokens_from_constructor_is_honored(self) -> None:
        """Regression: --client-param goes to the constructor, not the call.

        ProfilerRunner passes client params to ``__init__`` and never forwards
        per-call params, so the client must merge ``self._config``.
        """
        instance = AFMClient(max_tokens="16")
        if not instance.health():
            instance.close()
            pytest.skip("Apple Foundation Models is unavailable on this host")
        try:
            instance.prepare(MODEL)
            short = instance.stream_chat_completion(MODEL, "Describe the ocean.")
            assert short.usage.completion_tokens is not None
            assert short.usage.completion_tokens <= 24  # 16 + framing slack
        finally:
            instance.close()

    def test_context_overflow_returns_empty_response(self, client: AFMClient) -> None:
        """Must not raise: ProfilerRunner would abort the whole run."""
        response = client.stream_chat_completion(
            MODEL, "word " * 6000, max_tokens=16
        )
        assert response.content == ""
        assert response.usage.completion_tokens is None
        assert client.describe()["skipped_queries"]

    def test_rejects_unknown_model_label(self, client: AFMClient) -> None:
        with pytest.raises(ValueError, match="Unknown AFM model label"):
            client.prepare("afm-9-ultra")
