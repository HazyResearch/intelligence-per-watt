"""Pure helpers for the Apple Foundation Models client.

Deliberately free of any ``apple_fm_sdk`` import so this logic stays testable on
CI runners that cannot install the SDK (it builds Swift bindings and needs a
full Xcode on an Apple Silicon Mac).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

# ``--model`` is required by ``ipw profile`` but the SDK exposes no variant
# selector: the Foundation Models framework's dynamic profile picks AFM 3 Core
# (dense 3B) or AFM 3 Core Advanced (20B sparse MoE) from the host device's
# capabilities. These labels therefore only tag the run -- they do not steer
# which model executes.
MODEL_LABELS: Tuple[str, ...] = ("afm-3", "afm-3-core", "afm-3-core-advanced")

# Sampling modes exposed by fm.SamplingMode as constructors. ``top`` and
# ``probability_threshold`` are fields of the random mode, not modes themselves.
SAMPLING_MODES: Tuple[str, ...] = ("greedy", "random")

_RANDOM_SAMPLING_FIELDS: Tuple[str, ...] = ("top", "probability_threshold", "seed")


def validate_model_label(model: str) -> str:
    """Normalize and check the ``--model`` label.

    Raises ``ValueError`` naming the accepted labels, since a typo would
    otherwise silently mislabel a run's output directory and FLOPs lookup.
    """
    label = (model or "").strip().lower()
    if label not in MODEL_LABELS:
        raise ValueError(
            f"Unknown AFM model label {model!r}. Expected one of: "
            f"{', '.join(MODEL_LABELS)}. Note that these are run labels only -- "
            "the on-device variant is chosen by the framework's dynamic profile, "
            "not by this flag."
        )
    return label


def coerce_scalar(value: Any) -> Any:
    """Coerce a ``--client-param`` string into bool/int/float when it looks like one."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        if "." in text or "e" in lowered:
            return float(text)
        return int(text)
    except ValueError:
        return text


class SnapshotAccumulator:
    """Turn ``session.stream_response``'s cumulative snapshots into deltas.

    The SDK yields the *entire* text generated so far on each iteration, not the
    newly added piece. Appending each chunk (the pattern the MLX client uses for
    ``mlx_lm.stream_generate``) would duplicate text quadratically here.

    ``add`` returns the newly arrived text, or ``""`` when a snapshot repeats.
    Guided generation can revise a snapshot rather than extend it, so a snapshot
    that does not continue the previous one is treated as a replacement and
    reported whole.
    """

    __slots__ = ("text",)

    def __init__(self) -> None:
        self.text: str = ""

    def add(self, snapshot: Any) -> str:
        if snapshot is None:
            return ""
        if not isinstance(snapshot, str):
            # Defensive: tolerate a chunk object exposing text via an attribute.
            snapshot = getattr(snapshot, "content", None) or str(snapshot)
        if snapshot == self.text:
            return ""
        if snapshot.startswith(self.text):
            delta = snapshot[len(self.text) :]
        else:
            delta = snapshot
        self.text = snapshot
        return delta


def snapshot_deltas(snapshots) -> List[str]:
    """Eager form of :class:`SnapshotAccumulator`, for tests and offline use."""
    accumulator = SnapshotAccumulator()
    return [delta for delta in (accumulator.add(s) for s in snapshots) if delta]


def build_options_kwargs(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Map IPW client params onto ``fm.GenerationOptions`` fields.

    Only non-``None`` values are returned so the SDK's own defaults apply for
    anything the caller did not set. ``max_tokens`` is IPW's cross-backend name
    for what the SDK calls ``maximum_response_tokens``.
    """
    options: Dict[str, Any] = {}

    max_tokens = params.get("max_tokens")
    if max_tokens is not None:
        options["maximum_response_tokens"] = int(coerce_scalar(max_tokens))

    temperature = params.get("temperature")
    if temperature is not None:
        options["temperature"] = float(coerce_scalar(temperature))

    return options


def parse_sampling_spec(params: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Resolve the sampling mode name and its keyword arguments.

    Defaults to ``greedy`` so repeated profiling runs are reproducible; the SDK
    itself defaults to random sampling. This mirrors the MLX client, which
    treats an unset temperature as greedy argmax.
    """
    mode = str(params.get("sampling", "greedy")).strip().lower()
    if mode not in SAMPLING_MODES:
        raise ValueError(
            f"Unknown AFM sampling mode {mode!r}. Expected one of: "
            f"{', '.join(SAMPLING_MODES)}."
        )

    kwargs: Dict[str, Any] = {}
    if mode == "random":
        for field in _RANDOM_SAMPLING_FIELDS:
            if params.get(field) is not None:
                kwargs[field] = coerce_scalar(params[field])
    return mode, kwargs


__all__ = [
    "MODEL_LABELS",
    "SAMPLING_MODES",
    "SnapshotAccumulator",
    "build_options_kwargs",
    "coerce_scalar",
    "parse_sampling_spec",
    "snapshot_deltas",
    "validate_model_label",
]
