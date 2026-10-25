"""Pure helpers for the Track-C serving-efficiency benchmark (prefill/decode throughput +
Joules/token), shared identically across hardware (AMD MI350X vs NVIDIA H100).

The benchmark runs on a GPU box (bash orchestration + a probe client), but all throughput and
energy math lives here so it is unit-tested and identical across backends. Power is sampled as
(epoch_seconds, watts) pairs — `rocm-smi --showpower` on AMD, `nvidia-smi power.draw` on NVIDIA
— and integrated trapezoidally to Joules, the same convention as `ipj.fast_measure`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ipj.fast_measure import trapezoid_joules


@dataclass(slots=True)
class EfficiencyCell:
    model: str
    hardware: str          # "mi350x" | "h100" | ...
    prefill_tok_s: float   # input tokens / time-to-first-token
    decode_tok_s: float    # output tokens / generation time (excl. prefill)
    avg_power_w: float     # mean watts over the measured window
    joules_per_tok: float  # integrated J over the window / tokens produced in it
    mean_latency_s: float  # mean end-to-end request latency
    n_gpus: int = 1


def prefill_tok_s(input_tokens: int, ttft_s: float) -> float:
    """Prefill throughput: input tokens processed per second (TTFT == prefill time)."""
    return input_tokens / ttft_s if ttft_s > 0 else 0.0


def decode_tok_s(output_tokens: int, decode_s: float) -> float:
    """Decode throughput: output tokens generated per second (after the first token)."""
    return output_tokens / decode_s if decode_s > 0 else 0.0


def avg_power_w(power_samples: list[tuple[float, float]], start: float, end: float) -> float:
    """Mean watts over [start, end] = integrated Joules / elapsed seconds."""
    elapsed = end - start
    return trapezoid_joules(power_samples, start, end) / elapsed if elapsed > 0 else 0.0


def joules_per_token(power_samples: list[tuple[float, float]], start: float, end: float,
                     total_tokens: int) -> float:
    """Energy per token = integrated Joules over the window / tokens produced in it."""
    return trapezoid_joules(power_samples, start, end) / total_tokens if total_tokens else 0.0


def build_efficiency_cell(model: str, hardware: str, *, input_tokens: int, ttft_s: float,
                          output_tokens: int, decode_s: float,
                          power_samples: list[tuple[float, float]], win_start: float,
                          win_end: float, mean_latency_s: float, n_gpus: int = 1) -> EfficiencyCell:
    """Assemble the cross-hardware efficiency cell from the probe timings + power samples."""
    total_tok = input_tokens + output_tokens
    return EfficiencyCell(
        model=model, hardware=hardware,
        prefill_tok_s=prefill_tok_s(input_tokens, ttft_s),
        decode_tok_s=decode_tok_s(output_tokens, decode_s),
        avg_power_w=avg_power_w(power_samples, win_start, win_end),
        joules_per_tok=joules_per_token(power_samples, win_start, win_end, total_tok),
        mean_latency_s=mean_latency_s, n_gpus=n_gpus,
    )
