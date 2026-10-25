from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(slots=True)
class TrainConfig:
    """All knobs for one LoRA SFT run; presets pick the base model + sizes, real run = a preset swap."""

    base_model: str
    data_path: str = "runs/sft/dummy/sft.jsonl"
    out_dir: str = "runs/sft/adapter"
    max_seq_len: int = 1024
    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    # Attention projection names shared by Qwen3.5 and Gemma4 (HF transformers); the smoke confirms LoRA
    # attaches. If a future family uses different names, override per-preset (or set "all-linear").
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    # training
    lr: float = 2e-4
    epochs: float = 1.0
    per_device_batch_size: int = 1
    grad_accum: int = 8
    bf16: bool = True
    gradient_checkpointing: bool = False  # trade compute for memory — needed to fit a 27B+ on one 80GB GPU
    seed: int = 0
    val_frac: float = 0.1


def pilot() -> TrainConfig:
    """Smallest dense Qwen3.5 — the mechanics smoke / DoD."""
    return TrainConfig(base_model="Qwen/Qwen3.5-0.8B")


def pilot_gemma() -> TrainConfig:
    """Smallest full Gemma4 instruct — validates the Gemma render/template path."""
    return TrainConfig(base_model="google/gemma-4-E2B-it")


def real() -> TrainConfig:
    """The real pilot fine-tune target (swap base_model to google/gemma-4-31B-it for the Gemma run)."""
    return TrainConfig(
        base_model="Qwen/Qwen3.5-27B",
        data_path="runs/sft/real/sft.jsonl",
        lora_r=16,
        lora_alpha=32,
        grad_accum=16,
    )


PRESETS: dict[str, Callable[[], TrainConfig]] = {"pilot": pilot, "pilot_gemma": pilot_gemma, "real": real}


def build_config(preset: str, **overrides) -> TrainConfig:
    """Build a TrainConfig from a preset name + non-None field overrides (used by the CLI)."""
    if preset not in PRESETS:
        raise SystemExit(f"unknown preset {preset!r}; choices: {sorted(PRESETS)}")
    config = PRESETS[preset]()
    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)
    return config
