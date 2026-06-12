"""Synthetic fixed-length prompt dataset for systematic profiling sweeps.

Designed for input-length × output-length sweeps without an LLM-as-judge.
Emits ``num_samples`` synthetic prompts whose token count approximates
``prompt_length`` via the rule of thumb 1 token ≈ 4 ASCII characters for
English BPE-style tokenizers.

Tokenization is model-dependent, so the realized length will differ from the
target. Always check the actual ``Input Tokens`` field reported in the
profile summary and adjust ``prompt_length`` if needed.

Usage::

    ipw profile --client mlx --model <id> \\
        --dataset fixed_length \\
        --dataset-param prompt_length=2000 \\
        --dataset-param num_samples=6 \\
        --client-param max_tokens=64

Scoring is intentionally skipped — there is no ground truth — so no eval
client (and no OpenAI API key) is required.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..clients.base import InferenceClient
from ..core.registry import DatasetRegistry
from ..core.types import DatasetRecord
from .base import DatasetProvider

# Self-contained English paragraph used to pad prompts. Diverse word choice
# avoids tokenizer pathologies that would arise from a single repeated token
# (e.g., a degenerate KV-cache pattern).
_SEED = (
    "In the early years of the twentieth century, advances in physics began "
    "to transform our understanding of space, time, and matter. Einstein's "
    "papers of 1905 introduced special relativity, the photoelectric effect, "
    "and Brownian motion. These ideas reshaped the foundations of classical "
    "mechanics and laid the groundwork for quantum theory and general "
    "relativity in the decades that followed. "
)

_INSTRUCTION = "\n\nBriefly summarize the passage above."

# Default chars-per-token. Empirically calibrated against Qwen3 / Llama-3 BPE
# tokenizers on this seed text — 5 chars/token lands within ~5% of target.
# Override per-tokenizer via the ``chars_per_token`` dataset param.
_DEFAULT_CHARS_PER_TOKEN = 5


@DatasetRegistry.register("fixed_length")
class FixedLengthDataset(DatasetProvider):
    """Emit ``num_samples`` synthetic prompts of approximately
    ``prompt_length`` tokens. Skips scoring (no ground truth).

    Dataset params (pass via ``--dataset-param key=value``):
        prompt_length:    target token count. Default 100.
        num_samples:      number of prompts to emit. Default 8.
        chars_per_token:  conversion ratio used to size the prompt body.
                          Default 5 (calibrated for Qwen3/Llama-3 BPE on
                          English text). If realized tokens come in
                          systematically off, bump this up (smaller
                          tokens-per-char) or down.
    """

    dataset_id = "fixed_length"
    dataset_name = "Fixed-Length Synthetic"

    # No LLM-as-judge — there is no ground truth.
    eval_client = None
    eval_base_url = None
    eval_model = None

    def __init__(
        self,
        *,
        prompt_length: Any = 100,
        num_samples: Any = 8,
        chars_per_token: Any = _DEFAULT_CHARS_PER_TOKEN,
    ) -> None:
        L = max(1, int(prompt_length))
        N = max(1, int(num_samples))
        cpt = max(1, int(chars_per_token))
        self._target_length = L
        self._chars_per_token = cpt
        self._records: Tuple[DatasetRecord, ...] = tuple(self._build(L, N, cpt))

    def _build(
        self, target_tokens: int, num_samples: int, chars_per_token: int
    ) -> Iterable[DatasetRecord]:
        # Reserve characters for the trailing instruction.
        target_chars = max(1, target_tokens * chars_per_token - len(_INSTRUCTION))
        if target_chars < len(_SEED):
            body = _SEED[:target_chars]
        else:
            reps = (target_chars // len(_SEED)) + 1
            body = (_SEED * reps)[:target_chars]
        prompt = body + _INSTRUCTION

        for i in range(num_samples):
            yield DatasetRecord(
                problem=prompt,
                answer="",
                subject="synthetic",
                dataset_metadata={
                    "target_token_length": target_tokens,
                    "chars_per_token": chars_per_token,
                    "sample_id": i,
                },
            )

    def iter_records(self) -> Iterable[DatasetRecord]:
        return iter(self._records)

    def size(self) -> int:
        return len(self._records)

    def verify_requirements(self) -> List[str]:
        return []  # no API key, no scoring backend

    def score(
        self,
        record: DatasetRecord,
        response: str,
        *,
        eval_client: Optional[InferenceClient] = None,
    ) -> Tuple[Optional[bool], Dict[str, object]]:
        return None, {
            "skipped": "fixed_length is a synthetic profiling dataset with no ground truth",
        }


__all__ = ["FixedLengthDataset"]
