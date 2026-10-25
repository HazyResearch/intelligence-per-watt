"""Overlay of IPJ-side additions to the inherited IPW pricing tables.

Keeping IPJ-side model entries here (not in ``intelligence-per-watt/src/ipw/cost/pricing.py``)
ensures the substrate file stays byte-identical with upstream, so an accidental cherry-pick
back to public IPW cannot leak the private baseline model name(s).

Importing this module (or the ``ipj`` package, which imports it) is enough to make the
extra models resolvable via ``ipw.cost.pricing.calculate_cost``.
"""

from __future__ import annotations

try:
    from ipw.cost.pricing import ANTHROPIC_PRICING, GEMINI_PRICING, OPENAI_PRICING
except ImportError:
    # No IPW venv present (e.g., running ipj.rigor tests in a bare venv). The overlay is a
    # no-op — nothing in ipj/ needs the pricing tables at import time. When code actually calls
    # into `ipw.cost.pricing.calculate_cost`, `ipw` will be importable and the overlay is
    # re-applied by re-importing this module (or, in practice, by installing the extra).
    ANTHROPIC_PRICING = GEMINI_PRICING = OPENAI_PRICING = None  # type: ignore[assignment]


_OPENAI_EXTRA: dict[str, dict[str, float]] = {
    "gpt-5.5": {"input": 5.00, "output": 30.00},
    "gpt-5.5-2026-04-23": {"input": 5.00, "output": 30.00},
    "gpt-5.5-pro": {"input": 30.00, "output": 180.00},
    "gpt-5.4-mini": {"input": 0.25, "output": 2.00},
}

_ANTHROPIC_EXTRA: dict[str, dict[str, float]] = {
    "claude-opus-4-8": {"input": 5.00, "output": 25.00},
    "claude-opus-4-7": {"input": 5.00, "output": 25.00},
    "claude-fable-5": {"input": 10.00, "output": 50.00},
}

_GEMINI_EXTRA: dict[str, dict[str, float]] = {
    "gemini-3.5-flash": {"input": 1.50, "output": 9.00},
}

if OPENAI_PRICING is not None:
    for _key, _val in _OPENAI_EXTRA.items():
        OPENAI_PRICING.setdefault(_key, _val)
    for _key, _val in _ANTHROPIC_EXTRA.items():
        ANTHROPIC_PRICING.setdefault(_key, _val)
    for _key, _val in _GEMINI_EXTRA.items():
        GEMINI_PRICING.setdefault(_key, _val)
    del _key, _val
