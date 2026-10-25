"""LocalAdapterChatClient: run a base model + LoRA adapter as a ChatClient, with per-call energy.

Heavy imports (torch/transformers/peft) are lazy and live only in the real generate path; inject
``_generate`` to exercise the wrapping (prompt render, token counts, energy, dollars) torch-free.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from .brain import ChatResponse
from .dispatch import AcceleratorEnergyMeasurement, normalize_energy_measurement
from .prompt import render_message

# (content, n_prompt_tokens, n_new_tokens)
GenerateFn = Callable[[str], "tuple[str, int, int]"]
CostEstimator = Callable[[int, int, float], "float | None"]


class LocalAdapterChatClient:
    def __init__(
        self,
        base_model: str,
        adapter_dir: str | None = None,
        *,
        max_new_tokens: int = 512,
        energy_probe: Callable[
            [float, float],
            "AcceleratorEnergyMeasurement | float | None",
        ]
        | None = None,
        cost_estimator: CostEstimator | None = None,
        clock: Callable[[], float] = time.time,
        _generate: GenerateFn | None = None,
    ) -> None:
        self.base_model = base_model
        self.adapter_dir = adapter_dir
        self.max_new_tokens = max_new_tokens
        self.energy_probe = energy_probe
        self.cost_estimator = cost_estimator
        self.clock = clock
        if _generate is not None:
            self._gen: GenerateFn = _generate  # test seam: no model is loaded
        else:
            self._load_model()
            self._gen = self._raw_generate

    def _load_model(self) -> None:
        import torch  # noqa: F401
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.base_model)
        if self._tok.pad_token_id is None:
            self._tok.pad_token = self._tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(self.base_model)
        if self.adapter_dir:
            model = PeftModel.from_pretrained(model, self.adapter_dir)
        self._model = model.eval()

    def _build_prompt(self, messages: list[dict[str, Any]]) -> str:
        return "".join(render_message(m) for m in messages) + "<|assistant|>\n"

    def _raw_generate(self, prompt: str) -> tuple[str, int, int]:
        import torch

        inputs = self._tok(prompt, return_tensors="pt").to(self._model.device)
        n_in = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        new_ids = out[0][n_in:]
        content = self._tok.decode(new_ids, skip_special_tokens=True)
        return content, n_in, int(new_ids.shape[0])

    def complete(self, messages: list[dict[str, Any]], **params: Any) -> ChatResponse:
        prompt = self._build_prompt(messages)
        t0 = self.clock()
        content, n_in, n_out = self._gen(prompt)
        t1 = self.clock()
        measurement = normalize_energy_measurement(
            self.energy_probe(t0, t1) if self.energy_probe is not None else None
        )
        dollars = (
            self.cost_estimator(n_in, n_out, t1 - t0)
            if self.cost_estimator is not None
            else None
        )
        return ChatResponse(
            content=content,
            tokens_in=n_in,
            tokens_out=n_out,
            dollars=dollars,
            latency_s=t1 - t0,
            gpu_energy_j=measurement.joules,
            gpu_energy_provenance=measurement.provenance,
            gpu_energy_complete=measurement.complete,
            accelerator_device_coverage_complete=(
                measurement.accelerator_device_coverage_complete
            ),
        )
