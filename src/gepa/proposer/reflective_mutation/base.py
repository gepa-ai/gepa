# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, Protocol, runtime_checkable

from gepa.core.adapter import Trajectory
from gepa.core.state import GEPAState


@runtime_checkable
class CandidateSelector(Protocol):
    def select_candidate_idx(self, state: GEPAState) -> int: ...


class ReflectionComponentSelector(Protocol):
    def __call__(
        self,
        state: GEPAState,
        trajectories: list[Trajectory],
        subsample_scores: list[float],
        candidate_idx: int,
        candidate: dict[str, str],
    ) -> list[str]: ...


class LanguageModel(Protocol):
    def __call__(self, prompt: str | list[dict[str, Any]]) -> str: ...


@dataclass
class Signature:
    prompt_template: ClassVar[str]
    input_keys: ClassVar[list[str]]
    output_keys: ClassVar[list[str]]

    @classmethod
    def prompt_renderer(cls, input_dict: Mapping[str, Any]) -> str | list[dict[str, Any]]:
        raise NotImplementedError

    @classmethod
    def output_extractor(cls, lm_out: str) -> dict[str, str]:
        raise NotImplementedError

    @classmethod
    def run(cls, lm: LanguageModel, input_dict: Mapping[str, Any]) -> dict[str, str]:
        full_prompt = cls.prompt_renderer(input_dict)
        lm_res = lm(full_prompt)
        lm_out = lm_res.strip()
        return cls.output_extractor(lm_out)

    @classmethod
    def run_with_metadata(
        cls, lm: LanguageModel, input_dict: Mapping[str, Any]
    ) -> tuple[dict[str, str], str | list[dict[str, Any]], str, float]:
        """Like ``run()``, but also returns the rendered prompt, raw LM output,
        and the USD cost of the reflection call.

        If ``lm`` exposes ``call_with_cost(prompt) -> (text, cost)``, that path
        is used. Otherwise this falls back to ``lm(prompt)`` with cost ``0.0``,
        keeping plain-callable reflection LMs working unchanged.

        Returns:
            A tuple of (extracted_output, rendered_prompt, raw_lm_output, cost_usd).
        """
        full_prompt = cls.prompt_renderer(input_dict)

        call_with_cost = getattr(lm, "call_with_cost", None)
        if call_with_cost is not None:
            lm_res, call_cost = call_with_cost(full_prompt)
        else:
            lm_res = lm(full_prompt)
            call_cost = 0.0

        lm_out = lm_res.strip()
        return cls.output_extractor(lm_out), full_prompt, lm_out, call_cost
