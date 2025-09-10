# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from dataclasses import dataclass
from typing import Callable, Protocol

from gepa.core.adapter import Trajectory
from gepa.core.state import GEPAState


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


class BatchSampler(Protocol):
    def next_minibatch_indices(self, trainset_size: int, iteration: int) -> list[int]: ...


class LanguageModel(Protocol):
    def __call__(self, prompt: str) -> str: ...


@dataclass
class Signature:
    prompt_template: str
    input_keys: list[str]
    output_keys: list[str]
    prompt_renderer: Callable[[dict[str, str]], str]
    output_extractor: Callable[[str], dict[str, str]]

    @classmethod
    def run(cls, lm: LanguageModel, input_dict: dict[str, str]) -> dict[str, str]:
        full_prompt = cls.prompt_renderer(input_dict)
        lm_out = lm(full_prompt).strip()
        return cls.output_extractor(lm_out)
