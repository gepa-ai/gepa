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
    def __call__(self, prompt: str) -> str | dict[str, str]: ...


@dataclass
class Signature:
    prompt_template: ClassVar[str]
    input_keys: ClassVar[list[str]]
    output_keys: ClassVar[list[str]]

    @classmethod
    def prompt_renderer(cls, input_dict: Mapping[str, Any]) -> str:
        raise NotImplementedError

    @classmethod
    def output_extractor(cls, lm_out: str) -> dict[str, str]:
        raise NotImplementedError

    @classmethod
    def run(cls, lm: LanguageModel, input_dict: Mapping[str, Any]) -> dict[str, str]:
        full_prompt = cls.prompt_renderer(input_dict)
        lm_out = lm(full_prompt)
        
        if type(lm_out) == str:
            text_out = lm_out.strip()
        elif type(lm_out) == dict:
            text_out = lm_out["text"].strip()            
        else:
            raise TypeError("The output of the lm call should be str or dict!")
        
        return cls.output_extractor(text_out)
