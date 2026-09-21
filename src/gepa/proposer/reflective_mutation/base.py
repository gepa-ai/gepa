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


@dataclass(frozen=True)
class SignatureParseResult:
    """Non-throwing result returned by signature adapters."""

    output: dict[str, str] | None
    error: str | None = None

    @classmethod
    def success(cls, output: dict[str, str]) -> "SignatureParseResult":
        return cls(output=output)

    @classmethod
    def failure(cls, error: str) -> "SignatureParseResult":
        return cls(output=None, error=error)

    def require(self, error_type: type[Exception] = ValueError) -> dict[str, str]:
        """Return the output or raise at an explicitly hard-fail boundary."""
        if self.output is None:
            raise error_type(self.error or "signature output could not be parsed")
        return self.output


class SignatureAdapter(Protocol):
    """Translate between a signature's inputs and an LM's wire format.

    This is deliberately optional: signatures that implement the historical
    ``prompt_renderer`` / ``output_extractor`` hooks continue to work unchanged.
    """

    def format(self, signature: type["Signature"], input_dict: Mapping[str, Any]) -> str | list[dict[str, Any]]: ...

    def parse(self, signature: type["Signature"], lm_out: str) -> SignatureParseResult: ...


@dataclass
class Signature:
    prompt_template: ClassVar[str]
    input_keys: ClassVar[list[str]]
    output_keys: ClassVar[list[str]]
    adapter: ClassVar[SignatureAdapter | None] = None

    @classmethod
    def prompt_renderer(cls, input_dict: Mapping[str, Any]) -> str | list[dict[str, Any]]:
        raise NotImplementedError

    @classmethod
    def output_extractor(cls, lm_out: str) -> dict[str, str]:
        raise NotImplementedError

    @classmethod
    def _render(cls, input_dict: Mapping[str, Any]) -> str | list[dict[str, Any]]:
        # An explicitly overridden legacy hook wins. This makes adapters an
        # additive seam rather than a breaking change for existing subclasses.
        if cls.adapter is None or "prompt_renderer" in cls.__dict__:
            return cls.prompt_renderer(input_dict)
        return cls.adapter.format(cls, input_dict)

    @classmethod
    def _parse(cls, lm_out: str) -> dict[str, str]:
        if cls.adapter is None or "output_extractor" in cls.__dict__:
            return cls.output_extractor(lm_out)
        return cls.adapter.parse(cls, lm_out).require()

    @classmethod
    def run(cls, lm: LanguageModel, input_dict: Mapping[str, Any]) -> dict[str, str]:
        full_prompt = cls._render(input_dict)
        lm_res = lm(full_prompt)
        lm_out = lm_res.strip()
        return cls._parse(lm_out)

    @classmethod
    def run_with_metadata(
        cls, lm: LanguageModel, input_dict: Mapping[str, Any]
    ) -> tuple[dict[str, str], str | list[dict[str, Any]], str]:
        """Like ``run()``, but also returns the rendered prompt and raw LM output.

        Returns:
            A tuple of (extracted_output, rendered_prompt, raw_lm_output).
        """
        full_prompt = cls._render(input_dict)
        lm_res = lm(full_prompt)
        lm_out = lm_res.strip()
        return cls._parse(lm_out), full_prompt, lm_out
