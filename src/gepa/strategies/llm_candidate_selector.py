# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import logging
import random
import re
from collections.abc import Mapping
from typing import Any, ClassVar, Literal

from gepa.core.state import GEPAState
from gepa.lm import LM
from gepa.proposer.reflective_mutation.base import CandidateSelector, LanguageModel, Signature
from gepa.strategies.candidate_selector import CurrentBestCandidateSelector, ParetoCandidateSelector

logger = logging.getLogger(__name__)


class CandidateSelectionSignature(Signature):
    """LLM prompt/parse layer for candidate selection.

    Formats candidate information (texts, scores, lineage) into a structured
    prompt and parses the LLM response to extract selected candidate index(es).
    """

    prompt_template: ClassVar[str] = ""  # unused; mode-specific templates below

    _HEADER = """You are an expert evaluator for an evolutionary optimization system.
Below are {num_candidates} candidate solutions. Each has text content for its components and performance scores.
Your task is to evaluate them by comparing their relative quality.

## Candidates

{candidates_block}"""

    _MODE_BEST = """Based on both the quantitative scores AND qualitative assessment of the candidate texts, select the single best candidate to use as the basis for the next mutation. Consider:
- Overall score (higher is better)
- Quality, clarity, and sophistication of the text
- Potential for further improvement through mutation
- Diversity of approach compared to other candidates

Respond with ONLY the candidate number (e.g., "0" or "3"). Do not include any other text."""

    _MODE_PARETO_FRONT = """Identify the set of Pareto-optimal candidates -- those that represent meaningfully different approaches or trade-offs that should be preserved.
A candidate is Pareto-optimal if no other candidate is strictly better in ALL respects (score, text quality, approach diversity).

Consider:
- Which candidates represent genuinely different strategies?
- Which candidates excel in different dimensions (even if not highest overall)?
- Which candidates have unique strengths worth preserving?

Respond with a comma-separated list of candidate numbers (e.g., "0, 2, 5"). Do not include any other text."""

    input_keys: ClassVar[list[str]] = ["candidates", "mode", "num_candidates"]
    output_keys: ClassVar[list[str]] = ["selected_index", "selected_indices"]

    @classmethod
    def prompt_renderer(cls, input_dict: Mapping[str, Any]) -> str:
        candidates = input_dict["candidates"]
        mode = input_dict["mode"]
        num_candidates = input_dict["num_candidates"]

        # Build candidates block
        parts: list[str] = []
        for display_idx, candidate_info in enumerate(candidates):
            score = candidate_info["score"]
            objective_scores = candidate_info.get("objective_scores", {})
            texts = candidate_info["texts"]
            parents = candidate_info.get("parents", [])

            section = f"### Candidate {display_idx}\nAggregate Score: {score:.4f}\n"

            if objective_scores:
                obj_parts = [f"  {name}: {val:.4f}" for name, val in sorted(objective_scores.items())]
                section += "Objective Scores:\n" + "\n".join(obj_parts) + "\n"

            section += "Components:\n"
            for comp_name, comp_text in sorted(texts.items()):
                section += f"  {comp_name}: {comp_text}\n"

            if parents:
                parent_str = ", ".join(str(p) for p in parents if p is not None)
                if parent_str:
                    section += f"Parent: derived from candidate(s) {parent_str}\n"

            parts.append(section)

        candidates_block = "\n".join(parts)
        header = cls._HEADER.format(num_candidates=num_candidates, candidates_block=candidates_block)

        if mode == "best":
            return header + "\n" + cls._MODE_BEST
        elif mode == "pareto_front":
            return header + "\n" + cls._MODE_PARETO_FRONT
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @classmethod
    def output_extractor(cls, lm_out: str) -> dict[str, str]:
        stripped = lm_out.strip()

        # Fast path: response is just a single integer (e.g., "2")
        if stripped.isdigit():
            return {"selected_index": stripped, "selected_indices": stripped}

        # Try comma-separated integers (e.g., "0, 2, 5")
        comma_parts = [p.strip() for p in stripped.split(",")]
        if all(p.isdigit() for p in comma_parts if p):
            digits = [p for p in comma_parts if p]
            if digits:
                unique = list(dict.fromkeys(digits))
                return {"selected_index": unique[0], "selected_indices": ",".join(unique)}

        # Fallback: find "candidate N" patterns first, then bare integers
        candidate_refs = re.findall(r"[Cc]andidate\s+(\d+)", lm_out)
        if candidate_refs:
            unique = list(dict.fromkeys(candidate_refs))
            return {"selected_index": unique[0], "selected_indices": ",".join(unique)}

        # Last resort: extract all bare integers
        numbers = re.findall(r"\b(\d+)\b", lm_out)
        if not numbers:
            raise ValueError(f"No candidate numbers found in LLM response: {lm_out!r}")
        unique = list(dict.fromkeys(numbers))
        return {"selected_index": unique[0], "selected_indices": ",".join(unique)}


class LLMCandidateSelector:
    """Candidate selector that uses an LLM to judge and select candidates.

    Instead of relying on absolute numeric scores, this selector presents
    candidate texts and scores to an LLM and asks it to select based on
    relative comparison -- treating the LLM as the evolutionary "world"
    (selection pressure / world model).

    Args:
        lm: Language model to use for selection. Can be a ``LanguageModel``
            callable or a model name string (e.g., ``"openai/gpt-4.1-mini"``).
        mode: Selection mode. ``"best"`` selects the single best candidate.
            ``"pareto_front"`` identifies the Pareto-optimal set and samples
            from it.
        pre_filter_k: Maximum number of candidates to show the LLM (top K by
            aggregate score). Keeps prompts within context limits.
        max_text_chars: Truncate each component's text to this many characters.
        rng: Random number generator for Pareto front sampling and fallback.
        fallback: Fallback selector used when the LLM call fails or returns
            an unparseable response. Defaults to ``CurrentBestCandidateSelector``
            for ``"best"`` mode and ``ParetoCandidateSelector`` for
            ``"pareto_front"`` mode.
    """

    def __init__(
        self,
        lm: LanguageModel | str,
        mode: Literal["best", "pareto_front"] = "best",
        pre_filter_k: int = 10,
        max_text_chars: int = 500,
        rng: random.Random | None = None,
        fallback: CandidateSelector | None = None,
    ):
        self.lm: LanguageModel = LM(lm) if isinstance(lm, str) else lm
        self.mode = mode
        self.pre_filter_k = pre_filter_k
        self.max_text_chars = max_text_chars
        self.rng = rng if rng is not None else random.Random(0)
        if fallback is not None:
            self.fallback = fallback
        elif mode == "pareto_front":
            self.fallback = ParetoCandidateSelector(rng=self.rng)
        else:
            self.fallback = CurrentBestCandidateSelector()

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_text_chars:
            return text
        return text[: self.max_text_chars] + "...[truncated]"

    def _build_candidates_info(
        self, state: GEPAState, top_k_real_indices: list[int]
    ) -> tuple[list[dict[str, Any]], dict[int, int]]:
        """Build candidate info list and display-to-real index mapping."""
        candidates_info: list[dict[str, Any]] = []
        index_mapping: dict[int, int] = {}  # display_idx -> real program idx

        scores = state.program_full_scores_val_set

        for display_idx, real_idx in enumerate(top_k_real_indices):
            candidate = state.program_candidates[real_idx]
            truncated_texts = {name: self._truncate(text) for name, text in candidate.items()}

            candidates_info.append(
                {
                    "real_idx": real_idx,
                    "score": scores[real_idx],
                    "objective_scores": state.prog_candidate_objective_scores[real_idx],
                    "texts": truncated_texts,
                    "parents": state.parent_program_for_candidate[real_idx],
                }
            )
            index_mapping[display_idx] = real_idx

        return candidates_info, index_mapping

    def select_candidate_idx(self, state: GEPAState) -> int:
        num_candidates = len(state.program_candidates)

        # Early exit: single candidate
        if num_candidates <= 1:
            return 0

        # Pre-filter: top K by aggregate score
        scores = state.program_full_scores_val_set
        sorted_indices = sorted(range(num_candidates), key=lambda i: scores[i], reverse=True)
        top_k_real_indices = sorted_indices[: min(self.pre_filter_k, num_candidates)]

        candidates_info, index_mapping = self._build_candidates_info(state, top_k_real_indices)

        try:
            result = CandidateSelectionSignature.run(
                self.lm,
                {
                    "candidates": candidates_info,
                    "mode": self.mode,
                    "num_candidates": len(candidates_info),
                },
            )

            if self.mode == "pareto_front":
                # Parse multiple indices and sample from them
                indices_str = result.get("selected_indices", "")
                display_indices = [int(x.strip()) for x in indices_str.split(",") if x.strip()]
                valid_display_indices = [i for i in display_indices if i in index_mapping]
                if not valid_display_indices:
                    raise ValueError(f"No valid indices in LLM response: {indices_str}")
                selected_display = self.rng.choice(valid_display_indices)
            else:
                # Parse single index
                selected_display = int(result["selected_index"])
                if selected_display not in index_mapping:
                    raise ValueError(
                        f"LLM returned index {selected_display} outside valid range [0, {len(index_mapping)})"
                    )

            return index_mapping[selected_display]

        except Exception as e:
            logger.warning(f"LLM candidate selection failed ({e}), falling back to {type(self.fallback).__name__}")
            return self.fallback.select_candidate_idx(state)
