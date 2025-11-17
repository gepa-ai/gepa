"""Scoring callbacks for TurboGEPA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from .interfaces import Candidate, EvalResult


SCORE_KEY = "_score"


@dataclass(slots=True)
class ScoringContext:
    candidate: Candidate
    result: EvalResult
    rung_index: int
    rung_fraction: float
    is_final_rung: bool
    parent_objectives: Mapping[str, float] | None
    parent_rung_scores: Mapping[float, float] | None
    coverage_fraction: float | None


ScoringFn = Callable[[ScoringContext], float]


def maximize_metric(metric: str, *, default: float = 0.0) -> ScoringFn:
    """Return a scoring function that extracts ``metric`` from objectives."""

    def _callback(ctx: ScoringContext) -> float:
        value = ctx.result.objectives.get(metric)
        return float(value) if isinstance(value, (int, float)) else float(default)

    return _callback
