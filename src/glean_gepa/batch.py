"""Glean-specific evaluation result types."""

from dataclasses import dataclass
from typing import Generic

from gepa.core.adapter import EvaluationBatch, RolloutOutput, Trajectory


@dataclass
class GleanEvaluationBatch(EvaluationBatch[Trajectory, RolloutOutput], Generic[Trajectory, RolloutOutput]):
    """A GEPA evaluation batch with aggregate metrics used for child screening."""

    summary: dict[str, float] | None = None
