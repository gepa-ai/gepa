"""Glean-specific evaluation result types."""

from dataclasses import dataclass
from typing import Generic, NotRequired, TypedDict

from gepa.core.adapter import EvaluationBatch, RolloutOutput, Trajectory


class EvalRunIds(TypedDict):
    """Eval runs used for one eval-set item in a candidate evaluation."""

    eval_set_name: str
    eval_set_version: str
    student_eval_run_id: str
    teacher_eval_run_id: NotRequired[str]


@dataclass
class GleanEvaluationBatch(EvaluationBatch[Trajectory, RolloutOutput], Generic[Trajectory, RolloutOutput]):
    """A GEPA evaluation batch with aggregate metrics used for child screening."""

    summary: dict[str, float] | None = None
    eval_run_ids: list[EvalRunIds] | None = None
