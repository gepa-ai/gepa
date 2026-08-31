"""Typed evaluation records for the Glean adapters."""

from __future__ import annotations

from typing import Literal, NotRequired, TypeAlias, TypedDict

JudgingMode: TypeAlias = Literal["teacher_student", "single_model"]


class EvalSetALDataInst(TypedDict):
    """Configuration shared by both Glean evaluation adapters."""

    eval_set_name: str
    eval_set_version: str
    deployment_ids: list[str]
    status: str


class SingleModelALDataInst(EvalSetALDataInst):
    """Single-model eval data enriched with the failed eval execution identity."""

    eval_entry_id: NotRequired[str]
    eval_run_id: NotRequired[str]
    source_eval_run_id: NotRequired[str]
    eval_trace_id: NotRequired[str]
    eval_entry_ids: NotRequired[list[str]]
    focused_eval_set_name: NotRequired[str]
    focused_eval_set_version: NotRequired[str]
    cached_student_eval_run_id: NotRequired[str]


class TeacherStudentALDataInst(EvalSetALDataInst):
    """Eval-set configuration for a paired teacher/student comparison."""

    cached_student_eval_run_id: NotRequired[str]
    cached_teacher_eval_run_id: NotRequired[str]


class BaseALRolloutOutput(TypedDict):
    """Fields shared by all Glean rollout outputs."""

    deployment_id: str
    query: str
    entry_id: str


class SingleModelALRolloutOutput(BaseALRolloutOutput):
    """Shell reliability evidence from one evaluated student model."""

    student_tool_calls: int
    student_tool_errors: int
    shell_error_messages: list[str]
    student_eval_run_id: str
    shell_action_inputs: NotRequired[list[str]]
    eval_trace_id: NotRequired[str]


class TeacherStudentALRolloutOutput(BaseALRolloutOutput):
    """Full paired execution details used by teacher/student judging."""

    student_answer: str
    student_tool_events: list[str]
    student_loops: int
    student_tool_calls: int
    student_tool_errors: int
    student_input_tokens: int
    student_output_tokens: int
    student_latency_ms: int | None
    teacher_answer: str
    teacher_tool_events: list[str]
    teacher_loops: int
    teacher_tool_calls: int
    teacher_input_tokens: int
    teacher_output_tokens: int
    student_eval_run_id: NotRequired[str]
    teacher_eval_run_id: NotRequired[str]


class SingleModelALTrajectory(TypedDict):
    data: SingleModelALDataInst
    output: SingleModelALRolloutOutput
    score: float
    objective_scores: dict[str, float]


class TeacherStudentALTrajectory(TypedDict):
    data: TeacherStudentALDataInst
    output: TeacherStudentALRolloutOutput
    score: float
    objective_scores: dict[str, float]


# Shared infrastructure dispatches to one concrete adapter at runtime. Keep
# these unions internal/compatibility-facing; adapter users should use the
# concrete types above.
ALDataInst: TypeAlias = SingleModelALDataInst | TeacherStudentALDataInst
ALRolloutOutput: TypeAlias = SingleModelALRolloutOutput | TeacherStudentALRolloutOutput
ALTrajectory: TypeAlias = SingleModelALTrajectory | TeacherStudentALTrajectory


__all__ = [
    "ALDataInst",
    "ALRolloutOutput",
    "ALTrajectory",
    "EvalSetALDataInst",
    "JudgingMode",
    "SingleModelALDataInst",
    "SingleModelALRolloutOutput",
    "SingleModelALTrajectory",
    "TeacherStudentALDataInst",
    "TeacherStudentALRolloutOutput",
    "TeacherStudentALTrajectory",
]
