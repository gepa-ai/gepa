"""Adapter for teacher-vs-student Glean evaluations."""

from __future__ import annotations

from glean_gepa.adapter_types import (
    TeacherStudentALDataInst,
    TeacherStudentALRolloutOutput,
    TeacherStudentALTrajectory,
)
from glean_gepa.al_adapter import ALRunner, GleanAdapterBase, Judge, Thresholds


class TeacherStudentAdapter(GleanAdapterBase):
    """Optimize instructions from teacher-vs-student execution comparisons."""

    def __init__(
        self,
        runner: ALRunner,
        teacher_model: str,
        thresholds: Thresholds,
        student_model: str,
        *,
        judge: Judge | None = None,
        cache_file: str | None = None,
    ):
        super().__init__(
            runner=runner,
            teacher_model=teacher_model,
            thresholds=thresholds,
            student_model=student_model,
            judging_mode="teacher_student",
            judge=judge,
            cache_file=cache_file,
        )


__all__ = [
    "TeacherStudentALDataInst",
    "TeacherStudentALRolloutOutput",
    "TeacherStudentALTrajectory",
    "TeacherStudentAdapter",
]
