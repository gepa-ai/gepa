from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from glean_gepa.al_adapter import ALRunner, Judge, Thresholds
from glean_gepa.batch import GleanEvaluationBatch
from glean_gepa.evalcli_client import EvalCliClient
from glean_gepa.shell_tool_error_util import SHELL_SUCCESS_OBJECTIVE
from glean_gepa.single_model_adapter import SingleModelAdapter
from glean_gepa.teacher_student_adapter import TeacherStudentAdapter


def test_concrete_adapters_own_screening_configuration():
    runner = ALRunner(evalcli=EvalCliClient(binary="/fake/evalcli"))
    single_adapter = SingleModelAdapter(
        runner=runner,
        bigquery_client=MagicMock(),
        student_model="fast",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
    )
    teacher_adapter = TeacherStudentAdapter(
        runner=runner,
        judge=Judge(EvalCliClient(binary="/fake/evalcli")),
        teacher_model="gpt",
        student_model="fast",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
    )
    shell_eval = GleanEvaluationBatch(
        outputs=[],
        scores=[0.8],
        summary={SHELL_SUCCESS_OBJECTIVE: 0.8, "correctness": 0.5},
    )
    judge_eval = GleanEvaluationBatch(
        outputs=[],
        scores=[0.9],
        summary={SHELL_SUCCESS_OBJECTIVE: 0.2, "correctness": 0.9},
    )

    assert single_adapter.primary_objective == SHELL_SUCCESS_OBJECTIVE
    assert single_adapter.default_frontier_type == "objective"
    assert single_adapter.get_screening_score(shell_eval) == 0.8
    assert teacher_adapter.primary_objective == "correctness"
    assert teacher_adapter.default_frontier_type == "hybrid"
    assert teacher_adapter.get_screening_score(judge_eval) == 0.9
    assert not hasattr(single_adapter, "judging_mode")
    assert not hasattr(teacher_adapter, "judging_mode")


def test_teacher_student_adapter_requires_judge():
    runner = ALRunner(evalcli=EvalCliClient(binary="/fake/evalcli"))
    with pytest.raises(ValueError, match="judge is required"):
        TeacherStudentAdapter(
            runner=runner,
            teacher_model="gpt",
            student_model="fast",
            thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
        )


def test_single_model_adapter_requires_bigquery_client():
    runner = ALRunner(evalcli=EvalCliClient(binary="/fake/evalcli"))
    with pytest.raises(ValueError, match="bigquery_client is required"):
        SingleModelAdapter(
            runner=runner,
            student_model="fast",
            thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
        )


def test_teacher_student_adapter_accepts_judge():
    runner = ALRunner(evalcli=EvalCliClient(binary="/fake/evalcli"))
    adapter = TeacherStudentAdapter(
        runner=runner,
        judge=Judge(EvalCliClient(binary="/fake/evalcli")),
        teacher_model="gpt",
        student_model="fast",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
    )
    assert adapter.judge is not None
