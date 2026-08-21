from __future__ import annotations

import pytest

from glean_gepa.al_adapter import (
    DEFAULT_FRONTIER_TYPE_BY_MODE,
    PRIMARY_OBJECTIVE_BY_MODE,
    ALRunner,
    AssistantALAdapter,
    Judge,
    Thresholds,
    get_screening_score,
)
from glean_gepa.batch import GleanEvaluationBatch
from glean_gepa.evalcli_client import EvalCliClient
from glean_gepa.shell_tool_error_util import SHELL_SUCCESS_OBJECTIVE


def test_primary_objectives_and_frontier_defaults():
    assert PRIMARY_OBJECTIVE_BY_MODE["single_model"] == SHELL_SUCCESS_OBJECTIVE
    assert PRIMARY_OBJECTIVE_BY_MODE["teacher_student"] == "correctness"
    assert DEFAULT_FRONTIER_TYPE_BY_MODE["single_model"] == "objective"
    assert DEFAULT_FRONTIER_TYPE_BY_MODE["teacher_student"] == "hybrid"


def test_get_screening_score_uses_mode_specific_objective():
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

    assert get_screening_score(shell_eval, "single_model") == 0.8
    assert get_screening_score(judge_eval, "teacher_student") == 0.9


def test_teacher_student_mode_requires_judge():
    runner = ALRunner(evalcli=EvalCliClient(binary="/fake/evalcli"))
    with pytest.raises(ValueError, match="judge is required"):
        AssistantALAdapter(
            runner=runner,
            judging_mode="teacher_student",
            teacher_model="gpt",
            student_model="fast",
            thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
        )


def test_single_model_mode_requires_bigquery_client():
    runner = ALRunner(evalcli=EvalCliClient(binary="/fake/evalcli"))
    with pytest.raises(ValueError, match="bigquery_client is required"):
        AssistantALAdapter(
            runner=runner,
            judging_mode="single_model",
            teacher_model="gpt",
            student_model="fast",
            thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
        )


def test_teacher_student_adapter_accepts_judge():
    runner = ALRunner(evalcli=EvalCliClient(binary="/fake/evalcli"))
    adapter = AssistantALAdapter(
        runner=runner,
        judging_mode="teacher_student",
        judge=Judge(EvalCliClient(binary="/fake/evalcli")),
        teacher_model="gpt",
        student_model="fast",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
    )
    assert adapter.judging_mode == "teacher_student"
