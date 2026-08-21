from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

from glean_gepa.al_adapter import (
    ALRunner,
    AssistantALAdapter,
    Thresholds,
)
from glean_gepa.evalcli_client import EvalCliClient
from glean_gepa.focused_evalset import FocusedEvalSet
from glean_gepa.shell_tool_error_util import (
    HIGH_SIGNAL_VERIFY_OBJECTIVE,
    SHELL_SUCCESS_OBJECTIVE,
    EvalRunShellToolErrorAnalysis,
    ShellToolErrorEntryMetrics,
    ShellToolErrorExample,
    ShellToolErrorMetrics,
)


def test_evaluate_uses_shell_error_rate_objective():
    evalcli = EvalCliClient(binary="/fake/evalcli")
    runner = ALRunner(evalcli=evalcli)
    bigquery_client = MagicMock()
    adapter = AssistantALAdapter(
        runner=runner,
        judging_mode="single_model",
        bigquery_client=bigquery_client,
        teacher_model="gpt",
        student_model="fast",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
    )

    batch = [
        {
            "eval_set_name": "AI Answers Small",
            "eval_set_version": "20260403",
            "deployment_ids": ["scio-prod"],
            "status": "active",
        }
    ]
    analysis = EvalRunShellToolErrorAnalysis(
        eval_id="run_123",
        start_date=date(2026, 8, 8),
        end_date=date(2026, 8, 11),
        aggregate=ShellToolErrorMetrics(
            eval_id="run_123",
            shell_executions=8,
            shell_errors=2,
            shell_error_rate=0.25,
            shell_error_pct=25.0,
            recent_error_examples=(),
        ),
        per_entry={
            "entry-1": ShellToolErrorEntryMetrics(
                entry_id="entry-1",
                shell_executions=4,
                shell_errors=2,
                shell_error_rate=0.5,
                shell_error_pct=50.0,
                recent_error_examples=(),
            )
        },
        high_signal_entry_ids=("entry-1",),
    )

    verify_analysis = EvalRunShellToolErrorAnalysis(
        eval_id="run_verify",
        start_date=date(2026, 8, 11),
        end_date=date(2026, 8, 11),
        aggregate=ShellToolErrorMetrics(
            eval_id="run_verify",
            shell_executions=4,
            shell_errors=0,
            shell_error_rate=0.0,
            shell_error_pct=0.0,
            recent_error_examples=(),
        ),
        per_entry={
            "focused-1": ShellToolErrorEntryMetrics(
                entry_id="focused-1",
                shell_executions=4,
                shell_errors=0,
                shell_error_rate=0.0,
                shell_error_pct=0.0,
                recent_error_examples=(),
            )
        },
        high_signal_entry_ids=(),
    )

    with (
        patch.object(adapter, "_get_or_run_student_eval", side_effect=["run_123", "run_verify"]),
        patch(
            "glean_gepa.al_adapter.fetch_eval_run_shell_tool_error_analysis",
            side_effect=[analysis, verify_analysis],
        ),
        patch(
            "glean_gepa.al_adapter.ensure_focused_eval_set",
            return_value=FocusedEvalSet(
                name="GEPA High Signal AI Answers Small",
                version="20260403_hs_abc123",
                entry_count=1,
            ),
        ) as mock_focused,
    ):
        result = adapter.evaluate(batch, {"WRITING_CODE": "test prompt"}, capture_traces=True)

    assert mock_focused.call_args.kwargs["entry_ids"] == ("entry-1",)

    assert result.summary[SHELL_SUCCESS_OBJECTIVE] == 0.75
    assert result.summary[HIGH_SIGNAL_VERIFY_OBJECTIVE] == 1.0
    assert result.summary["high_signal_entry_count"] == 1.0
    assert len(result.outputs) == 1
    assert result.outputs[0]["entry_id"] == "entry-1"
    assert result.outputs[0]["student_tool_errors"] == 2
    assert result.trajectories is not None
    assert len(result.trajectories) == 1


def test_evaluate_logs_fetched_shell_error_rate_and_error(capsys):
    evalcli = EvalCliClient(binary="/fake/evalcli")
    adapter = AssistantALAdapter(
        runner=ALRunner(evalcli=evalcli),
        judging_mode="single_model",
        bigquery_client=MagicMock(),
        teacher_model="gpt",
        student_model="fast",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
    )
    error_example = ShellToolErrorExample(
        started_at="2026-08-11T12:00:00Z",
        project_id="project-1",
        entry_id="entry-1",
        eval_id="run_123",
        run_id="execution-1",
        trace_id="trace-1",
        span_id="span-1",
        span_name="Execute Action: Shell",
        action_id="Shell",
        action_status="error",
        span_status="error",
        provider_status="failed",
        output_status_code="1",
        error_str="command exited with status 1",
    )
    analysis = EvalRunShellToolErrorAnalysis(
        eval_id="run_123",
        start_date=date(2026, 8, 8),
        end_date=date(2026, 8, 11),
        aggregate=ShellToolErrorMetrics(
            eval_id="run_123",
            shell_executions=4,
            shell_errors=1,
            shell_error_rate=0.25,
            shell_error_pct=25.0,
            recent_error_examples=(error_example,),
        ),
        per_entry={},
        high_signal_entry_ids=(),
    )

    with (
        patch.object(adapter, "_get_or_run_student_eval", return_value="run_123"),
        patch(
            "glean_gepa.al_adapter.fetch_eval_run_shell_tool_error_analysis",
            return_value=analysis,
        ),
    ):
        adapter.evaluate(
            [
                {
                    "eval_set_name": "AI Answers Small",
                    "eval_set_version": "20260403",
                    "deployment_ids": ["scio-prod"],
                    "status": "active",
                }
            ],
            {"WRITING_CODE": "test prompt"},
        )

    output = capsys.readouterr().out
    assert "[Shell Tool] Fetched error rate for eval run_123: 25.00% (1/4)" in output
    assert "[Shell Tool] Error for eval run_123: command exited with status 1" in output
