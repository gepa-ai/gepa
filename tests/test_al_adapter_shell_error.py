from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock, patch

from glean_gepa.al_adapter import (
    ALRunner,
    Candidate,
    ModuleSpec,
    Thresholds,
    extract_shell_action_inputs,
)
from glean_gepa.evalcli_client import EvalCliClient
from glean_gepa.shell_tool_error_util import (
    SHELL_SUCCESS_OBJECTIVE,
    EvalRunShellToolErrorAnalysis,
    ShellToolErrorEntryMetrics,
    ShellToolErrorExample,
    ShellToolErrorMetrics,
)
from glean_gepa.single_model_adapter import SingleModelAdapter


def test_evaluate_uses_shell_error_rate_objective():
    evalcli = EvalCliClient(binary="/fake/evalcli")
    runner = ALRunner(evalcli=evalcli)
    bigquery_client = MagicMock()
    adapter = SingleModelAdapter(
        runner=runner,
        bigquery_client=bigquery_client,
        student_model="fast",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
    )

    batch = [
        {
            "eval_set_name": "AI Answers Small",
            "eval_set_version": "20260403",
            "deployment_ids": ["scio-prod"],
            "status": "active",
            "eval_trace_id": "trace-original",
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
                recent_error_examples=(
                    ShellToolErrorExample(
                        started_at="2026-08-11T12:00:00Z",
                        project_id="project-1",
                        entry_id="entry-1",
                        eval_id="run_123",
                        run_id="execution-1",
                        trace_id="trace-student-1",
                        span_id="span-1",
                        span_name="Execute Action: Shell",
                        action_id="Shell",
                        action_run_id="call-1",
                        action_status="error",
                        span_status="error",
                        provider_status="failed",
                        output_status_code="1",
                        error_str="command exited with status 1",
                    ),
                ),
                trace_ids=("trace-student-1",),
            )
        },
        high_signal_entry_ids=("entry-1",),
    )

    with (
        patch.object(adapter, "_get_or_run_student_eval", return_value="run_123") as run_eval,
        patch.object(
            adapter.runner.evalcli,
            "get_analysis_trace",
            return_value={
                "trace": {
                    "spans": [
                        {
                            "name": "Execute Action: Shell",
                            "attributes": {
                                "input": {
                                    "strValue": json.dumps(
                                        {"action_input": json.dumps({"command": "python3 broken.py"})}
                                    )
                                },
                                "span.gle": {"strValue": json.dumps({"action": {"action_run_id": "call-1"}})},
                            },
                        }
                    ]
                }
            },
        ) as get_trace,
        patch(
            "glean_gepa.single_model_adapter.fetch_eval_run_shell_tool_error_analysis",
            return_value=analysis,
        ),
    ):
        result = adapter.evaluate(batch, {"WRITING_CODE": "test prompt"}, capture_traces=True)

    run_eval.assert_called_once()
    get_trace.assert_called_once()

    assert result.summary[SHELL_SUCCESS_OBJECTIVE] == 0.75
    assert result.summary["high_signal_entry_count"] == 1.0
    assert len(result.outputs) == 1
    assert result.outputs[0]["entry_id"] == "entry-1"
    assert result.outputs[0]["student_tool_errors"] == 2
    assert result.outputs[0]["eval_trace_id"] == "trace-student-1"
    assert result.outputs[0]["shell_action_inputs"] == ['{"command": "python3 broken.py"}']
    assert result.trajectories is not None
    assert len(result.trajectories) == 1
    assert result.trajectories[0]["data"]["eval_entry_id"] == "entry-1"
    assert result.trajectories[0]["data"]["eval_run_id"] == "run_123"
    assert result.trajectories[0]["data"]["eval_trace_id"] == "trace-student-1"
    reflective = adapter.make_reflective_dataset({"WRITING_CODE": "test prompt"}, result, ["WRITING_CODE"], k=1)[
        "WRITING_CODE"
    ][0]
    assert reflective["Inputs"]["eval_trace_id"] == "trace-student-1"
    assert reflective["Execution Errors"] == ["command exited with status 1"]
    assert reflective["Action Inputs"] == ['{"command": "python3 broken.py"}']
    assert "command exited with status 1" in reflective["Feedback"]
    captured_prompts = []

    def reflection_lm(prompt: str) -> str:
        captured_prompts.append(prompt)
        return "NOT_RELEVANT" if len(captured_prompts) == 1 else "rewritten code instructions"

    variants, not_relevant = adapter.propose_new_texts(
        reflection_lm,
        Candidate(
            model="fast",
            prompt_modules={"WRITING_CODE": "test prompt"},
            module_specs={"WRITING_CODE": ModuleSpec("WRITING_CODE", "free_text", 1024)},
            global_token_cap=4096,
            baseline_prompt_hash="seed",
        ),
        ["WRITING_CODE"],
        [reflective],
    )
    assert variants == ["rewritten code instructions"]
    assert not not_relevant
    assert "NOT_RELEVANT" not in captured_prompts[0]
    assert "TEACHER_ANSWER:" not in captured_prompts[0]
    assert "STUDENT_ANSWER:" not in captured_prompts[0]
    assert "TEACHER_TOOLS:" not in captured_prompts[0]
    assert "STUDENT_TOOLS:" not in captured_prompts[0]
    assert 'ACTION_INPUT: {"command": "python3 broken.py"}' in captured_prompts[0]
    assert "EVAL_TRACE_ID: trace-student-1" in captured_prompts[0]
    assert "METRICS:" not in captured_prompts[0]
    assert "Shell success rate issue:" not in captured_prompts[0]
    assert "Recent shell errors: command exited with status 1" in captured_prompts[0]
    assert "EXECUTION_ERRORS: ['command exited with status 1']" in captured_prompts[1]


def test_extract_shell_action_inputs_matches_action_run_id():
    action_input = json.dumps({"command": "python3 broken.py", "destructive": False})
    trace = {
        "trace": {
            "spans": [
                {
                    "name": "Execute Action: Shell",
                    "attributes": {
                        "input": {"strValue": json.dumps({"action_input": action_input})},
                        "span.gle": {"strValue": json.dumps({"action": {"action_run_id": "call-shell-1"}})},
                    },
                }
            ]
        }
    }

    assert extract_shell_action_inputs(trace) == {"call-shell-1": action_input}


def test_evaluate_logs_fetched_shell_error_rate_and_error(capsys):
    evalcli = EvalCliClient(binary="/fake/evalcli")
    adapter = SingleModelAdapter(
        runner=ALRunner(evalcli=evalcli),
        bigquery_client=MagicMock(),
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
            "glean_gepa.single_model_adapter.fetch_eval_run_shell_tool_error_analysis",
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


def test_capture_traces_reuses_persisted_minimal_error_evidence(tmp_path):
    cache_file = tmp_path / "eval-cache.json"
    error = ShellToolErrorExample(
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
    entry = ShellToolErrorEntryMetrics(
        entry_id="entry-1",
        shell_executions=1,
        shell_errors=1,
        shell_error_rate=1.0,
        shell_error_pct=100.0,
        recent_error_examples=(error,),
        trace_ids=("trace-1",),
    )
    analysis = EvalRunShellToolErrorAnalysis(
        eval_id="run_123",
        start_date=date(2026, 8, 11),
        end_date=date(2026, 8, 11),
        aggregate=ShellToolErrorMetrics(
            eval_id="run_123",
            shell_executions=1,
            shell_errors=1,
            shell_error_rate=1.0,
            shell_error_pct=100.0,
            recent_error_examples=(error,),
        ),
        per_entry={"entry-1": entry},
        high_signal_entry_ids=("entry-1",),
    )
    batch = [
        {
            "eval_set_name": "AI Answers Small",
            "eval_set_version": "20260403",
            "deployment_ids": ["scio-prod"],
            "status": "active",
        }
    ]

    first_runner = ALRunner(evalcli=EvalCliClient(binary="/fake/evalcli"))
    first = SingleModelAdapter(
        runner=first_runner,
        bigquery_client=MagicMock(),
        student_model="fast",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
        cache_file=str(cache_file),
    )
    with (
        patch.object(first_runner, "run", return_value="run_123") as run,
        patch(
            "glean_gepa.single_model_adapter.fetch_eval_run_shell_tool_error_analysis", return_value=analysis
        ) as fetch,
    ):
        without_traces = first.evaluate(batch, {"WRITING_CODE": "prompt"}, capture_traces=False)
    assert without_traces.trajectories is None
    run.assert_called_once()
    fetch.assert_called_once()

    second_runner = ALRunner(evalcli=EvalCliClient(binary="/fake/evalcli"))
    second = SingleModelAdapter(
        runner=second_runner,
        bigquery_client=MagicMock(),
        student_model="fast",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
        cache_file=str(cache_file),
    )
    with (
        patch.object(second_runner, "run") as run,
        patch("glean_gepa.single_model_adapter.fetch_eval_run_shell_tool_error_analysis") as fetch,
    ):
        with_traces = second.evaluate(batch, {"WRITING_CODE": "prompt"}, capture_traces=True)

    run.assert_not_called()
    fetch.assert_not_called()
    assert with_traces.trajectories is not None
    trace_output = with_traces.trajectories[0]["output"]
    assert trace_output["shell_error_messages"] == ["command exited with status 1"]
    assert set(trace_output) == {
        "deployment_id",
        "query",
        "student_tool_calls",
        "student_tool_errors",
        "entry_id",
        "shell_error_messages",
        "student_eval_run_id",
        "eval_trace_id",
    }


def test_shell_error_analysis_cache_round_trip(tmp_path):
    cache_file = tmp_path / "eval-cache.json"
    analysis = EvalRunShellToolErrorAnalysis(
        eval_id="run_cached",
        start_date=date(2026, 8, 8),
        end_date=date(2026, 8, 11),
        aggregate=ShellToolErrorMetrics(
            eval_id="run_cached",
            shell_executions=10,
            shell_errors=3,
            shell_error_rate=0.3,
            shell_error_pct=30.0,
            recent_error_examples=(),
        ),
        per_entry={
            "entry-1": ShellToolErrorEntryMetrics(
                entry_id="entry-1",
                shell_executions=2,
                shell_errors=1,
                shell_error_rate=0.5,
                shell_error_pct=50.0,
                recent_error_examples=(),
                trace_ids=("trace-cached",),
            )
        },
        high_signal_entry_ids=("entry-1",),
    )
    evalcli = EvalCliClient(binary="/fake/evalcli")
    adapter = SingleModelAdapter(
        runner=ALRunner(evalcli=evalcli),
        bigquery_client=MagicMock(),
        student_model="fast",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
        cache_file=str(cache_file),
    )

    with patch(
        "glean_gepa.single_model_adapter.fetch_eval_run_shell_tool_error_analysis", return_value=analysis
    ) as fetch:
        assert adapter._get_or_fetch_shell_error_analysis("run_cached") is analysis
        fetch.assert_called_once()

    adapter._save_cache()

    reloaded = SingleModelAdapter(
        runner=ALRunner(evalcli=evalcli),
        bigquery_client=MagicMock(),
        student_model="fast",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
        cache_file=str(cache_file),
    )
    with patch("glean_gepa.single_model_adapter.fetch_eval_run_shell_tool_error_analysis") as fetch:
        cached = reloaded._get_or_fetch_shell_error_analysis("run_cached")

    fetch.assert_not_called()
    assert cached.aggregate.shell_error_rate == 0.3
    assert cached.high_signal_entry_ids == ("entry-1",)
    assert cached.per_entry["entry-1"].trace_ids == ("trace-cached",)


def test_legacy_shell_error_analysis_cache_is_refetched(tmp_path):
    cache_file = tmp_path / "eval-cache.json"
    cache_file.write_text(
        json.dumps(
            {
                "eval_analysis_cache": {
                    "run_legacy": {
                        "eval_id": "run_legacy",
                        "start_date": "2026-08-08",
                        "end_date": "2026-08-11",
                        "aggregate": {
                            "eval_id": "run_legacy",
                            "shell_executions": 1,
                            "shell_errors": 1,
                            "shell_error_rate": 1.0,
                            "shell_error_pct": 100.0,
                            "recent_error_examples": [],
                        },
                        "per_entry": {},
                        "high_signal_entry_ids": [],
                    }
                }
            }
        )
    )
    refreshed = EvalRunShellToolErrorAnalysis(
        eval_id="run_legacy",
        start_date=date(2026, 8, 8),
        end_date=date(2026, 8, 11),
        aggregate=ShellToolErrorMetrics(
            eval_id="run_legacy",
            shell_executions=0,
            shell_errors=0,
            shell_error_rate=0.0,
            shell_error_pct=0.0,
            recent_error_examples=(),
        ),
        per_entry={},
        high_signal_entry_ids=(),
    )
    adapter = SingleModelAdapter(
        runner=ALRunner(evalcli=EvalCliClient(binary="/fake/evalcli")),
        bigquery_client=MagicMock(),
        student_model="fast",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
        cache_file=str(cache_file),
    )

    with patch(
        "glean_gepa.single_model_adapter.fetch_eval_run_shell_tool_error_analysis", return_value=refreshed
    ) as fetch:
        assert adapter._get_or_fetch_shell_error_analysis("run_legacy") is refreshed

    fetch.assert_called_once()
