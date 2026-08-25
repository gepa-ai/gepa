"""Adapter for optimizing prompts in single-model iterations."""

from __future__ import annotations

from typing import Any, cast

from glean_gepa.adapter_types import (
    ALDataInst,
    SingleModelALDataInst,
    SingleModelALRolloutOutput,
    SingleModelALTrajectory,
)
from glean_gepa.al_adapter import (
    ALRunner,
    GleanAdapterBase,
    ReflectiveExample,
    ReflectiveExampleInputs,
    ReflectiveExampleMetrics,
    Thresholds,
    enrich_shell_error_action_inputs,
    log_shell_tool_error_analysis,
)
from glean_gepa.batch import GleanEvaluationBatch
from glean_gepa.prompt import compile_encoded_prompt
from glean_gepa.shell_tool_error_util import (
    SHELL_SUCCESS_OBJECTIVE,
    EvalRunShellToolErrorAnalysis,
    fetch_eval_run_shell_tool_error_analysis,
)


class SingleModelAdapter(GleanAdapterBase):
    """Optimize prompts for a single student model using shell-tool error evidence."""

    def __init__(
        self,
        runner: ALRunner,
        thresholds: Thresholds,
        student_model: str,
        *,
        bigquery_client: Any | None = None,
        shell_error_lookback_days: int = 7,
        cache_file: str | None = None,
    ):
        if bigquery_client is None:
            raise ValueError("bigquery_client is required")
        self.bigquery_client = bigquery_client
        self.shell_error_lookback_days = shell_error_lookback_days
        super().__init__(
            runner=runner,
            thresholds=thresholds,
            student_model=student_model,
            evaluate_fn=self._evaluate_single_model,
            failure_pattern_fn=self._create_failure_pattern,
            reflective_example_fn=self._build_reflective_example,
            reflection_prompt_fn=self._reflection_prompt,
            reflective_metrics_fn=self._format_reflective_metrics,
            failure_label="HIGH-SIGNAL FAILURES",
            primary_objective=SHELL_SUCCESS_OBJECTIVE,
            default_frontier_type="objective",
            cache_file=cache_file,
        )

    def _get_or_fetch_shell_error_analysis(self, eval_id: str) -> EvalRunShellToolErrorAnalysis:
        cached = self._eval_analysis_cache.get(eval_id)
        if cached is not None:
            print(f"[Cache HIT] Using cached shell error analysis for eval_id: {eval_id}")
            return cached
        analysis = fetch_eval_run_shell_tool_error_analysis(
            self.bigquery_client,
            eval_id=eval_id,
            lookback_days=self.shell_error_lookback_days,
        )
        analysis = enrich_shell_error_action_inputs(self.runner.evalcli, analysis)
        self._eval_analysis_cache[eval_id] = analysis
        self._save_cache()
        return analysis

    def _evaluate_single_model(
        self,
        batch: list[ALDataInst],
        candidate: dict[str, str],
        capture_traces: bool,
    ) -> GleanEvaluationBatch:
        # Always build the same minimal error trajectories so a later trace
        # request can reuse the cached eval run and shell-error analysis.
        result = self._evaluate_with_shell_error_rate(cast(list[SingleModelALDataInst], batch), candidate)
        if capture_traces:
            return result
        return GleanEvaluationBatch(
            outputs=result.outputs,
            scores=result.scores,
            trajectories=None,
            objective_scores=result.objective_scores,
            summary=result.summary,
        )

    def _create_failure_pattern(self, component_name: str, trajectory: SingleModelALTrajectory) -> tuple[Any, ...]:
        output = trajectory["output"]
        shell_success_rate = trajectory.get("objective_scores", {}).get(SHELL_SUCCESS_OBJECTIVE, 1.0)
        return (
            int(shell_success_rate < 0.9),
            int(output.get("student_tool_errors", 0) > 0),
            len(output.get("shell_error_messages", [])),
        )

    def _build_reflective_example(
        self,
        component_name: str,
        trajectory: SingleModelALTrajectory,
        candidate: dict[str, str],
    ) -> ReflectiveExample:
        output = trajectory["output"]
        shell_success_rate = trajectory.get("objective_scores", {}).get(SHELL_SUCCESS_OBJECTIVE, 1.0)
        shell_error_messages = output.get("shell_error_messages", [])
        feedback_parts = []
        if shell_error_messages:
            feedback_parts.append("Recent shell errors: " + "; ".join(shell_error_messages[:5]))
        elif output.get("student_tool_errors", 0) > 0:
            feedback_parts.append(
                f"Tool errors: Student encountered {output.get('student_tool_errors', 0)} shell tool errors."
            )

        inputs: ReflectiveExampleInputs = {
            "eval_set": trajectory["data"]["eval_set_name"],
            "entry_id": output["entry_id"],
            "deployment_id": output["deployment_id"],
            "query": output["query"],
        }
        if eval_run_id := trajectory["data"].get("eval_run_id"):
            inputs["eval_run_id"] = eval_run_id
        if eval_trace_id := trajectory["data"].get("eval_trace_id"):
            inputs["eval_trace_id"] = eval_trace_id

        return {
            "Inputs": inputs,
            "Generated Outputs": {
                "student_answer": "",
                "teacher_answer": "",
                "student_tools": [],
                "teacher_tools": [],
            },
            "Action Inputs": output.get("shell_action_inputs", [])[:5],
            "Execution Errors": shell_error_messages[:5],
            "Feedback": " ".join(feedback_parts) if feedback_parts else "General shell tool reliability issue.",
            "Metrics": {"score": trajectory["score"], "shell_success_rate": shell_success_rate},
        }

    @staticmethod
    def _reflection_prompt(module_name: str) -> str:
        if module_name == "WRITING_CODE":
            return (
                "Focus ONLY on coding instructions that affect shell tool reliability: SDK call patterns, "
                "ToolResult handling, parallelism via asyncio.gather, sandbox rules, and when to print vs extract. "
                "Use shell error examples as evidence. Propose minimal deltas."
            )
        return "Focus only on this module's responsibilities."

    @staticmethod
    def _format_reflective_metrics(metrics: ReflectiveExampleMetrics) -> str | None:
        return None

    def _evaluate_with_shell_error_rate(
        self,
        batch: list[SingleModelALDataInst],
        candidate: dict[str, str],
    ) -> GleanEvaluationBatch[SingleModelALTrajectory, SingleModelALRolloutOutput]:
        if not batch:
            return GleanEvaluationBatch(
                outputs=[],
                scores=[],
                trajectories=None,
                objective_scores=[],
                summary=None,
            )

        system_prompt = compile_encoded_prompt(candidate)
        all_outputs: list[SingleModelALRolloutOutput] = []
        all_scores: list[float] = []
        all_trajectories: list[SingleModelALTrajectory] = []
        all_objective_scores: list[dict[str, float]] = []
        summary_shell_rates: list[float] = []
        total_high_signal_entries = 0

        for al_data_inst in batch:
            eval_set_version = al_data_inst.get("eval_set_version", "")
            eval_set_name = al_data_inst.get("eval_set_name", "")
            deployment_ids = al_data_inst.get("deployment_ids", [])

            student_eval_id = self._get_or_run_student_eval(
                eval_set_name=eval_set_name,
                eval_set_version=eval_set_version,
                deployment_ids=deployment_ids,
                system_prompt=system_prompt,
            )

            analysis = self._get_or_fetch_shell_error_analysis(student_eval_id)
            log_shell_tool_error_analysis(analysis)
            high_signal_entry_ids = analysis.high_signal_entry_ids

            summary_shell_rates.append(analysis.aggregate.shell_success_rate)
            total_high_signal_entries += len(high_signal_entry_ids)

            if not high_signal_entry_ids:
                shell_error_messages = [
                    example.error_str for example in analysis.aggregate.recent_error_examples if example.error_str
                ]
                shell_action_inputs = [
                    example.action_input for example in analysis.aggregate.recent_error_examples if example.action_input
                ]
                output: SingleModelALRolloutOutput = {
                    "deployment_id": deployment_ids[0] if deployment_ids else "",
                    "query": f"{eval_set_name}:{eval_set_version}",
                    "student_tool_calls": analysis.aggregate.shell_executions,
                    "student_tool_errors": analysis.aggregate.shell_errors,
                    "entry_id": f"{eval_set_name}:{eval_set_version}",
                    "shell_error_messages": shell_error_messages,
                    "student_eval_run_id": student_eval_id,
                }
                if shell_action_inputs:
                    output["shell_action_inputs"] = shell_action_inputs
                all_outputs.append(output)
                all_scores.append(analysis.aggregate.shell_success_rate)
                objective_score = {
                    SHELL_SUCCESS_OBJECTIVE: analysis.aggregate.shell_success_rate,
                }
                all_objective_scores.append(objective_score)
                all_trajectories.append(
                    {
                        "data": al_data_inst,
                        "output": output,
                        "score": analysis.aggregate.shell_success_rate,
                        "objective_scores": objective_score,
                    }
                )
                continue

            for entry_id in high_signal_entry_ids:
                entry_metrics = analysis.per_entry[entry_id]
                failed_eval_example = next(
                    (example for example in entry_metrics.recent_error_examples if example.trace_id),
                    None,
                )
                eval_trace_id = failed_eval_example.trace_id if failed_eval_example else None
                eval_trace_examples = [
                    example
                    for example in entry_metrics.recent_error_examples
                    if eval_trace_id is None or example.trace_id == eval_trace_id
                ]
                shell_error_messages = [example.error_str for example in eval_trace_examples if example.error_str]
                shell_action_inputs = [example.action_input for example in eval_trace_examples if example.action_input]
                entry_output: SingleModelALRolloutOutput = {
                    "deployment_id": deployment_ids[0] if deployment_ids else "",
                    "query": f"{eval_set_name}:{eval_set_version} entry={entry_id}",
                    "student_tool_calls": entry_metrics.shell_executions,
                    "student_tool_errors": entry_metrics.shell_errors,
                    "entry_id": entry_id,
                    "shell_error_messages": shell_error_messages,
                    "student_eval_run_id": student_eval_id,
                }
                if shell_action_inputs:
                    entry_output["shell_action_inputs"] = shell_action_inputs
                if eval_trace_id:
                    entry_output["eval_trace_id"] = eval_trace_id
                entry_data: SingleModelALDataInst = {
                    **al_data_inst,
                    "eval_entry_id": entry_id,
                    "eval_run_id": student_eval_id,
                }
                if eval_trace_id:
                    entry_data["eval_trace_id"] = eval_trace_id
                all_outputs.append(entry_output)
                entry_score = entry_metrics.shell_success_rate
                all_scores.append(entry_score)
                entry_objective_score = {
                    SHELL_SUCCESS_OBJECTIVE: entry_metrics.shell_success_rate,
                }
                all_objective_scores.append(entry_objective_score)
                all_trajectories.append(
                    {
                        "data": entry_data,
                        "output": entry_output,
                        "score": entry_score,
                        "objective_scores": entry_objective_score,
                    }
                )

        summary = None
        if summary_shell_rates:
            summary = {
                SHELL_SUCCESS_OBJECTIVE: sum(summary_shell_rates) / len(summary_shell_rates),
                "high_signal_entry_count": float(total_high_signal_entries),
            }

        return GleanEvaluationBatch(
            outputs=all_outputs,
            scores=all_scores,
            trajectories=all_trajectories,
            objective_scores=all_objective_scores,
            summary=summary,
        )


__all__ = [
    "SingleModelALDataInst",
    "SingleModelALRolloutOutput",
    "SingleModelALTrajectory",
    "SingleModelAdapter",
]
