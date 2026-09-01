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
from glean_gepa.batch import EvalRunIds, GleanEvaluationBatch
from glean_gepa.focused_evalset import ensure_focused_eval_set
from glean_gepa.prompt import compile_encoded_prompt
from glean_gepa.reflection_sampling import strip_stdout_sections
from glean_gepa.shell_tool_error_util import (
    SHELL_SUCCESS_OBJECTIVE,
    EvalRunShellToolErrorAnalysis,
    fetch_eval_run_shell_tool_error_analysis,
    fetch_high_signal_evalset_entries,
)


class ShellToolTelemetryPendingError(RuntimeError):
    """Raised when an eval has not yet emitted shell telemetry."""


class SingleModelAdapter(GleanAdapterBase):
    """Optimize prompts for a single student model using shell-tool error evidence."""

    supports_high_signal_eval = True

    def high_signal_batch(self, eval_batch: GleanEvaluationBatch) -> list[ALDataInst]:
        """Keep the parent eval ID required to map trace-side UUIDs to source entries."""
        prepared = super().high_signal_batch(eval_batch)
        source_run_ids: dict[tuple[str, str, tuple[str, ...]], str] = {}
        for trajectory in eval_batch.trajectories or []:
            if trajectory["score"] >= 1.0:
                continue
            data = trajectory["data"]
            output = trajectory["output"]
            source_eval_run_id = data.get("eval_run_id") or output.get("student_eval_run_id")
            if source_eval_run_id:
                key = (data["eval_set_name"], data["eval_set_version"], tuple(data["deployment_ids"]))
                source_run_ids.setdefault(key, source_eval_run_id)

        enriched: list[ALDataInst] = []
        for data in prepared:
            key = (data["eval_set_name"], data["eval_set_version"], tuple(data["deployment_ids"]))
            source_eval_run_id = source_run_ids.get(key)
            enriched.append(
                {**data, "source_eval_run_id": source_eval_run_id} if source_eval_run_id else data
            )
        return enriched

    def prepare_high_signal_batch(self, batch: list[ALDataInst]) -> list[ALDataInst] | None:
        """Upload/reuse focused eval sets once, before concurrent child screening."""
        prepared: list[ALDataInst] = []
        for data in batch:
            entry_ids = data.get("eval_entry_ids")
            if not entry_ids:
                prepared.append(data)
                continue

            source_eval_run_id = data.get("source_eval_run_id")
            if not source_eval_run_id:
                print("[Focused eval set] Missing the parent eval run ID needed to resolve source entries")
                return None
            source_entries = fetch_high_signal_evalset_entries(
                self.bigquery_client,
                eval_set_name=data["eval_set_name"],
                eval_set_version=data["eval_set_version"],
                eval_run_id=source_eval_run_id,
                entry_ids=entry_ids,
                deployment_ids=data["deployment_ids"],
            )
            resolved_entry_ids = sorted({str(entry["id"]) for entry in source_entries})
            missing_entry_ids = sorted(set(entry_ids) - set(resolved_entry_ids))
            if missing_entry_ids:
                print(
                    f"[Focused eval set] Skipping {len(missing_entry_ids)} entries without resolved stt, runId, "
                    f"and traceId: {', '.join(missing_entry_ids)}"
                )
            if not resolved_entry_ids:
                print("[Focused eval set] None of the requested entries could be resolved")
                return None
            focused = ensure_focused_eval_set(
                self.runner.evalcli,
                base_eval_set_name=data["eval_set_name"],
                base_eval_set_version=data["eval_set_version"],
                deployment_ids=data["deployment_ids"],
                entry_ids=resolved_entry_ids,
                source_entries=source_entries,
            )
            if focused is None:
                return None
            prepared.append(
                {
                    **data,
                    "eval_entry_ids": resolved_entry_ids,
                    "eval_set_name": focused.name,
                    "eval_set_version": focused.version,
                    "focused_eval_set_name": focused.name,
                    "focused_eval_set_version": focused.version,
                }
            )
        return prepared

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

    def _get_or_fetch_shell_error_analysis(
        self, eval_id: str, *, include_error_examples: bool = True
    ) -> EvalRunShellToolErrorAnalysis:
        cached = self._eval_analysis_cache.get(eval_id)
        if cached is not None:
            print(f"[Cache HIT] Using cached shell error analysis for eval_id: {eval_id}")
            return cached
        analysis = fetch_eval_run_shell_tool_error_analysis(
            self.bigquery_client,
            eval_id=eval_id,
            lookback_days=self.shell_error_lookback_days,
            include_error_examples=include_error_examples,
        )
        if include_error_examples:
            analysis = enrich_shell_error_action_inputs(self.runner.evalcli, analysis)
            # BigQuery telemetry can arrive after the eval run reports
            # completion (and even in the next UTC shard). A 0/0 result is
            # therefore provisional; persisting it prevents the later, real
            # shell metrics from ever being fetched.
            if analysis.aggregate.shell_executions == 0:
                print(f"[Cache] Not caching provisional 0/0 shell analysis for eval_id: {eval_id}")
                return analysis
            with self._cache_lock:
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
        result = self._evaluate_with_shell_error_rate(
            cast(list[SingleModelALDataInst], batch), candidate, capture_traces=capture_traces
        )
        if capture_traces:
            return result
        return GleanEvaluationBatch(
            outputs=result.outputs,
            scores=result.scores,
            trajectories=None,
            objective_scores=result.objective_scores,
            summary=result.summary,
            eval_run_ids=result.eval_run_ids,
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
        shell_error_messages = [
            sanitized
            for error in output.get("shell_error_messages", [])
            if (sanitized := strip_stdout_sections(error))
        ]
        if shell_error_messages:
            # Keep the concrete text solely in ``Execution Errors``. Repeating
            # it in feedback wastes reflection context without adding signal.
            feedback = "Resolve the shell execution failures shown above."
        elif output.get("student_tool_errors", 0) > 0:
            feedback = (
                f"Tool errors: Student encountered {output.get('student_tool_errors', 0)} shell tool errors."
            )
        else:
            feedback = "General shell tool reliability issue."

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
            "Feedback": feedback,
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
        *,
        capture_traces: bool,
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
        all_eval_run_ids: list[EvalRunIds] = []
        summary_shell_rates: list[float] = []
        total_high_signal_entries = 0

        for al_data_inst in batch:
            base_eval_set_version = al_data_inst.get("eval_set_version", "")
            base_eval_set_name = al_data_inst.get("eval_set_name", "")
            deployment_ids = al_data_inst.get("deployment_ids", [])
            requested_entry_ids = al_data_inst.get("eval_entry_ids")
            eval_set_name = base_eval_set_name
            eval_set_version = base_eval_set_version
            run_label = "gepa"

            if requested_entry_ids:
                focused_eval_set_name = al_data_inst.get("focused_eval_set_name")
                focused_eval_set_version = al_data_inst.get("focused_eval_set_version")
                if focused_eval_set_name and focused_eval_set_version:
                    eval_set_name = focused_eval_set_name
                    eval_set_version = focused_eval_set_version
                else:
                    focused_eval_set = ensure_focused_eval_set(
                        self.runner.evalcli,
                        base_eval_set_name=base_eval_set_name,
                        base_eval_set_version=base_eval_set_version,
                        deployment_ids=deployment_ids,
                        entry_ids=requested_entry_ids,
                    )
                    if focused_eval_set is None:
                        # Do not fall back to the full eval set: a failed focused
                        # setup must not let a candidate bypass the gate.
                        summary_shell_rates.append(0.0)
                        continue
                    eval_set_name = focused_eval_set.name
                    eval_set_version = focused_eval_set.version
                run_label = "gepa_high_signal"

            student_eval_id = al_data_inst.get("cached_student_eval_run_id")
            if student_eval_id:
                print(f"[Child cache HIT] Using cached student eval_id: {student_eval_id} ({run_label})")
            else:
                student_eval_id = self._get_or_run_student_eval(
                    eval_set_name=eval_set_name,
                    eval_set_version=eval_set_version,
                    deployment_ids=deployment_ids,
                    system_prompt=system_prompt,
                    run_label=run_label,
                )
            all_eval_run_ids.append(
                {
                    "eval_set_name": eval_set_name,
                    "eval_set_version": eval_set_version,
                    "student_eval_run_id": student_eval_id,
                }
            )

            is_focused_eval = bool(requested_entry_ids)
            if not is_focused_eval:
                evaluation_kind = "Trace evaluation" if capture_traces else "Validation"
                print(
                    f"[{evaluation_kind}] Reading "
                    f"{'eval-set' if capture_traces else 'full-validation'} shell results for "
                    f"{eval_set_name} {eval_set_version}: {student_eval_id}"
                )
            analysis = self._get_or_fetch_shell_error_analysis(
                student_eval_id,
                include_error_examples=not is_focused_eval,
            )
            if analysis.aggregate.shell_executions == 0:
                raise ShellToolTelemetryPendingError(
                    f"No shell telemetry is available yet for eval {student_eval_id}; refusing to score 0/0 as success"
                )
            if not is_focused_eval:
                log_shell_tool_error_analysis(analysis)
            # Focused eval sets get fresh entry IDs on upload, so trajectories
            # must use the IDs returned by the focused run. The requested source
            # IDs still provide the denominator for the screening rate.
            high_signal_entry_ids = (
                tuple(analysis.per_entry) or tuple(requested_entry_ids)
                if requested_entry_ids
                else analysis.high_signal_entry_ids
            )

            if is_focused_eval:
                passed_entries = sum(
                    1
                    for entry_metrics in analysis.per_entry.values()
                    if entry_metrics.shell_errors == 0
                )
                summary_shell_rates.append(passed_entries / len(requested_entry_ids))
            else:
                summary_shell_rates.append(analysis.aggregate.shell_success_rate)
            total_high_signal_entries += len(requested_entry_ids) if is_focused_eval else len(high_signal_entry_ids)

            # A full validation eval-set item represents the whole eval run,
            # not each of its high-signal entries. The engine has one val ID
            # per configured eval set, so returning entry-level scores here
            # would let its zip() persist an arbitrary entry as the full-val
            # result. Entry-level results are retained only for trace-capturing
            # training evaluations, where reflection needs those examples.
            if not is_focused_eval and not capture_traces:
                output: SingleModelALRolloutOutput = {
                    "deployment_id": deployment_ids[0] if deployment_ids else "",
                    "query": f"{eval_set_name}:{eval_set_version}",
                    "student_tool_calls": analysis.aggregate.shell_executions,
                    "student_tool_errors": analysis.aggregate.shell_errors,
                    "entry_id": f"{eval_set_name}:{eval_set_version}",
                    "shell_error_messages": [
                        example.error_str for example in analysis.aggregate.recent_error_examples if example.error_str
                    ],
                    "student_eval_run_id": student_eval_id,
                }
                shell_action_inputs = [
                    example.action_input for example in analysis.aggregate.recent_error_examples if example.action_input
                ]
                if shell_action_inputs:
                    output["shell_action_inputs"] = shell_action_inputs
                aggregate_score = analysis.aggregate.shell_success_rate
                objective_score = {SHELL_SUCCESS_OBJECTIVE: aggregate_score}
                all_outputs.append(output)
                all_scores.append(aggregate_score)
                all_objective_scores.append(objective_score)
                continue

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
                entry_metrics = analysis.per_entry.get(entry_id, analysis.aggregate)
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
                # Focused screening is entry-level: an entry passes only when it
                # has no tool errors. This keeps the 50% gate independent of the
                # number of shell calls each entry happens to make.
                entry_score = (
                    float(entry_id in analysis.per_entry and entry_metrics.shell_errors == 0)
                    if is_focused_eval
                    else entry_metrics.shell_success_rate
                )
                all_scores.append(entry_score)
                entry_objective_score = {
                    SHELL_SUCCESS_OBJECTIVE: entry_score,
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
            eval_run_ids=all_eval_run_ids,
        )


__all__ = [
    "SingleModelALDataInst",
    "SingleModelALRolloutOutput",
    "SingleModelALTrajectory",
    "SingleModelAdapter",
]
