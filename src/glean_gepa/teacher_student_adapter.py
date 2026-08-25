"""Adapter for teacher-vs-student Glean evaluations."""

from __future__ import annotations

import hashlib
from typing import Any, cast

from glean_gepa.adapter_types import (
    ALDataInst,
    TeacherStudentALDataInst,
    TeacherStudentALRolloutOutput,
    TeacherStudentALTrajectory,
)
from glean_gepa.al_adapter import (
    ALRunner,
    GleanAdapterBase,
    Judge,
    JudgeResult,
    ReflectiveExample,
    ReflectiveExampleInputs,
    ReflectiveExampleMetrics,
    Thresholds,
    extract_tool_names_from_spans,
    get_tool_alignment_from_traces,
)
from glean_gepa.batch import GleanEvaluationBatch
from glean_gepa.prompt import compile_encoded_prompt

PRIMARY_OBJECTIVE = "correctness"


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
        if judge is None:
            raise ValueError("judge is required")
        self.teacher_model = teacher_model
        self.judge = judge
        super().__init__(
            runner=runner,
            thresholds=thresholds,
            student_model=student_model,
            evaluate_fn=self._evaluate_teacher_student,
            failure_pattern_fn=self._create_failure_pattern,
            reflective_example_fn=self._build_reflective_example,
            reflection_prompt_fn=self._reflection_prompt,
            reflective_metrics_fn=self._format_reflective_metrics,
            failure_label="HIGH-SIGNAL FAILURES (teacher vs student)",
            primary_objective=PRIMARY_OBJECTIVE,
            default_frontier_type="hybrid",
            cache_file=cache_file,
        )

    def _extract_per_entry_metrics(
        self, judge_result: JudgeResult, student_eval_id: str, teacher_eval_id: str
    ) -> dict[str, dict[str, float]]:
        """Extract per-entry metrics from judge result traces.

        Returns:
            Dict mapping entry_id -> {correctness, tool_alignment, grounding}
        """
        per_entry_metrics = {}

        if not judge_result.traces:
            return per_entry_metrics

        # Process each entry's traces
        for entry_id, trace_infos in judge_result.traces.items():
            student_trace_info = None
            teacher_trace_info = None

            # Find student and teacher trace infos for this entry
            for trace_info in trace_infos:
                eval_id = trace_info.get("eval_id")
                if eval_id == student_eval_id:
                    student_trace_info = trace_info
                elif eval_id == teacher_eval_id:
                    teacher_trace_info = trace_info

            if not student_trace_info or not teacher_trace_info:
                continue

            # Extract correctness from student trace
            correctness = student_trace_info.get("correctness_score", 0.0)

            # Compute tool alignment from spans if available
            tool_alignment = get_tool_alignment_from_traces(
                student_trace_info.get("spans", []), teacher_trace_info.get("spans", [])
            )

            # Use correctness as proxy for grounding (could be improved with actual grounding metric)
            grounding = correctness

            per_entry_metrics[entry_id] = {
                "correctness": correctness,
                "tool_alignment": tool_alignment,
                "grounding": grounding,
            }

        return per_entry_metrics

    def _evaluate_teacher_student(
        self,
        batch: list[ALDataInst],
        candidate: dict[str, str],
        capture_traces: bool,
    ) -> GleanEvaluationBatch:
        return self._evaluate_with_judge(cast(list[TeacherStudentALDataInst], batch), candidate, capture_traces)

    @staticmethod
    def _check_primary_tool_mismatch(output: TeacherStudentALRolloutOutput) -> bool:
        student_tools = output.get("student_tool_events", [])
        teacher_tools = output.get("teacher_tool_events", [])
        if not student_tools or not teacher_tools:
            return len(student_tools) != len(teacher_tools)
        return student_tools[0] != teacher_tools[0]

    def _create_failure_pattern(self, component_name: str, trajectory: TeacherStudentALTrajectory) -> tuple[Any, ...]:
        output = trajectory["output"]
        correctness = trajectory.get("objective_scores", {}).get("correctness", 1.0)
        return (
            int(correctness < 0.7),
            int(self._check_primary_tool_mismatch(output)),
            int(output.get("student_tool_errors", 0) > 0),
        )

    def _build_reflective_example(
        self,
        component_name: str,
        trajectory: TeacherStudentALTrajectory,
        candidate: dict[str, str],
    ) -> ReflectiveExample:
        output = trajectory["output"]
        objective_scores = trajectory.get("objective_scores", {})
        correctness = objective_scores.get("correctness", trajectory["score"])
        tool_alignment = objective_scores.get("tool_alignment", 0.0)
        student_tools = output.get("student_tool_events", [])
        teacher_tools = output.get("teacher_tool_events", [])
        feedback_parts = []
        if correctness < 0.7:
            feedback_parts.append(f"Correctness issue: Student scored {correctness:.2f} vs teacher baseline.")
        if self._check_primary_tool_mismatch(output):
            feedback_parts.append(f"Tool mismatch: student used {student_tools[:3]} vs teacher {teacher_tools[:3]}.")
        if tool_alignment < 0.7:
            feedback_parts.append(f"Tool alignment issue: score={tool_alignment:.2f}.")

        inputs: ReflectiveExampleInputs = {
            "eval_set": trajectory["data"]["eval_set_name"],
            "entry_id": output["entry_id"],
            "deployment_id": output["deployment_id"],
            "query": output["query"],
        }
        return {
            "Inputs": inputs,
            "Generated Outputs": {
                "student_answer": output.get("student_answer", ""),
                "teacher_answer": output.get("teacher_answer", ""),
                "student_tools": student_tools,
                "teacher_tools": teacher_tools,
            },
            "Action Inputs": [],
            "Execution Errors": [],
            "Feedback": " ".join(feedback_parts) if feedback_parts else "General teacher/student divergence.",
            "Metrics": {"score": trajectory["score"], "correctness": correctness},
        }

    @staticmethod
    def _reflection_prompt(module_name: str) -> str:
        if module_name == "WRITING_CODE":
            return (
                "Focus ONLY on coding instructions: SDK call patterns, ToolResult handling, "
                "parallelism via asyncio.gather, sandbox rules, and when to print vs extract. "
                "Use teacher/student tool divergences as evidence. Propose minimal deltas."
            )
        return "Focus only on this module's responsibilities."

    @staticmethod
    def _format_reflective_metrics(metrics: ReflectiveExampleMetrics) -> str:
        return f"score={metrics['score']:.2f}, correctness={metrics.get('correctness', metrics['score']):.2f}"

    def _evaluate_with_judge(
        self,
        batch: list[TeacherStudentALDataInst],
        candidate: dict[str, str],
        capture_traces: bool,
    ) -> GleanEvaluationBatch[TeacherStudentALTrajectory, TeacherStudentALRolloutOutput]:
        if self.judge is None:
            raise RuntimeError("Judge is required for judge-based evaluation")

        # Handle empty batch
        if not batch:
            return GleanEvaluationBatch(outputs=[], scores=[], trajectories=None, objective_scores=[], summary=None)

        # Compile system prompt from candidate
        system_prompt = compile_encoded_prompt(candidate)

        # Collect results across all eval sets
        all_outputs: list[TeacherStudentALRolloutOutput] = []
        all_scores: list[float] = []
        all_trajectories: list[TeacherStudentALTrajectory] | None = [] if capture_traces else None
        all_objective_scores: list[dict[str, float]] = []

        for al_data_inst in batch:
            eval_set_version = al_data_inst.get("eval_set_version", "")
            eval_set_name = al_data_inst.get("eval_set_name", "")
            deployment_ids = al_data_inst.get("deployment_ids", [])

            # Create cache keys for teacher and student
            teacher_prompt_hash = hashlib.md5(b"<<TEACHER_PROD_PROMPT>>").hexdigest()[:16]
            student_prompt_hash = hashlib.md5(system_prompt.encode()).hexdigest()[:16]

            teacher_cache_key = (eval_set_name, eval_set_version, self.teacher_model, teacher_prompt_hash)
            student_cache_key = (eval_set_name, eval_set_version, self.student_model, student_prompt_hash)

            # Check cache for teacher eval
            teacher_eval_id = self._eval_cache.get(teacher_cache_key)
            if teacher_eval_id:
                print(f"[Cache HIT] Using cached teacher eval_id: {teacher_eval_id}")
            else:
                # Trigger teacher eval run
                teacher_eval_id = self.runner.run(
                    self.teacher_model,
                    system_prompt="<<TEACHER_PROD_PROMPT>>",
                    eval_set_name=eval_set_name,
                    eval_set_version=eval_set_version,
                    deployment_ids=deployment_ids,
                )
                # Cache and save immediately
                self._eval_cache[teacher_cache_key] = teacher_eval_id
                self._save_cache()
                print(f"[Cache MISS] Started and cached teacher eval_id: {teacher_eval_id}")

            # Check cache for student eval
            student_eval_id = self._eval_cache.get(student_cache_key)
            if student_eval_id:
                print(f"[Cache HIT] Using cached student eval_id: {student_eval_id}")
            else:
                # Trigger student eval run
                student_eval_id = self.runner.run(
                    self.student_model,
                    system_prompt=system_prompt,
                    eval_set_name=eval_set_name,
                    eval_set_version=eval_set_version,
                    deployment_ids=deployment_ids,
                )
                # Cache and save immediately
                self._eval_cache[student_cache_key] = student_eval_id
                self._save_cache()
                print(f"[Cache MISS] Started and cached student eval_id: {student_eval_id}")
            # teacher_eval_id = "gepa_gpt_3070257bbe5f1340_1774652253"
            # student_eval_id = "gepa_fast_1ad33e85e6067b04_1774652258"

            # Check if judge has been triggered for this pair
            judge_cache_key = (teacher_eval_id, student_eval_id)
            skip_trigger = judge_cache_key in self._judge_triggered

            if skip_trigger:
                print(f"[Judge Cache HIT] Judge already triggered for {teacher_eval_id} vs {student_eval_id}")
            else:
                print(f"[Judge Cache MISS] Will trigger judge for {teacher_eval_id} vs {student_eval_id}")
                # Mark as triggered and save immediately
                self._judge_triggered.add(judge_cache_key)
                self._save_cache()

            # Run judge to compare teacher vs student
            judge_result = self.judge.judge(teacher_eval_id, student_eval_id, skip_trigger=skip_trigger)

            # Build per-entry metrics map from judge traces
            per_entry_metrics = self._extract_per_entry_metrics(judge_result, student_eval_id, teacher_eval_id)

            # Process each entry in the eval set (from judge traces)
            if not judge_result.traces:
                continue

            for entry_id, trace_infos in judge_result.traces.items():
                # Find student and teacher traces for this entry
                student_trace = None
                teacher_trace = None
                for trace_info in trace_infos:
                    if trace_info["eval_id"] == student_eval_id:
                        student_trace = trace_info
                    elif trace_info["eval_id"] == teacher_eval_id:
                        teacher_trace = trace_info

                if student_trace is None or teacher_trace is None:
                    continue

                # Get per-entry metrics or fall back to aggregate
                entry_metrics = per_entry_metrics.get(
                    entry_id,
                    {
                        "correctness": judge_result.correctness,
                        "tool_alignment": judge_result.tool_alignment,
                        "grounding": judge_result.grounding,
                    },
                )

                # Create comprehensive output with full execution details
                output: TeacherStudentALRolloutOutput = {
                    # Student execution
                    "deployment_id": student_trace["deployment_id"],
                    "query": student_trace["query"],
                    "student_answer": student_trace["answer"],
                    "student_tool_events": extract_tool_names_from_spans(student_trace.get("spans")),
                    "student_loops": student_trace["num_loops"],
                    "student_tool_calls": len(extract_tool_names_from_spans(student_trace.get("spans"))),
                    "student_tool_errors": student_trace["num_tool_errors"],
                    "student_input_tokens": student_trace["input_tokens"],
                    "student_output_tokens": student_trace["output_tokens"],
                    "student_latency_ms": student_trace.get("latency_ms"),
                    # Teacher execution
                    "teacher_answer": teacher_trace["answer"],
                    "teacher_tool_events": extract_tool_names_from_spans(teacher_trace.get("spans")),
                    "teacher_loops": teacher_trace["num_loops"],
                    "teacher_tool_calls": teacher_trace["num_tool_calls"],
                    "teacher_input_tokens": teacher_trace["input_tokens"],
                    "teacher_output_tokens": teacher_trace["output_tokens"],
                    # Metadata
                    "entry_id": entry_id,
                }
                all_outputs.append(output)

                # Create score (weighted combination of metrics)
                score = (
                    0.5 * entry_metrics["correctness"]
                    + 0.3 * entry_metrics["tool_alignment"]
                    + 0.2 * entry_metrics["grounding"]
                )
                all_scores.append(score)

                # Create objective scores for multi-objective optimization
                objective_score = {
                    "correctness": entry_metrics["correctness"],
                    "tool_alignment": entry_metrics["tool_alignment"],
                    "grounding": entry_metrics["grounding"],
                    "tokens": float(student_trace["input_tokens"] + student_trace["output_tokens"]),
                    "loops": float(student_trace["num_loops"]),
                    "tool_errors": float(student_trace["num_tool_errors"]),
                }
                all_objective_scores.append(objective_score)

                # Create trajectory if requested
                if capture_traces and all_trajectories is not None:
                    trajectory: TeacherStudentALTrajectory = {
                        "data": al_data_inst,
                        "output": output,
                        "score": score,
                        "objective_scores": objective_score,
                    }
                    all_trajectories.append(trajectory)

        # Compute summary by averaging objective scores across all dimensions
        summary = None
        if all_objective_scores:
            summary = {}
            # Get all unique dimension names
            all_dims = set()
            for obj_score in all_objective_scores:
                all_dims.update(obj_score.keys())

            # Average each dimension
            for dim in all_dims:
                values = [obj_score.get(dim, 0.0) for obj_score in all_objective_scores if dim in obj_score]
                summary[dim] = sum(values) / len(values) if values else 0.0

        return GleanEvaluationBatch(
            outputs=all_outputs,
            scores=all_scores,
            trajectories=all_trajectories,
            objective_scores=all_objective_scores,
            summary=summary,
        )


__all__ = [
    "TeacherStudentALDataInst",
    "TeacherStudentALRolloutOutput",
    "TeacherStudentALTrajectory",
    "TeacherStudentAdapter",
]
