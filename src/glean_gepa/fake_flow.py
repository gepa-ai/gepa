"""Offline fake evaluation flow for exercising the Glean GEPA run trigger.

This intentionally uses the same GEPA engine and evolutionary proposer as a
real Glean run, while replacing every external dependency with deterministic
in-memory data.
"""

from __future__ import annotations

import re
from typing import Any

from gepa.core.adapter import EvaluationBatch
from glean_gepa.adapter_types import ALDataInst
from glean_gepa.al_adapter import MODULES, Candidate, GleanAdapterBase, ModuleSpec
from glean_gepa.batch import GleanEvaluationBatch
from glean_gepa.prompt import WRITING_CODE_KEY

FAKE_FLOW_MARKER = "[fake-flow iteration="


def fake_evalset(split: str = "train") -> list[ALDataInst]:
    """Return a small, stable eval set that needs progressively better prompts."""
    return [
        {
            "eval_set_name": f"Fake Glean coding eval ({split})",
            "eval_set_version": f"{split}-{index}",
            "deployment_ids": ["fake-deployment"],
            "status": "active",
        }
        for index in range(1, 4)
    ]


def fake_seed_candidate() -> dict[str, str]:
    return {
        WRITING_CODE_KEY: (
            "Write clear, safe code that directly addresses the request. "
            "Explain assumptions and validate important edge cases. "
            "[fake-flow iteration=0]"
        )
    }


def _iteration(candidate: dict[str, str]) -> int:
    match = re.search(r"\[fake-flow iteration=(\d+)\]", candidate.get(WRITING_CODE_KEY, ""))
    return int(match.group(1)) if match else 0


class FakeFlowAdapter(GleanAdapterBase):
    """A deterministic adapter that produces realistic-looking eval outputs."""

    supports_high_signal_eval = False

    def __init__(self) -> None:
        # The base class supplies the same batch and reflection-dataset behavior
        # used in production. These callbacks are never invoked because this class
        # overrides the corresponding public methods below.
        super().__init__(
            runner=None,  # type: ignore[arg-type]
            thresholds=None,  # type: ignore[arg-type]
            student_model="fake-model",
            evaluate_fn=lambda *_args: self.evaluate(*_args),
            failure_pattern_fn=lambda _module, trajectory: (trajectory["output"]["entry_id"],),
            reflective_example_fn=lambda _module, trajectory, _candidate: self._reflective_example(trajectory),
            reflection_prompt_fn=lambda _module: "Improve the fake coding instructions using the failed examples.",
            reflective_metrics_fn=lambda metrics: f"fake score={metrics['score']:.2f}",
            failure_label="Fake eval evidence",
            primary_objective="fake_score",
            default_frontier_type="objective",
        )

    def evaluate(
        self, batch: list[ALDataInst], candidate: dict[str, str], capture_traces: bool = False
    ) -> GleanEvaluationBatch:
        iteration = _iteration(candidate)
        scores = [min(1.0, 0.2 + (0.15 * index) + (0.2 * iteration)) for index, _ in enumerate(batch)]
        outputs: list[dict[str, Any]] = []
        trajectories: list[dict[str, Any]] = []
        eval_run_ids = []
        for index, (data, score) in enumerate(zip(batch, scores, strict=True), start=1):
            entry_id = f"fake-entry-{data['eval_set_version']}"
            eval_run_id = data.get("cached_student_eval_run_id") or f"fake-eval-{iteration}-{index}"
            output = {
                "deployment_id": "fake-deployment",
                "query": f"Fake coding task {index}",
                "entry_id": entry_id,
                "student_tool_calls": 2,
                "student_tool_errors": int(score < 1.0),
                "shell_error_messages": ["fake shell error: add validation"] if score < 1.0 else [],
                "student_eval_run_id": eval_run_id,
            }
            outputs.append(output)
            eval_run_ids.append(
                {
                    "eval_set_name": data["eval_set_name"],
                    "eval_set_version": data["eval_set_version"],
                    "student_eval_run_id": eval_run_id,
                }
            )
            if capture_traces:
                trajectories.append(
                    {
                        "data": data,
                        "output": output,
                        "score": score,
                        "objective_scores": {"fake_score": score},
                    }
                )

        average = sum(scores) / len(scores) if scores else 0.0
        print(
            f"[FAKE FLOW] eval iteration={iteration} entries={len(batch)} "
            f"score={average:.2f} traces={'yes' if capture_traces else 'no'}"
        )
        return GleanEvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories if capture_traces else None,
            objective_scores=[{"fake_score": score} for score in scores],
            num_metric_calls=len(scores),
            summary={"fake_score": average},
            eval_run_ids=eval_run_ids,
        )

    def make_reflective_dataset(
        self,
        candidate: Candidate,
        eval_batch: EvaluationBatch[Any, Any],
        components_to_update: list[str],
        k: int | None,
        error_hamming_distance_k: int | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        del candidate, error_hamming_distance_k
        examples = [self._reflective_example(trajectory) for trajectory in (eval_batch.trajectories or []) if trajectory["score"] < 1.0]
        if k is not None:
            examples = examples[:k]
        return dict.fromkeys(components_to_update, examples)

    def propose_new_texts(
        self,
        reflection_llm: Any,
        candidate: Candidate,
        components_to_update: list[str],
        reflective_examples: list[dict[str, Any]],
        max_variants: int = 3,
    ) -> tuple[list[str], bool]:
        del reflection_llm, components_to_update, reflective_examples, max_variants
        current = candidate.prompt_modules[WRITING_CODE_KEY]
        next_iteration = _iteration(candidate.prompt_modules) + 1
        rewritten = re.sub(
            r"\[fake-flow iteration=\d+\]",
            f"[fake-flow iteration={next_iteration}]",
            current,
        )
        print(f"[FAKE FLOW] proposing fake iteration={next_iteration}")
        return [rewritten], False

    @staticmethod
    def _reflective_example(trajectory: dict[str, Any]) -> dict[str, Any]:
        output = trajectory["output"]
        return {
            "Inputs": {
                "eval_set": f"{trajectory['data']['eval_set_name']}:{trajectory['data']['eval_set_version']}",
                "entry_id": output["entry_id"],
                "deployment_id": output["deployment_id"],
                "query": output["query"],
                "eval_run_id": output["student_eval_run_id"],
            },
            "Generated Outputs": {
                "student_answer": "fake implementation",
                "teacher_answer": "fake expected behavior",
                "student_tools": ["shell"],
                "teacher_tools": ["shell"],
            },
            "Action Inputs": ["pytest tests/fake_case.py"],
            "Execution Errors": output["shell_error_messages"],
            "Feedback": "Fake evaluation feedback: improve validation and error handling.",
            "Metrics": {"score": trajectory["score"], "shell_success_rate": trajectory["score"]},
        }


def build_fake_flow_components() -> tuple[
    dict[str, str], list[ALDataInst], list[ALDataInst], FakeFlowAdapter, dict[str, ModuleSpec]
]:
    """Build all deterministic inputs required by the ordinary GEPA runner."""
    return (
        fake_seed_candidate(),
        fake_evalset("train"),
        fake_evalset("val"),
        FakeFlowAdapter(),
        {name: ModuleSpec(name, "free_text", 1024) for name in MODULES},
    )
