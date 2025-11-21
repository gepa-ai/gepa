"""
GEPA adapter that evaluates candidates via Opik's dataset/agent hooks.

This module exists so GEPA can directly operate on Opik prompts/datasets without
duplicating the optimizer-side bridge. It is intentionally lightweight and simply
captures what Opik already provides (metrics, experiment metadata, agent creation)
through injected callables.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import logging

from gepa.core.adapter import EvaluationBatch, GEPAAdapter
from opik import Dataset

from opik_optimizer import helpers
from opik_optimizer.api_objects import chat_prompt
from opik_optimizer.optimizable_agent import OptimizableAgent
from opik_optimizer.task_evaluator import evaluate_with_result

logger = logging.getLogger(__name__)


@dataclass
class OpikDataInst:
    """Data instance handed to GEPA."""

    input_text: str
    answer: str
    additional_context: dict[str, str]
    opik_item: dict[str, Any]


def _extract_system_text(candidate: dict[str, str], fallback: str) -> str:
    for key in ("system_prompt", "system", "prompt"):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return fallback


def _apply_system_text(
    prompt_obj: chat_prompt.ChatPrompt, system_text: str
) -> chat_prompt.ChatPrompt:
    updated = prompt_obj.copy()
    if updated.messages is not None:
        messages = updated.get_messages()
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = system_text
        else:
            messages.insert(0, {"role": "system", "content": system_text})
        updated.set_messages(messages)
    else:
        updated.system = system_text
    return updated


InstantiateAgentFn = Callable[
    [chat_prompt.ChatPrompt, str | None], OptimizableAgent
]
PrepareExperimentFn = Callable[
    [
        chat_prompt.ChatPrompt,
        Dataset,
        Callable[[dict[str, Any], str], Any],
        dict[str, Any] | None,
        dict[str, Any] | None,
    ],
    dict[str, Any],
]


class OpikAdapter(GEPAAdapter[OpikDataInst, dict[str, Any], dict[str, Any]]):
    """GEPA adapter that delegates evaluation to Opik's task evaluator."""

    def __init__(
        self,
        base_prompt: chat_prompt.ChatPrompt,
        dataset: Dataset,
        metric: Callable[[dict[str, Any], str], Any],
        instantiate_agent: InstantiateAgentFn,
        prepare_experiment_config: PrepareExperimentFn,
        *,
        experiment_config: dict[str, Any] | None = None,
        system_fallback: str = "You are a helpful assistant.",
        project_name: str | None = None,
        num_threads: int = 1,
        optimizer_metric_tracker: Callable[[], None] | None = None,
    ) -> None:
        self._base_prompt = base_prompt
        self._dataset = dataset
        self._metric = metric
        self._instantiate_agent = instantiate_agent
        self._prepare_experiment_config = prepare_experiment_config
        self._experiment_config = experiment_config
        self._system_fallback = system_fallback
        self._project_name = project_name
        self._num_threads = num_threads
        self._metric_tracker = optimizer_metric_tracker or (lambda: None)
        self._metric_name = getattr(metric, "__name__", str(metric))

    def evaluate(
        self,
        batch: list[OpikDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[dict[str, Any], dict[str, Any]]:
        system_text = _extract_system_text(candidate, self._system_fallback)
        prompt_variant = _apply_system_text(self._base_prompt, system_text)

        dataset_item_ids: list[str] = []
        missing_ids = False
        for inst in batch:
            dataset_item_id = inst.opik_item.get("id")
            if dataset_item_id is None:
                missing_ids = True
                break
            dataset_item_ids.append(str(dataset_item_id))

        configuration_updates = helpers.drop_none(
            {
                "gepa": helpers.drop_none(
                    {
                        "phase": "candidate",
                        "source": candidate.get("source"),
                        "candidate_id": candidate.get("id"),
                    }
                )
            }
        )
        experiment_config = self._prepare_experiment_config(
            prompt=prompt_variant,
            dataset=self._dataset,
            metric=self._metric,
            experiment_config=self._experiment_config,
            configuration_updates=configuration_updates,
        )
        project_name = experiment_config.get("project_name") or self._project_name

        def _create_agent() -> OptimizableAgent:
            try:
                return self._instantiate_agent(prompt_variant, project_name)
            except TypeError:
                return self._instantiate_agent(prompt_variant, None)

        def _local_evaluation() -> EvaluationBatch[dict[str, Any], dict[str, Any]]:
            outputs: list[dict[str, Any]] = []
            scores: list[float] = []
            trajectories: list[dict[str, Any]] | None = (
                [] if capture_traces else None
            )

            agent = _create_agent()
            for inst in batch:
                dataset_item = inst.opik_item
                messages = prompt_variant.get_messages(dataset_item)
                raw_output = agent.invoke(messages).strip()
                metric_result = self._metric(dataset_item, raw_output)
                if hasattr(metric_result, "value"):
                    score = float(metric_result.value)
                elif hasattr(metric_result, "score"):
                    score = float(metric_result.score)
                else:
                    score = float(metric_result)
                outputs.append({"output": raw_output})
                scores.append(score)
                self._metric_tracker()
                if trajectories is not None:
                    trajectories.append(
                        {
                            "input": dataset_item,
                            "output": raw_output,
                            "score": score,
                        }
                    )
            return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

        if missing_ids:
            logger.debug("Dataset items missing IDs; using local evaluation")
            return _local_evaluation()

        agent = _create_agent()

        def llm_task(dataset_item: dict[str, Any]) -> dict[str, str]:
            messages = prompt_variant.get_messages(dataset_item)
            raw_output = agent.invoke(messages).strip()
            return {"llm_output": raw_output}

        try:
            _, eval_result = evaluate_with_result(
                dataset=self._dataset,
                evaluated_task=llm_task,
                metric=self._metric,
                num_threads=self._num_threads,
                optimization_id=None,
                dataset_item_ids=dataset_item_ids,
                project_name=project_name,
                n_samples=None,
                experiment_config=experiment_config,
                verbose=0,
            )
        except Exception:
            logger.exception(
                "GEPA OpikAdapter evaluation failed; falling back to local evaluation"
            )
            return _local_evaluation()

        if eval_result is None:
            logger.debug("Opik evaluator returned no results; using local evaluation")
            return _local_evaluation()

        results_by_id = {
            test_result.test_case.dataset_item_id: test_result
            for test_result in eval_result.test_results
        }

        outputs: list[dict[str, Any]] = []
        scores: list[float] = []
        trajectories: list[dict[str, Any]] | None = (
            [] if capture_traces else None
        )

        for inst in batch:
            dataset_item = inst.opik_item
            dataset_item_id = dataset_item.get("id")
            test_result = (
                results_by_id.get(dataset_item_id) if dataset_item_id is not None else None
            )

            output_text = ""
            score_value = 0.0
            if test_result is not None:
                output_text = str(
                    test_result.test_case.task_output.get("llm_output", "")
                ).strip()
                score_result = next(
                    (
                        sr
                        for sr in test_result.score_results
                        if sr.name == self._metric_name
                    ),
                    None,
                )
                if score_result is not None and score_result.value is not None:
                    score_value = float(score_result.value)

            outputs.append({"output": output_text})
            scores.append(score_value)
            self._metric_tracker()

            if trajectories is not None:
                trajectories.append(
                    {
                        "input": dataset_item,
                        "output": output_text,
                        "score": score_value,
                    }
                )

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[dict[str, Any], dict[str, Any]],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        components = components_to_update or ["system_prompt"]
        trajectories = eval_batch.trajectories or []

        def _records() -> Iterable[dict[str, Any]]:
            for traj in trajectories:
                dataset_item = traj.get("input", {})
                output_text = traj.get("output", "")
                score = traj.get("score", 0.0)
                feedback = (
                    f"Observed score={score:.4f}. Expected answer: "
                    f"{dataset_item.get('answer', '')}"
                )
                yield {
                    "Inputs": {
                        "text": dataset_item.get("input")
                        or dataset_item.get("question")
                        or "",
                    },
                    "Generated Outputs": output_text,
                    "Feedback": feedback,
                }

        reflective_records = list(_records())
        if not reflective_records:
            logger.debug(
                "No trajectories captured for candidate; returning empty reflective dataset"
            )
            reflective_records = []

        return {component: reflective_records for component in components}
