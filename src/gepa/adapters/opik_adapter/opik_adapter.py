"""
GEPA adapter that evaluates candidates via Opik's dataset/agent hooks.

This module exists so GEPA can directly operate on Opik prompts/datasets without
duplicating the optimizer-side bridge.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from opik import Dataset

from gepa.core.adapter import EvaluationBatch, GEPAAdapter

if TYPE_CHECKING:
    from opik_optimizer.api_objects import chat_prompt
    from opik_optimizer.core.evaluation import evaluate_with_result
    from opik_optimizer.utils.candidate_selection import select_candidate
    from opik_optimizer.agents.optimizable_agent import OptimizableAgent

_IMPORT_ERROR: Exception | None = None
helpers: Any = None
chat_prompt: Any = None
evaluate_with_result: Any = None
select_candidate: Any = None
OptimizableAgent: Any = None

try:
    from opik_optimizer import helpers
    from opik_optimizer.api_objects import chat_prompt
    from opik_optimizer.core.evaluation import evaluate_with_result
    from opik_optimizer.utils.candidate_selection import select_candidate
    from opik_optimizer.agents.optimizable_agent import OptimizableAgent
except Exception as exc:  # pragma: no cover - optional dependency path
    _IMPORT_ERROR = exc

if _IMPORT_ERROR is not None:  # pragma: no cover - optional dependency path
    raise ImportError(
        "gepa.adapters.opik_adapter requires the optional 'opik-optimizer' package."
    ) from _IMPORT_ERROR

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


def _apply_system_text(prompt_obj: chat_prompt.ChatPrompt, system_text: str) -> chat_prompt.ChatPrompt:
    # TODO(opik-adapter-hooks): add prompt-segment hook support (e.g., user/developer/tool
    # segments) once the Opik integration exposes a stable customization interface.
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


InstantiateAgentFn = Callable[[chat_prompt.ChatPrompt, str | None], OptimizableAgent]
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
        validation_dataset: Dataset | None = None,
        experiment_config: dict[str, Any] | None = None,
        system_fallback: str = "You are a helpful assistant.",
        project_name: str | None = None,
        num_threads: int = 1,
        optimizer_metric_tracker: Callable[[], None] | None = None,
        allow_tool_use: bool = True,
        candidate_selection_policy: str | None = None,
    ) -> None:
        self._base_prompt = base_prompt
        self._dataset = dataset
        self._validation_dataset = validation_dataset
        self._metric = metric
        self._instantiate_agent = instantiate_agent
        self._prepare_experiment_config = prepare_experiment_config
        self._experiment_config = experiment_config
        self._system_fallback = system_fallback
        self._project_name = project_name
        self._num_threads = num_threads
        self._metric_tracker = optimizer_metric_tracker or (lambda: None)
        self._metric_name = getattr(metric, "__name__", str(metric))
        self._allow_tool_use = allow_tool_use
        self._candidate_selection_policy = candidate_selection_policy
        self._val_item_ids: set[str] = set()
        if validation_dataset is not None:
            try:
                self._val_item_ids = {
                    str(item.get("id"))
                    for item in validation_dataset.get_items()
                    if item.get("id") is not None
                }
            except Exception:
                self._val_item_ids = set()

    def _resolve_policy(self, prompt_variant: chat_prompt.ChatPrompt) -> str:
        if isinstance(self._candidate_selection_policy, str):
            policy = self._candidate_selection_policy.strip()
            if policy:
                return policy
        model_kwargs = prompt_variant.model_kwargs or {}
        policy = model_kwargs.get("candidate_selection_policy") or model_kwargs.get(
            "selection_policy"
        )
        if isinstance(policy, str) and policy.strip():
            return policy.strip()
        return "best_by_metric"

    def _extract_candidate_logprobs(self, agent: OptimizableAgent, candidates: list[str]) -> list[float] | None:
        logprobs = getattr(agent, "_last_candidate_logprobs", None)
        if not isinstance(logprobs, list) or len(logprobs) != len(candidates):
            return None
        try:
            return [float(v) for v in logprobs]
        except Exception:
            return None

    def _to_score(self, dataset_item: dict[str, Any], output: str) -> float:
        # NOTE: release path keeps GEPA scalar-only. We pass the combined/finalized score to
        # GEPA and intentionally ignore objective vectors to preserve current frontier behavior.
        # FIXME(opik-adapter-multimetric): support opt-in objective_scores propagation once
        # objective frontier mode is rolled out end-to-end.
        metric_result = self._metric(dataset_item, output)
        if hasattr(metric_result, "value"):
            return float(metric_result.value)
        if hasattr(metric_result, "score"):
            return float(metric_result.score)
        return float(metric_result)

    def _resolve_dataset_for_item_ids(self, dataset_item_ids: list[str]) -> Dataset:
        if self._validation_dataset is None or not self._val_item_ids:
            return self._dataset
        if any(item_id in self._val_item_ids for item_id in dataset_item_ids):
            return self._validation_dataset
        return self._dataset

    def _collect_candidates(
        self,
        *,
        agent: OptimizableAgent,
        prompt_variant: chat_prompt.ChatPrompt,
        dataset_item: dict[str, Any],
    ) -> list[str]:
        has_tools = bool(prompt_variant.tools)
        allow_tool_use = self._allow_tool_use and has_tools

        candidates: list[str] = []
        if hasattr(agent, "invoke_agent_candidates"):
            try:
                raw_candidates = agent.invoke_agent_candidates(
                    prompts={"prompt": prompt_variant},
                    dataset_item=dataset_item,
                    allow_tool_use=allow_tool_use,
                )
            except TypeError:
                raw_candidates = agent.invoke_agent_candidates(
                    prompts={"prompt": prompt_variant},
                    dataset_item=dataset_item,
                )
            candidates = [str(c).strip() for c in raw_candidates if c is not None and str(c).strip()]

        if not candidates:
            if hasattr(agent, "invoke_agent"):
                try:
                    single_output = agent.invoke_agent(
                        prompts={"prompt": prompt_variant},
                        dataset_item=dataset_item,
                        allow_tool_use=allow_tool_use,
                    )
                except TypeError:
                    messages = prompt_variant.get_messages(dataset_item)
                    single_output = agent.invoke(messages)
            else:
                messages = prompt_variant.get_messages(dataset_item)
                single_output = agent.invoke(messages)

            if single_output is not None:
                normalized = str(single_output).strip()
                if normalized:
                    candidates = [normalized]

        return candidates

    def _select_output_and_score(
        self,
        *,
        agent: OptimizableAgent,
        prompt_variant: chat_prompt.ChatPrompt,
        dataset_item: dict[str, Any],
        candidates: list[str],
    ) -> tuple[str, float]:
        if not candidates:
            return "", 0.0

        policy = self._resolve_policy(prompt_variant)
        selection = select_candidate(
            candidates=candidates,
            policy=policy,
            metric=lambda item, output: self._metric(item, output),
            dataset_item=dataset_item,
            candidate_logprobs=self._extract_candidate_logprobs(agent, candidates),
            rng=random.Random(0),
        )

        if (
            selection.candidate_scores is not None
            and selection.chosen_index is not None
            and 0 <= selection.chosen_index < len(selection.candidate_scores)
        ):
            return selection.output, float(selection.candidate_scores[selection.chosen_index])

        return selection.output, self._to_score(dataset_item, selection.output)

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

        eval_dataset = self._resolve_dataset_for_item_ids(dataset_item_ids)

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
            dataset=eval_dataset,
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
            trajectories: list[dict[str, Any]] | None = [] if capture_traces else None

            agent = _create_agent()
            for inst in batch:
                dataset_item = inst.opik_item
                candidates = self._collect_candidates(
                    agent=agent,
                    prompt_variant=prompt_variant,
                    dataset_item=dataset_item,
                )
                raw_output, score = self._select_output_and_score(
                    agent=agent,
                    prompt_variant=prompt_variant,
                    dataset_item=dataset_item,
                    candidates=candidates,
                )

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
            return EvaluationBatch(
                outputs=outputs,
                scores=scores,
                trajectories=trajectories,
            )

        if missing_ids:
            logger.debug("Dataset items missing IDs; using local evaluation")
            return _local_evaluation()

        agent = _create_agent()

        def llm_task(dataset_item: dict[str, Any]) -> dict[str, str]:
            candidates = self._collect_candidates(
                agent=agent,
                prompt_variant=prompt_variant,
                dataset_item=dataset_item,
            )
            output, _ = self._select_output_and_score(
                agent=agent,
                prompt_variant=prompt_variant,
                dataset_item=dataset_item,
                candidates=candidates,
            )
            return {"llm_output": output}

        try:
            _, eval_result = evaluate_with_result(
                dataset=eval_dataset,
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
            logger.exception("GEPA OpikAdapter evaluation failed; falling back to local evaluation")
            return _local_evaluation()

        if eval_result is None:
            logger.debug("Opik evaluator returned no results; using local evaluation")
            return _local_evaluation()

        results_by_id = {test_result.test_case.dataset_item_id: test_result for test_result in eval_result.test_results}

        outputs: list[dict[str, Any]] = []
        scores: list[float] = []
        trajectories: list[dict[str, Any]] | None = [] if capture_traces else None

        for inst in batch:
            dataset_item = inst.opik_item
            dataset_item_id = dataset_item.get("id")
            test_result = results_by_id.get(dataset_item_id) if dataset_item_id is not None else None

            output_text = ""
            score_value = 0.0
            if test_result is not None:
                output_text = str(test_result.test_case.task_output.get("llm_output", "")).strip()
                score_result = next(
                    (sr for sr in test_result.score_results if sr.name == self._metric_name),
                    None,
                )
                if score_result is None and test_result.score_results:
                    score_result = test_result.score_results[0]
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

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

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
                feedback = f"Observed score={score:.4f}. Expected answer: {dataset_item.get('answer', '')}"
                yield {
                    "Inputs": {
                        "text": dataset_item.get("input") or dataset_item.get("question") or "",
                    },
                    "Generated Outputs": output_text,
                    "Feedback": feedback,
                }

        reflective_records = list(_records())
        if not reflective_records:
            logger.debug("No trajectories captured for candidate; returning empty reflective dataset")

        return dict.fromkeys(components, reflective_records)
