# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""GEPA adapter for Comet Opik datasets and metrics.

Optimizes a system prompt against an Opik dataset, scoring each rollout with an
Opik-style metric (one returning an object with ``value`` and ``reason``). The
metric's ``reason`` becomes the natural-language feedback GEPA's reflection LM
reads. Uses litellm via ``gepa.lm.LM`` for batch chat completion — no dependency
on ``opik_optimizer``; only the lightweight ``opik`` SDK is needed (and only at
call time if the user passes an ``opik.Dataset``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, TypedDict, cast, runtime_checkable

from gepa.core.adapter import EvaluationBatch, GEPAAdapter

# An Opik dataset item is a free-form dict provided by the user's dataset. We
# don't pin its schema — only the field used as the rollout input is configurable.
OpikDataInst = Mapping[str, Any]


class _ScoreResultLike(Protocol):
    """Structural shape of an Opik ``ScoreResult``.

    Opik's evaluation metrics return objects with ``name``, ``value``, and
    optional ``reason``. We accept any object matching this surface, so users
    can plug in custom scorers without importing Opik at all.
    """

    value: float | None
    reason: str | None


@runtime_checkable
class _ChatCompletionCallable(Protocol):
    def __call__(self, messages: Sequence[Mapping[str, str]]) -> str: ...


OpikMetric = Callable[[OpikDataInst, str], "_ScoreResultLike | float | tuple[float, str]"]


class OpikTrajectory(TypedDict):
    data: OpikDataInst
    full_assistant_response: str
    feedback: str


class OpikRolloutOutput(TypedDict):
    full_assistant_response: str


OpikReflectiveRecord = TypedDict(
    "OpikReflectiveRecord",
    {"Inputs": str, "Generated Outputs": str, "Feedback": str},
)


def _coerce_score_and_feedback(result: Any, fallback_feedback: str = "") -> tuple[float, str]:
    """Normalize a metric return into ``(score, feedback)``.

    Accepts:
      - ``opik.evaluation.metrics.score_result.ScoreResult`` (or any duck-typed
        object with ``value`` / ``score`` and optional ``reason``)
      - a bare ``float`` (feedback defaults to ``fallback_feedback``)
      - a ``(score, feedback)`` tuple
    """
    if isinstance(result, tuple) and len(result) == 2:
        score, feedback = result
        return float(score), str(feedback) if feedback is not None else fallback_feedback
    value = getattr(result, "value", None)
    if value is None:
        value = getattr(result, "score", None)
    if value is None:
        # Last resort: try to treat the result as a float-like.
        return float(result), fallback_feedback
    reason = getattr(result, "reason", None)
    if reason is None:
        reason = getattr(result, "metadata", None)
    return float(value), str(reason) if reason is not None else fallback_feedback


def opik_dataset_to_examples(dataset: Any) -> list[dict[str, Any]]:
    """Materialize an ``opik.Dataset`` into a plain list of dict items.

    Useful for handing the dataset to ``gepa.optimize`` as ``trainset`` /
    ``valset``. Falls through if ``dataset`` already supports iteration as
    a list of dicts.
    """
    if hasattr(dataset, "get_items"):
        return [dict(item) for item in dataset.get_items()]
    return [dict(item) for item in dataset]


class OpikAdapter(GEPAAdapter[OpikDataInst, OpikTrajectory, OpikRolloutOutput]):
    """GEPA adapter for system-prompt optimization against an Opik dataset.

    Mirrors the ``GepaOptimizer`` API surface documented at
    https://www.comet.com/docs/opik/development/optimization-runs/algorithms/gepa_optimizer
    but as a native GEPA adapter, so it works through ``gepa.optimize`` without
    depending on the ``opik_optimizer`` package.

    Args:
        model: A litellm model identifier (``"openai/gpt-4o-mini"``,
            ``"anthropic/claude-3-5-sonnet-latest"``, …) **or** any callable
            of the form ``(messages: Sequence[ChatMessage]) -> str`` for users
            who want to plug in a custom chat backend.
        metric: A callable ``(dataset_item, llm_output) -> ScoreResult`` (or
            ``(score, feedback)`` tuple, or bare float). The ``reason`` field
            of the ``ScoreResult`` is fed to GEPA's reflection LM as feedback.
        input_field: The key in each dataset item that holds the user message
            (default: ``"input"``).
        system_fallback: Used when the candidate dict doesn't contain a
            ``system_prompt`` (or ``system`` / ``prompt``) key. Should rarely
            trigger in practice since GEPA always supplies the candidate.
        max_litellm_workers: Concurrency for batch completion.
        litellm_batch_completion_kwargs: Forwarded to ``LM.batch_complete``.
    """

    def __init__(
        self,
        model: str | _ChatCompletionCallable,
        metric: OpikMetric,
        *,
        input_field: str = "input",
        system_fallback: str = "You are a helpful assistant.",
        max_litellm_workers: int = 10,
        litellm_batch_completion_kwargs: dict[str, Any] | None = None,
    ):
        if isinstance(model, str):
            from gepa.lm import LM

            self._lm = LM(model)
        else:
            self._lm = None
        self.model = model
        self.metric = metric
        self.input_field = input_field
        self.system_fallback = system_fallback
        self.max_litellm_workers = max_litellm_workers
        self.litellm_batch_completion_kwargs = litellm_batch_completion_kwargs or {}

    def _resolve_system_text(self, candidate: Mapping[str, str]) -> str:
        for key in ("system_prompt", "system", "prompt"):
            value = candidate.get(key)
            if isinstance(value, str) and value.strip():
                return value
        # Fall back to the first candidate value if none of the standard keys hit.
        if candidate:
            first = next(iter(candidate.values()))
            if isinstance(first, str) and first.strip():
                return first
        return self.system_fallback

    def _resolve_user_text(self, data: OpikDataInst) -> str:
        value = data.get(self.input_field)
        if value is None:
            raise KeyError(
                f"Dataset item is missing the configured input field {self.input_field!r}. "
                f"Set OpikAdapter(input_field=...) to a key that exists in your dataset items."
            )
        return str(value)

    def evaluate(
        self,
        batch: list[OpikDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[OpikTrajectory, OpikRolloutOutput]:
        system_text = self._resolve_system_text(candidate)

        litellm_requests: list[list[Mapping[str, str]]] = []
        for data in batch:
            litellm_requests.append(
                [
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": self._resolve_user_text(data)},
                ]
            )

        if self._lm is not None:
            responses = self._lm.batch_complete(
                litellm_requests,
                max_workers=self.max_litellm_workers,
                **self.litellm_batch_completion_kwargs,
            )
        else:
            model_fn = cast(_ChatCompletionCallable, self.model)
            responses = [model_fn(messages) for messages in litellm_requests]

        outputs: list[OpikRolloutOutput] = []
        scores: list[float] = []
        trajectories: list[OpikTrajectory] | None = [] if capture_traces else None

        for data, response in zip(batch, responses, strict=True):
            metric_result = self.metric(data, response)
            score, feedback = _coerce_score_and_feedback(
                metric_result,
                fallback_feedback=f"Observed score on item; no reason provided by metric.",
            )

            outputs.append({"full_assistant_response": response})
            scores.append(score)
            if trajectories is not None:
                trajectories.append(
                    {
                        "data": data,
                        "full_assistant_response": response,
                        "feedback": feedback,
                    }
                )

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[OpikTrajectory, OpikRolloutOutput],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        trajectories = eval_batch.trajectories
        if trajectories is None:
            raise ValueError("Trajectories are required to build a reflective dataset.")

        items: list[OpikReflectiveRecord] = []
        for traj in trajectories:
            items.append(
                {
                    "Inputs": str(traj["data"].get(self.input_field, "")),
                    "Generated Outputs": traj["full_assistant_response"],
                    "Feedback": traj["feedback"],
                }
            )

        if not items:
            raise ValueError("No trajectories captured; cannot build a reflective dataset.")

        # Independent list per component to avoid the shared-mutable-list bug
        # flagged in #141.
        return {comp: list(items) for comp in components_to_update}
