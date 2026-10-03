# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Repeated evaluation with per-example means and delegated reflection."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Generic

from gepa.core.adapter import DataInst, EvaluationBatch, GEPAAdapter, RolloutOutput, Trajectory, invoke_batch_evaluate


@dataclass(frozen=True)
class Rollout(Generic[Trajectory, RolloutOutput]):
    """An original adapter rollout, retained without interpreting its output or trajectory."""

    output: RolloutOutput
    score: float
    trajectory: Trajectory
    objective_scores: dict[str, float] | None = None


class MultiRolloutAdapter(Generic[DataInst, Trajectory, RolloutOutput]):
    """Evaluate each example repeatedly before comparing candidates.

    Args:
        adapter: The existing adapter whose evaluation and reflection semantics are reused.
        num_rollouts: Positive number of independent evaluations per example, including validation.
        reflection_score_threshold: If set, reflect only on original rollouts scoring strictly below it.
            None retains all rollouts and leaves filtering to the wrapped adapter. Set 1.0 for failed-only
            reflection when the metric uses 1.0 as its perfect score.

    Scores and each objective are arithmetic means, never best-of-N scores. Outputs are lists of original
    outputs in rollout order. Trace-enabled evaluation additionally retains every original score and trajectory;
    reflection delegates one original-shaped batch per rollout and concatenates its component records.
    The wrapper does not interpret task-specific traces or concatenate arbitrary feedback strings.

    Optional proposal and checkpoint methods delegate to the underlying adapter. Its optional batch evaluator
    is called once per repetition over all candidate/batch pairs. Repetitions are sequential; concurrency within
    a repetition belongs to the underlying adapter. Seed, mutation, validation, and merge evaluations all count
    the underlying metric calls. Cached results reuse the entire previously measured group of rollouts.
    """

    def __init__(
        self,
        adapter: GEPAAdapter[DataInst, Trajectory, RolloutOutput],
        num_rollouts: int = 3,
        *,
        reflection_score_threshold: float | None = None,
    ):
        if isinstance(num_rollouts, bool) or not isinstance(num_rollouts, int) or num_rollouts < 1:
            raise ValueError("num_rollouts must be a positive integer")
        if reflection_score_threshold is not None and not math.isfinite(reflection_score_threshold):
            raise ValueError("reflection_score_threshold must be finite")
        self.adapter = adapter
        self.num_rollouts = num_rollouts
        self.reflection_score_threshold = reflection_score_threshold

    def __getattr__(self, name: str) -> Any:
        # Preserve the optional proposal/checkpoint capabilities without advertising
        # them on adapters that do not implement them.
        return getattr(object.__getattribute__(self, "adapter"), name)

    def evaluate(
        self, batch: list[DataInst], candidate: dict[str, str], capture_traces: bool = False
    ) -> EvaluationBatch[list[Rollout[Trajectory, RolloutOutput]], list[RolloutOutput]]:
        results = [
            self.adapter.evaluate(batch, candidate, capture_traces=capture_traces) for _ in range(self.num_rollouts)
        ]
        return self._aggregate(results, len(batch), capture_traces)

    def batch_evaluate(
        self, items: list[tuple[dict[str, str], list[DataInst]]], *, capture_traces: bool = True
    ) -> list[EvaluationBatch[list[Rollout[Trajectory, RolloutOutput]], list[RolloutOutput]]]:
        if not items:
            return []
        repetitions = [
            invoke_batch_evaluate(self.adapter, items, capture_traces=capture_traces) for _ in range(self.num_rollouts)
        ]
        if any(len(results) != len(items) for results in repetitions):
            raise ValueError("batch_evaluate must return one evaluation per candidate/batch pair")
        return [
            self._aggregate([results[i] for results in repetitions], len(batch), capture_traces)
            for i, (_, batch) in enumerate(items)
        ]

    def _aggregate(
        self, results: list[EvaluationBatch[Trajectory, RolloutOutput]], size: int, capture_traces: bool
    ) -> EvaluationBatch[list[Rollout[Trajectory, RolloutOutput]], list[RolloutOutput]]:
        for result in results:
            if len(result.outputs) != size or len(result.scores) != size:
                raise ValueError("each rollout must return one output and score per example")
            if capture_traces and (result.trajectories is None or len(result.trajectories) != size):
                raise ValueError("trace-enabled rollouts must return one trajectory per example")
            if result.objective_scores is not None and len(result.objective_scores) != size:
                raise ValueError("each rollout must return one objective map per example")
            if any(not math.isfinite(score) for score in result.scores):
                raise ValueError("rollout scores must be finite")
            if (
                isinstance(result.metric_calls, bool)
                or not isinstance(result.metric_calls, int)
                or result.metric_calls < 0
            ):
                raise ValueError("num_metric_calls must be a nonnegative integer")

        objectives = None
        if any(result.objective_scores is not None for result in results):
            if any(result.objective_scores is None for result in results):
                raise ValueError("all rollouts must provide objective scores when any rollout does")
            objectives = []
            for i in range(size):
                maps = [result.objective_scores[i] for result in results if result.objective_scores is not None]
                keys = maps[0].keys()
                if any(values.keys() != keys for values in maps):
                    raise ValueError("objective names must agree across rollouts of the same example")
                if any(not math.isfinite(value) for values in maps for value in values.values()):
                    raise ValueError("objective scores must be finite")
                objectives.append({key: fmean(values[key] for values in maps) for key in keys})

        trajectories = None
        if capture_traces:
            trajectories = [
                [
                    Rollout(
                        result.outputs[i],
                        result.scores[i],
                        result.trajectories[i],
                        result.objective_scores[i] if result.objective_scores is not None else None,
                    )
                    for result in results
                    if result.trajectories is not None
                ]
                for i in range(size)
            ]
        return EvaluationBatch(
            outputs=[[result.outputs[i] for result in results] for i in range(size)],
            scores=[fmean(result.scores[i] for result in results) for i in range(size)],
            trajectories=trajectories,
            objective_scores=objectives,
            num_metric_calls=sum(result.metric_calls for result in results),
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[list[Rollout[Trajectory, RolloutOutput]], list[RolloutOutput]],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        if eval_batch.trajectories is None:
            raise ValueError("multi-rollout reflection requires captured trajectories")
        if any(len(group) != self.num_rollouts for group in eval_batch.trajectories):
            raise ValueError("each example must retain all rollout trajectories")
        combined: dict[str, list[Mapping[str, Any]]] = {}
        for index in range(self.num_rollouts):
            rollouts = [group[index] for group in eval_batch.trajectories]
            if self.reflection_score_threshold is not None:
                rollouts = [rollout for rollout in rollouts if rollout.score < self.reflection_score_threshold]
            if not rollouts:
                continue
            original_batch = EvaluationBatch(
                outputs=[rollout.output for rollout in rollouts],
                scores=[rollout.score for rollout in rollouts],
                trajectories=[rollout.trajectory for rollout in rollouts],
                objective_scores=(
                    [rollout.objective_scores for rollout in rollouts if rollout.objective_scores is not None]
                    if rollouts[0].objective_scores is not None
                    else None
                ),
            )
            records = self.adapter.make_reflective_dataset(candidate, original_batch, components_to_update)
            for component, rows in records.items():
                combined.setdefault(component, []).extend(rows)
        return combined
