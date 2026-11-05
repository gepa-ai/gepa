"""Guard native evaluation dispatch without changing score or trace semantics."""

from __future__ import annotations

from typing import Any

from gepa.core.adapter import EvaluationBatch
from gepa.evidence.budget import BudgetExhaustedError, MetricBudgetGuard
from gepa.evidence.config import SequentialEvidenceConfig
from gepa.evidence.observations import TrialContext


class GuardedAdapter:
    """Serialize bounded metric dispatch; reflection methods stay adapter-owned.

    A scorer's declared maximum is a contract, not a runtime call interceptor
    inside the provider. A dishonest adapter can already spend beyond it; a
    reported violation is charged at the reserved maximum and rejected. Native
    proposal/final eval paths use this same object. Batch reservation happens
    before the first call, avoiding partial native batches on budget shortage.
    """

    def __init__(self, adapter: Any, guard: MetricBudgetGuard, config: SequentialEvidenceConfig) -> None:
        self.wrapped = adapter
        self.guard = guard
        self.config = config
        self.purpose = "native"

    def __getattr__(self, name: str) -> Any:
        return getattr(self.wrapped, name)

    def upper_cost(self, size: int) -> int:
        return size * self.config.max_calls_per_example

    def _validate_batch(self, result: EvaluationBatch, size: int, capture_traces: bool) -> int:
        if len(result.scores) != size or len(result.outputs) != size:
            raise ValueError("adapter must return exactly one score and output per scheduled example")
        if capture_traces and (result.trajectories is None or len(result.trajectories) != size):
            raise ValueError("trace-capturing native evaluations require aligned trajectories")
        if result.objective_scores is not None and len(result.objective_scores) != size:
            raise ValueError("objective scores must align with scheduled examples")
        for score in result.scores:
            self.config.validate_score(score)
        cost = result.num_metric_calls if result.num_metric_calls is not None else size
        if isinstance(cost, bool) or not isinstance(cost, int) or not size <= cost <= self.upper_cost(size):
            raise ValueError("fresh adapter result must report between one and the declared maximum calls per example")
        return cost

    def evaluate(self, batch: list, candidate: dict[str, str], capture_traces: bool = False) -> EvaluationBatch:
        return self._evaluate_items([(candidate, batch)], capture_traces=capture_traces)[0]

    def batch_evaluate(
        self, items: list[tuple[dict[str, str], list]], *, capture_traces: bool = True
    ) -> list[EvaluationBatch]:
        return self._evaluate_items(items, capture_traces=capture_traces)

    def _evaluate_items(
        self, items: list[tuple[dict[str, str], list]], *, capture_traces: bool
    ) -> list[EvaluationBatch]:
        upper = sum(self.upper_cost(len(batch)) for _, batch in items)
        if upper == 0:
            return [] if not items else [EvaluationBatch(outputs=[], scores=[]) for _ in items]
        if not self.guard.can_fit(upper):
            raise BudgetExhaustedError(
                f"complete {self.purpose} batch requires {upper}; {self.guard.available} available"
            )
        results = []
        # Coordinator/native API is single-threaded. Deliberately do not invoke
        # an arbitrary adapter batch_evaluate, whose parallel partial failures
        # cannot be accounted individually under this initial contract.
        for candidate, batch in items:
            if not batch:
                results.append(EvaluationBatch(outputs=[], scores=[]))
                continue
            token = self.guard.reserve(self.guard.next_token(self.purpose), self.upper_cost(len(batch)), self.purpose)
            try:
                result = self.wrapped.evaluate(batch, candidate, capture_traces=capture_traces)
                cost = self._validate_batch(result, len(batch), capture_traces)
            except BaseException as error:
                # SystemExit/KeyboardInterrupt in-process are known failures;
                # an actual process kill leaves the durable reservation pending.
                self.guard.fail(token, f"{type(error).__name__}: {error}")
                raise
            self.guard.complete(token, cost)
            results.append(result)
        return results

    def evaluate_replication(
        self, batch: list, candidate: dict[str, str], *, trial_context: TrialContext
    ) -> EvaluationBatch:
        """Dispatch an already reserved evidence side without reserving it twice."""
        replicate = getattr(self.wrapped, "evaluate_replication", None)
        if callable(replicate):
            result = replicate(batch, candidate, trial_context=trial_context, capture_traces=False)
        else:
            result = self.wrapped.evaluate(batch, candidate, capture_traces=False)
        self._validate_batch(result, len(batch), False)
        return result
