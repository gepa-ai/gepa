# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Callback protocol for GEPA optimization instrumentation.

This module provides a callback system for observing GEPA optimization runs.
Callbacks are synchronous, observational (cannot modify state), and receive
full GEPAState access for maximum flexibility.

Each callback receives a single event TypedDict containing all parameters.
This allows easy extension via NotRequired fields without breaking changes.

Example usage:

    class MyCallback:
        def on_optimization_start(self, event: OptimizationStartEvent) -> None:
            print(f"Starting optimization with {event['trainset_size']} training examples")

        def on_iteration_end(self, event: IterationEndEvent) -> None:
            status = 'accepted' if event['proposal_accepted'] else 'rejected'
            print(f"Iteration {event['iteration']}: {status}")

    result = optimize(
        seed_candidate={"instructions": "..."},
        trainset=data,
        callbacks=[MyCallback()],
        ...
    )
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, runtime_checkable

if TYPE_CHECKING:
    from gepa.core.state import GEPAState, ProgramIdx

logger = logging.getLogger(__name__)


# =============================================================================
# Event TypedDicts
# =============================================================================
# Each callback receives a single event object containing all parameters.
# Use NotRequired for optional fields when extending in the future.


class OptimizationStartEvent(TypedDict):
    """Event for on_optimization_start callback."""

    seed_candidate: dict[str, str]
    trainset_size: int
    valset_size: int
    config: dict[str, Any]


class OptimizationEndEvent(TypedDict):
    """Event for on_optimization_end callback."""

    best_candidate_idx: int
    total_iterations: int
    total_metric_calls: int
    final_state: GEPAState


class IterationStartEvent(TypedDict):
    """Event for on_iteration_start callback."""

    iteration: int
    state: GEPAState


class IterationEndEvent(TypedDict):
    """Event for on_iteration_end callback."""

    iteration: int
    state: GEPAState
    proposal_accepted: bool


class CandidateSelectedEvent(TypedDict):
    """Event for on_candidate_selected callback."""

    iteration: int
    candidate_idx: int
    candidate: dict[str, str]
    score: float


class MinibatchSampledEvent(TypedDict):
    """Event for on_minibatch_sampled callback."""

    iteration: int
    minibatch_ids: list[Any]
    trainset_size: int


class EvaluationStartEvent(TypedDict):
    """Event for on_evaluation_start callback."""

    iteration: int
    candidate_idx: int | None
    batch_size: int
    capture_traces: bool
    parent_ids: Sequence[ProgramIdx]
    inputs: list[Any]
    is_seed_candidate: bool


class EvaluationEndEvent(TypedDict):
    """Event for on_evaluation_end callback."""

    iteration: int
    candidate_idx: int | None
    scores: list[float]
    has_trajectories: bool
    parent_ids: Sequence[ProgramIdx]
    outputs: list[Any]
    trajectories: list[Any] | None
    objective_scores: list[dict[str, float]] | None
    is_seed_candidate: bool


class EvaluationSkippedEvent(TypedDict):
    """Event for on_evaluation_skipped callback."""

    iteration: int
    candidate_idx: int
    reason: str
    scores: list[float] | None
    is_seed_candidate: bool


class ReflectiveDatasetBuiltEvent(TypedDict):
    """Event for on_reflective_dataset_built callback."""

    iteration: int
    candidate_idx: int
    components: list[str]
    dataset: dict[str, list[dict[str, Any]]]


class ProposalStartEvent(TypedDict):
    """Event for on_proposal_start callback."""

    iteration: int
    parent_candidate: dict[str, str]
    components: list[str]
    reflective_dataset: dict[str, list[dict[str, Any]]]


class ProposalEndEvent(TypedDict):
    """Event for on_proposal_end callback."""

    iteration: int
    new_instructions: dict[str, str]


class CandidateAcceptedEvent(TypedDict):
    """Event for on_candidate_accepted callback."""

    iteration: int
    new_candidate_idx: int
    new_score: float
    parent_ids: Sequence[ProgramIdx]


class CandidateRejectedEvent(TypedDict):
    """Event for on_candidate_rejected callback."""

    iteration: int
    old_score: float
    new_score: float
    reason: str


class MergeAttemptedEvent(TypedDict):
    """Event for on_merge_attempted callback."""

    iteration: int
    parent_ids: Sequence[ProgramIdx]
    merged_candidate: dict[str, str]


class MergeAcceptedEvent(TypedDict):
    """Event for on_merge_accepted callback."""

    iteration: int
    new_candidate_idx: int
    parent_ids: Sequence[ProgramIdx]


class MergeRejectedEvent(TypedDict):
    """Event for on_merge_rejected callback."""

    iteration: int
    parent_ids: Sequence[ProgramIdx]
    reason: str


class ParetoFrontUpdatedEvent(TypedDict):
    """Event for on_pareto_front_updated callback."""

    iteration: int
    new_front: list[int]
    displaced_candidates: list[int]


class ValsetEvaluatedEvent(TypedDict):
    """Event for on_valset_evaluated callback."""

    iteration: int
    candidate_idx: int
    candidate: dict[str, str]
    scores_by_val_id: dict[Any, float]
    average_score: float
    num_examples_evaluated: int
    total_valset_size: int
    parent_ids: Sequence[ProgramIdx]
    is_best_program: bool
    outputs_by_val_id: dict[Any, Any] | None


class StateSavedEvent(TypedDict):
    """Event for on_state_saved callback."""

    iteration: int
    run_dir: str | None


class BudgetUpdatedEvent(TypedDict):
    """Event for on_budget_updated callback."""

    iteration: int
    metric_calls_used: int
    metric_calls_delta: int
    metric_calls_remaining: int | None


class ErrorEvent(TypedDict):
    """Event for on_error callback."""

    iteration: int
    exception: Exception
    will_continue: bool


@runtime_checkable
class GEPACallback(Protocol):
    """Protocol for GEPA optimization callbacks.

    All methods are optional - implement only those you need.
    Callbacks are called synchronously and should not modify the state.
    """

    # =========================================================================
    # Optimization Lifecycle
    # =========================================================================

    def on_optimization_start(self, event: OptimizationStartEvent) -> None:
        """Called when optimization begins."""
        ...

    def on_optimization_end(self, event: OptimizationEndEvent) -> None:
        """Called when optimization completes."""
        ...

    # =========================================================================
    # Iteration Lifecycle
    # =========================================================================

    def on_iteration_start(self, event: IterationStartEvent) -> None:
        """Called at the start of each iteration."""
        ...

    def on_iteration_end(self, event: IterationEndEvent) -> None:
        """Called at the end of each iteration."""
        ...

    # =========================================================================
    # Candidate Selection and Sampling
    # =========================================================================

    def on_candidate_selected(self, event: CandidateSelectedEvent) -> None:
        """Called when a candidate is selected for mutation."""
        ...

    def on_minibatch_sampled(self, event: MinibatchSampledEvent) -> None:
        """Called when a training minibatch is sampled."""
        ...

    # =========================================================================
    # Evaluation Events
    # =========================================================================

    def on_evaluation_start(self, event: EvaluationStartEvent) -> None:
        """Called before evaluating a candidate."""
        ...

    def on_evaluation_end(self, event: EvaluationEndEvent) -> None:
        """Called after evaluating a candidate."""
        ...

    def on_evaluation_skipped(self, event: EvaluationSkippedEvent) -> None:
        """Called when an evaluation is skipped or its results are not used."""
        ...

    def on_valset_evaluated(self, event: ValsetEvaluatedEvent) -> None:
        """Called after a candidate is evaluated on the validation set."""
        ...

    # =========================================================================
    # Reflection Events
    # =========================================================================

    def on_reflective_dataset_built(self, event: ReflectiveDatasetBuiltEvent) -> None:
        """Called after building the reflective dataset."""
        ...

    def on_proposal_start(self, event: ProposalStartEvent) -> None:
        """Called before proposing new instructions."""
        ...

    def on_proposal_end(self, event: ProposalEndEvent) -> None:
        """Called after proposing new instructions."""
        ...

    # =========================================================================
    # Acceptance/Rejection Events
    # =========================================================================

    def on_candidate_accepted(self, event: CandidateAcceptedEvent) -> None:
        """Called when a new candidate is accepted."""
        ...

    def on_candidate_rejected(self, event: CandidateRejectedEvent) -> None:
        """Called when a candidate is rejected."""
        ...

    # =========================================================================
    # Merge Events
    # =========================================================================

    def on_merge_attempted(self, event: MergeAttemptedEvent) -> None:
        """Called when a merge is attempted."""
        ...

    def on_merge_accepted(self, event: MergeAcceptedEvent) -> None:
        """Called when a merge is accepted."""
        ...

    def on_merge_rejected(self, event: MergeRejectedEvent) -> None:
        """Called when a merge is rejected."""
        ...

    # =========================================================================
    # State Events
    # =========================================================================

    def on_pareto_front_updated(self, event: ParetoFrontUpdatedEvent) -> None:
        """Called when the Pareto front is updated."""
        ...

    def on_state_saved(self, event: StateSavedEvent) -> None:
        """Called after state is saved to disk."""
        ...

    # =========================================================================
    # Budget Tracking
    # =========================================================================

    def on_budget_updated(self, event: BudgetUpdatedEvent) -> None:
        """Called when the evaluation budget is updated."""
        ...

    # =========================================================================
    # Error Handling
    # =========================================================================

    def on_error(self, event: ErrorEvent) -> None:
        """Called when an error occurs during optimization."""
        ...


class CompositeCallback:
    """A callback that delegates to multiple child callbacks.

    This allows registering multiple callbacks and having them all
    receive events.

    Example:
        composite = CompositeCallback([callback1, callback2])
        optimize(..., callbacks=[composite])
    """

    def __init__(self, callbacks: list[Any] | None = None):
        """Initialize with a list of callbacks.

        Args:
            callbacks: List of callback objects. Each should implement
                       some or all of the GEPACallback methods.
        """
        self._callbacks: list[Any] = []
        # Cache: method_name -> list of (callback, bound_method) tuples
        self._method_cache: dict[str, list[tuple[Any, Any]]] = {}
        if callbacks:
            for cb in callbacks:
                self.add(cb)

    @property
    def callbacks(self) -> list[Any]:
        """Return the list of registered callbacks."""
        return self._callbacks

    def add(self, callback: Any) -> None:
        """Add a callback to the composite.

        Args:
            callback: A callback object to add.
        """
        self._callbacks.append(callback)
        # Update cache for all known method names
        for method_name in self._method_cache:
            method = getattr(callback, method_name, None)
            if method is not None:
                self._method_cache[method_name].append((callback, method))

    def _notify(self, method_name: str, event: Any) -> None:
        """Notify all callbacks of an event.

        Args:
            method_name: Name of the callback method to invoke.
            event: The event TypedDict to pass to the callback method.
        """
        # Build cache on first access for this method_name
        if method_name not in self._method_cache:
            self._method_cache[method_name] = []
            for callback in self._callbacks:
                method = getattr(callback, method_name, None)
                if method is not None:
                    self._method_cache[method_name].append((callback, method))

        # Use cached methods
        for callback, method in self._method_cache[method_name]:
            try:
                method(event)
            except Exception as e:
                logger.warning(f"Callback {callback} failed on {method_name}: {e}")

    # Delegate all callback methods

    def on_optimization_start(self, event: OptimizationStartEvent) -> None:
        self._notify("on_optimization_start", event)

    def on_optimization_end(self, event: OptimizationEndEvent) -> None:
        self._notify("on_optimization_end", event)

    def on_iteration_start(self, event: IterationStartEvent) -> None:
        self._notify("on_iteration_start", event)

    def on_iteration_end(self, event: IterationEndEvent) -> None:
        self._notify("on_iteration_end", event)

    def on_candidate_selected(self, event: CandidateSelectedEvent) -> None:
        self._notify("on_candidate_selected", event)

    def on_minibatch_sampled(self, event: MinibatchSampledEvent) -> None:
        self._notify("on_minibatch_sampled", event)

    def on_evaluation_start(self, event: EvaluationStartEvent) -> None:
        self._notify("on_evaluation_start", event)

    def on_evaluation_end(self, event: EvaluationEndEvent) -> None:
        self._notify("on_evaluation_end", event)

    def on_evaluation_skipped(self, event: EvaluationSkippedEvent) -> None:
        self._notify("on_evaluation_skipped", event)

    def on_valset_evaluated(self, event: ValsetEvaluatedEvent) -> None:
        self._notify("on_valset_evaluated", event)

    def on_reflective_dataset_built(self, event: ReflectiveDatasetBuiltEvent) -> None:
        self._notify("on_reflective_dataset_built", event)

    def on_proposal_start(self, event: ProposalStartEvent) -> None:
        self._notify("on_proposal_start", event)

    def on_proposal_end(self, event: ProposalEndEvent) -> None:
        self._notify("on_proposal_end", event)

    def on_candidate_accepted(self, event: CandidateAcceptedEvent) -> None:
        self._notify("on_candidate_accepted", event)

    def on_candidate_rejected(self, event: CandidateRejectedEvent) -> None:
        self._notify("on_candidate_rejected", event)

    def on_merge_attempted(self, event: MergeAttemptedEvent) -> None:
        self._notify("on_merge_attempted", event)

    def on_merge_accepted(self, event: MergeAcceptedEvent) -> None:
        self._notify("on_merge_accepted", event)

    def on_merge_rejected(self, event: MergeRejectedEvent) -> None:
        self._notify("on_merge_rejected", event)

    def on_pareto_front_updated(self, event: ParetoFrontUpdatedEvent) -> None:
        self._notify("on_pareto_front_updated", event)

    def on_state_saved(self, event: StateSavedEvent) -> None:
        self._notify("on_state_saved", event)

    def on_budget_updated(self, event: BudgetUpdatedEvent) -> None:
        self._notify("on_budget_updated", event)

    def on_error(self, event: ErrorEvent) -> None:
        self._notify("on_error", event)


def notify_callbacks(
    callbacks: list[Any] | None,
    method_name: str,
    event: Any,
) -> None:
    """Utility function to notify a list of callbacks.

    This is a convenience function for calling callback methods
    without needing to wrap them in a CompositeCallback.

    Args:
        callbacks: List of callback objects, or None.
        method_name: Name of the callback method to invoke.
        event: The event TypedDict to pass to the callback method.
    """
    if callbacks is None:
        return

    for callback in callbacks:
        method = getattr(callback, method_name, None)
        if method is not None:
            try:
                method(event)
            except Exception as e:
                logger.warning(f"Callback {callback} failed on {method_name}: {e}")
