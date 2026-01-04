# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Callback protocol for GEPA optimization instrumentation.

This module provides a callback system for observing GEPA optimization runs.
Callbacks are synchronous, observational (cannot modify state), and receive
full GEPAState access for maximum flexibility.

Example usage:

    class MyCallback:
        def on_optimization_start(self, seed_candidate, trainset_size, valset_size, config):
            print(f"Starting optimization with {trainset_size} training examples")

        def on_iteration_end(self, iteration, state, proposal_accepted):
            print(f"Iteration {iteration}: {'accepted' if proposal_accepted else 'rejected'}")

    result = optimize(
        seed_candidate={"instructions": "..."},
        trainset=data,
        callbacks=[MyCallback()],
        ...
    )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from gepa.core.state import GEPAState

logger = logging.getLogger(__name__)


@runtime_checkable
class GEPACallback(Protocol):
    """Protocol for GEPA optimization callbacks.

    All methods are optional - implement only those you need.
    Callbacks are called synchronously and should not modify the state.
    """

    # =========================================================================
    # Optimization Lifecycle
    # =========================================================================

    def on_optimization_start(
        self,
        seed_candidate: dict[str, str],
        trainset_size: int,
        valset_size: int,
        config: dict[str, Any],
    ) -> None:
        """Called when optimization begins.

        Args:
            seed_candidate: The initial program instructions.
            trainset_size: Number of training examples.
            valset_size: Number of validation examples.
            config: Optimization configuration parameters.
        """
        ...

    def on_optimization_end(
        self,
        best_candidate_idx: int,
        total_iterations: int,
        total_metric_calls: int,
        final_state: GEPAState,
    ) -> None:
        """Called when optimization completes.

        Args:
            best_candidate_idx: Index of the best program found.
            total_iterations: Total number of iterations run.
            total_metric_calls: Total number of metric evaluations.
            final_state: The final optimization state.
        """
        ...

    # =========================================================================
    # Iteration Lifecycle
    # =========================================================================

    def on_iteration_start(
        self,
        iteration: int,
        state: GEPAState,
    ) -> None:
        """Called at the start of each iteration.

        Args:
            iteration: Current iteration number (1-indexed).
            state: Current optimization state.
        """
        ...

    def on_iteration_end(
        self,
        iteration: int,
        state: GEPAState,
        proposal_accepted: bool,
    ) -> None:
        """Called at the end of each iteration.

        Args:
            iteration: Current iteration number (1-indexed).
            state: Current optimization state.
            proposal_accepted: Whether a new candidate was accepted this iteration.
        """
        ...

    # =========================================================================
    # Candidate Selection and Sampling
    # =========================================================================

    def on_candidate_selected(
        self,
        iteration: int,
        candidate_idx: int,
        candidate: dict[str, str],
        score: float,
    ) -> None:
        """Called when a candidate is selected for mutation.

        Args:
            iteration: Current iteration number.
            candidate_idx: Index of the selected candidate.
            candidate: The candidate's instructions.
            score: The candidate's current score.
        """
        ...

    def on_minibatch_sampled(
        self,
        iteration: int,
        minibatch_ids: list[Any],
        trainset_size: int,
    ) -> None:
        """Called when a training minibatch is sampled.

        Args:
            iteration: Current iteration number.
            minibatch_ids: IDs of the sampled training examples.
            trainset_size: Total size of the training set.
        """
        ...

    # =========================================================================
    # Evaluation Events
    # =========================================================================

    def on_evaluation_start(
        self,
        iteration: int,
        candidate_idx: int,
        batch_size: int,
        capture_traces: bool,
    ) -> None:
        """Called before evaluating a candidate.

        Args:
            iteration: Current iteration number.
            candidate_idx: Index of the candidate being evaluated.
            batch_size: Number of examples in the evaluation batch.
            capture_traces: Whether execution traces are being captured.
        """
        ...

    def on_evaluation_end(
        self,
        iteration: int,
        candidate_idx: int,
        scores: list[float],
        has_trajectories: bool,
    ) -> None:
        """Called after evaluating a candidate.

        Args:
            iteration: Current iteration number.
            candidate_idx: Index of the candidate evaluated.
            scores: Per-example scores from the evaluation.
            has_trajectories: Whether trajectories were captured.
        """
        ...

    # =========================================================================
    # Reflection Events
    # =========================================================================

    def on_reflective_dataset_built(
        self,
        iteration: int,
        candidate_idx: int,
        components: list[str],
        dataset: dict[str, list[dict[str, Any]]],
    ) -> None:
        """Called after building the reflective dataset.

        Args:
            iteration: Current iteration number.
            candidate_idx: Index of the candidate.
            components: Names of components being updated.
            dataset: The reflective dataset with Inputs/Outputs/Feedback.
        """
        ...

    def on_proposal_start(
        self,
        iteration: int,
        parent_candidate: dict[str, str],
        components: list[str],
        reflective_dataset: dict[str, list[dict[str, Any]]],
    ) -> None:
        """Called before proposing new instructions.

        Args:
            iteration: Current iteration number.
            parent_candidate: Current instructions being improved.
            components: Names of components being updated.
            reflective_dataset: The feedback data for reflection.
        """
        ...

    def on_proposal_end(
        self,
        iteration: int,
        new_instructions: dict[str, str],
    ) -> None:
        """Called after proposing new instructions.

        Args:
            iteration: Current iteration number.
            new_instructions: The newly proposed instructions.
        """
        ...

    # =========================================================================
    # Acceptance/Rejection Events
    # =========================================================================

    def on_candidate_accepted(
        self,
        iteration: int,
        new_candidate_idx: int,
        new_score: float,
        parent_idx: int,
    ) -> None:
        """Called when a new candidate is accepted.

        Args:
            iteration: Current iteration number.
            new_candidate_idx: Index assigned to the new candidate.
            new_score: The new candidate's validation score.
            parent_idx: Index of the parent candidate.
        """
        ...

    def on_candidate_rejected(
        self,
        iteration: int,
        old_score: float,
        new_score: float,
        reason: str,
    ) -> None:
        """Called when a candidate is rejected.

        Args:
            iteration: Current iteration number.
            old_score: Score of the original candidate.
            new_score: Score of the proposed candidate.
            reason: Explanation for rejection.
        """
        ...

    # =========================================================================
    # Merge Events
    # =========================================================================

    def on_merge_attempted(
        self,
        iteration: int,
        parent_indices: list[int],
        merged_candidate: dict[str, str],
    ) -> None:
        """Called when a merge is attempted.

        Args:
            iteration: Current iteration number.
            parent_indices: Indices of the parent candidates being merged.
            merged_candidate: The merged candidate instructions.
        """
        ...

    def on_merge_accepted(
        self,
        iteration: int,
        new_candidate_idx: int,
        parent_indices: list[int],
    ) -> None:
        """Called when a merge is accepted.

        Args:
            iteration: Current iteration number.
            new_candidate_idx: Index assigned to the merged candidate.
            parent_indices: Indices of the parent candidates.
        """
        ...

    def on_merge_rejected(
        self,
        iteration: int,
        parent_indices: list[int],
        reason: str,
    ) -> None:
        """Called when a merge is rejected.

        Args:
            iteration: Current iteration number.
            parent_indices: Indices of the parent candidates.
            reason: Explanation for rejection.
        """
        ...

    # =========================================================================
    # State Events
    # =========================================================================

    def on_pareto_front_updated(
        self,
        iteration: int,
        new_front: list[int],
        displaced_candidates: list[int],
    ) -> None:
        """Called when the Pareto front is updated.

        Args:
            iteration: Current iteration number.
            new_front: Indices of candidates now on the Pareto front.
            displaced_candidates: Indices of candidates removed from front.
        """
        ...

    def on_state_saved(
        self,
        iteration: int,
        run_dir: str | None,
    ) -> None:
        """Called after state is saved to disk.

        Args:
            iteration: Current iteration number.
            run_dir: Directory where state was saved, or None if not saving.
        """
        ...

    # =========================================================================
    # Budget Tracking
    # =========================================================================

    def on_budget_updated(
        self,
        iteration: int,
        metric_calls_used: int,
        metric_calls_remaining: int | None,
    ) -> None:
        """Called when the evaluation budget is updated.

        Args:
            iteration: Current iteration number.
            metric_calls_used: Total metric calls consumed so far.
            metric_calls_remaining: Remaining calls, or None if unlimited.
        """
        ...

    # =========================================================================
    # Error Handling
    # =========================================================================

    def on_error(
        self,
        iteration: int,
        exception: Exception,
        will_continue: bool,
    ) -> None:
        """Called when an error occurs during optimization.

        Args:
            iteration: Current iteration number.
            exception: The exception that occurred.
            will_continue: Whether optimization will continue after this error.
        """
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
        self.callbacks = callbacks or []

    def add(self, callback: Any) -> None:
        """Add a callback to the composite.

        Args:
            callback: A callback object to add.
        """
        self.callbacks.append(callback)

    def _notify(self, method_name: str, **kwargs: Any) -> None:
        """Notify all callbacks of an event.

        Args:
            method_name: Name of the callback method to invoke.
            **kwargs: Arguments to pass to the callback method.
        """
        for callback in self.callbacks:
            method = getattr(callback, method_name, None)
            if method is not None:
                try:
                    method(**kwargs)
                except Exception as e:
                    logger.warning(f"Callback {callback} failed on {method_name}: {e}")

    # Delegate all callback methods

    def on_optimization_start(self, **kwargs: Any) -> None:
        self._notify("on_optimization_start", **kwargs)

    def on_optimization_end(self, **kwargs: Any) -> None:
        self._notify("on_optimization_end", **kwargs)

    def on_iteration_start(self, **kwargs: Any) -> None:
        self._notify("on_iteration_start", **kwargs)

    def on_iteration_end(self, **kwargs: Any) -> None:
        self._notify("on_iteration_end", **kwargs)

    def on_candidate_selected(self, **kwargs: Any) -> None:
        self._notify("on_candidate_selected", **kwargs)

    def on_minibatch_sampled(self, **kwargs: Any) -> None:
        self._notify("on_minibatch_sampled", **kwargs)

    def on_evaluation_start(self, **kwargs: Any) -> None:
        self._notify("on_evaluation_start", **kwargs)

    def on_evaluation_end(self, **kwargs: Any) -> None:
        self._notify("on_evaluation_end", **kwargs)

    def on_reflective_dataset_built(self, **kwargs: Any) -> None:
        self._notify("on_reflective_dataset_built", **kwargs)

    def on_proposal_start(self, **kwargs: Any) -> None:
        self._notify("on_proposal_start", **kwargs)

    def on_proposal_end(self, **kwargs: Any) -> None:
        self._notify("on_proposal_end", **kwargs)

    def on_candidate_accepted(self, **kwargs: Any) -> None:
        self._notify("on_candidate_accepted", **kwargs)

    def on_candidate_rejected(self, **kwargs: Any) -> None:
        self._notify("on_candidate_rejected", **kwargs)

    def on_merge_attempted(self, **kwargs: Any) -> None:
        self._notify("on_merge_attempted", **kwargs)

    def on_merge_accepted(self, **kwargs: Any) -> None:
        self._notify("on_merge_accepted", **kwargs)

    def on_merge_rejected(self, **kwargs: Any) -> None:
        self._notify("on_merge_rejected", **kwargs)

    def on_pareto_front_updated(self, **kwargs: Any) -> None:
        self._notify("on_pareto_front_updated", **kwargs)

    def on_state_saved(self, **kwargs: Any) -> None:
        self._notify("on_state_saved", **kwargs)

    def on_budget_updated(self, **kwargs: Any) -> None:
        self._notify("on_budget_updated", **kwargs)

    def on_error(self, **kwargs: Any) -> None:
        self._notify("on_error", **kwargs)


def notify_callbacks(
    callbacks: list[Any] | None,
    method_name: str,
    **kwargs: Any,
) -> None:
    """Utility function to notify a list of callbacks.

    This is a convenience function for calling callback methods
    without needing to wrap them in a CompositeCallback.

    Args:
        callbacks: List of callback objects, or None.
        method_name: Name of the callback method to invoke.
        **kwargs: Arguments to pass to the callback method.
    """
    if callbacks is None:
        return

    for callback in callbacks:
        method = getattr(callback, method_name, None)
        if method is not None:
            try:
                method(**kwargs)
            except Exception as e:
                logger.warning(f"Callback {callback} failed on {method_name}: {e}")
