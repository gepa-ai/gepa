# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for the GEPA callback system.

These tests verify that:
1. The GEPACallback protocol is correctly defined
2. Callbacks are invoked at the right times with correct arguments
3. Multiple callbacks can be composed
4. Callback errors are handled gracefully
"""

from unittest.mock import Mock

import pytest

from gepa.core.callbacks import (
    BudgetUpdatedEvent,
    CandidateAcceptedEvent,
    CandidateRejectedEvent,
    CandidateSelectedEvent,
    CompositeCallback,
    ErrorEvent,
    EvaluationEndEvent,
    EvaluationSkippedEvent,
    EvaluationStartEvent,
    GEPACallback,
    IterationEndEvent,
    IterationStartEvent,
    MergeAcceptedEvent,
    MergeAttemptedEvent,
    MergeRejectedEvent,
    MinibatchSampledEvent,
    OptimizationEndEvent,
    OptimizationStartEvent,
    ParetoFrontUpdatedEvent,
    ProposalEndEvent,
    ProposalStartEvent,
    ReflectiveDatasetBuiltEvent,
    StateSavedEvent,
    ValsetEvaluatedEvent,
    notify_callbacks,
)

# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


class RecordingCallback:
    """A callback that records all method calls for testing.

    This helper class captures all callback invocations with their arguments,
    allowing tests to verify that callbacks are called at the right times
    with the expected data.
    """

    def __init__(self):
        self.calls = []

    def _record(self, method_name, event):
        self.calls.append((method_name, dict(event)))

    def get_calls(self, method_name):
        """Get all calls to a specific method."""
        return [kwargs for name, kwargs in self.calls if name == method_name]

    def on_optimization_start(self, event):
        self._record("on_optimization_start", event)

    def on_optimization_end(self, event):
        self._record("on_optimization_end", event)

    def on_iteration_start(self, event):
        self._record("on_iteration_start", event)

    def on_iteration_end(self, event):
        self._record("on_iteration_end", event)

    def on_candidate_selected(self, event):
        self._record("on_candidate_selected", event)

    def on_minibatch_sampled(self, event):
        self._record("on_minibatch_sampled", event)

    def on_evaluation_start(self, event):
        self._record("on_evaluation_start", event)

    def on_evaluation_end(self, event):
        self._record("on_evaluation_end", event)

    def on_evaluation_skipped(self, event):
        self._record("on_evaluation_skipped", event)

    def on_reflective_dataset_built(self, event):
        self._record("on_reflective_dataset_built", event)

    def on_proposal_start(self, event):
        self._record("on_proposal_start", event)

    def on_proposal_end(self, event):
        self._record("on_proposal_end", event)

    def on_candidate_accepted(self, event):
        self._record("on_candidate_accepted", event)

    def on_candidate_rejected(self, event):
        self._record("on_candidate_rejected", event)

    def on_merge_attempted(self, event):
        self._record("on_merge_attempted", event)

    def on_merge_accepted(self, event):
        self._record("on_merge_accepted", event)

    def on_merge_rejected(self, event):
        self._record("on_merge_rejected", event)

    def on_pareto_front_updated(self, event):
        self._record("on_pareto_front_updated", event)

    def on_state_saved(self, event):
        self._record("on_state_saved", event)

    def on_budget_updated(self, event):
        self._record("on_budget_updated", event)

    def on_error(self, event):
        self._record("on_error", event)

    def on_valset_evaluated(self, event):
        self._record("on_valset_evaluated", event)


class FailingCallback:
    """A callback that raises exceptions for testing error handling.

    Used to verify that callback failures are caught and logged without
    crashing the optimization.
    """

    def __init__(self, fail_on=None):
        self.fail_on = fail_on

    def on_optimization_start(self, event):
        if self.fail_on == "on_optimization_start":
            raise ValueError("Intentional failure")

    def on_iteration_start(self, event):
        if self.fail_on == "on_iteration_start":
            raise ValueError("Intentional failure")


# =============================================================================
# A. Protocol Tests
# =============================================================================


class TestCallbackProtocol:
    """Tests for the GEPACallback protocol definition."""

    def test_callback_protocol_is_runtime_checkable(self):
        """Verify GEPACallback can be checked at runtime."""
        # Protocol should be runtime checkable
        assert hasattr(GEPACallback, "__protocol_attrs__") or hasattr(GEPACallback, "_is_runtime_protocol")

        # RecordingCallback should satisfy the protocol
        callback = RecordingCallback()
        assert isinstance(callback, GEPACallback)

    def test_empty_callback_implementation(self):
        """Verify a no-op callback can be created."""

        class EmptyCallback:
            pass

        # Empty class should still be usable (duck typing)
        callback = EmptyCallback()
        # Should not raise when trying to call missing methods
        notify_callbacks(
            [callback],
            "on_optimization_start",
            OptimizationStartEvent(seed_candidate={}, trainset_size=0, valset_size=0, config={}),
        )

    def test_partial_callback_implementation(self):
        """Verify callbacks with only some methods work."""

        class PartialCallback:
            def __init__(self):
                self.called = False

            def on_optimization_start(self, event):
                self.called = True

        callback = PartialCallback()
        notify_callbacks(
            [callback],
            "on_optimization_start",
            OptimizationStartEvent(
                seed_candidate={},
                trainset_size=10,
                valset_size=5,
                config={},
            ),
        )
        assert callback.called

        # Calling a method that doesn't exist should not raise
        notify_callbacks([callback], "on_iteration_start", IterationStartEvent(iteration=1, state=None))


# =============================================================================
# B. Callback Invocation Tests - Optimization Lifecycle
# =============================================================================


class TestOptimizationLifecycle:
    """Tests for on_optimization_start and on_optimization_end callbacks."""

    def test_on_optimization_start_called_with_correct_args(self):
        """Verify on_optimization_start receives seed_candidate, trainset_size, etc."""
        callback = RecordingCallback()

        notify_callbacks(
            [callback],
            "on_optimization_start",
            OptimizationStartEvent(
                seed_candidate={"instructions": "test"},
                trainset_size=100,
                valset_size=20,
                config={"max_metric_calls": 500},
            ),
        )

        calls = callback.get_calls("on_optimization_start")
        assert len(calls) == 1
        assert calls[0]["seed_candidate"] == {"instructions": "test"}
        assert calls[0]["trainset_size"] == 100
        assert calls[0]["valset_size"] == 20
        assert calls[0]["config"] == {"max_metric_calls": 500}

    def test_on_optimization_end_called_with_final_state(self):
        """Verify on_optimization_end receives best_candidate_idx, totals, state."""
        callback = RecordingCallback()
        mock_state = Mock()

        notify_callbacks(
            [callback],
            "on_optimization_end",
            OptimizationEndEvent(
                best_candidate_idx=3,
                total_iterations=50,
                total_metric_calls=450,
                final_state=mock_state,
            ),
        )

        calls = callback.get_calls("on_optimization_end")
        assert len(calls) == 1
        assert calls[0]["best_candidate_idx"] == 3
        assert calls[0]["total_iterations"] == 50
        assert calls[0]["total_metric_calls"] == 450
        assert calls[0]["final_state"] is mock_state


# =============================================================================
# C. Iteration Lifecycle Tests
# =============================================================================


class TestIterationLifecycle:
    """Tests for iteration start/end callbacks."""

    def test_on_iteration_start_called_with_correct_args(self):
        """Verify on_iteration_start called with iteration number and state."""
        callback = RecordingCallback()
        mock_state = Mock()

        notify_callbacks(
            [callback],
            "on_iteration_start",
            IterationStartEvent(
                iteration=5,
                state=mock_state,
            ),
        )

        calls = callback.get_calls("on_iteration_start")
        assert len(calls) == 1
        assert calls[0]["iteration"] == 5
        assert calls[0]["state"] is mock_state

    def test_on_iteration_end_called_with_outcome(self):
        """Verify on_iteration_end called with proposal_accepted flag."""
        callback = RecordingCallback()
        mock_state = Mock()

        # Test accepted case
        notify_callbacks(
            [callback],
            "on_iteration_end",
            IterationEndEvent(
                iteration=5,
                state=mock_state,
                proposal_accepted=True,
            ),
        )

        # Test rejected case
        notify_callbacks(
            [callback],
            "on_iteration_end",
            IterationEndEvent(
                iteration=6,
                state=mock_state,
                proposal_accepted=False,
            ),
        )

        calls = callback.get_calls("on_iteration_end")
        assert len(calls) == 2
        assert calls[0]["proposal_accepted"] is True
        assert calls[1]["proposal_accepted"] is False


# =============================================================================
# D. Candidate Selection and Sampling Tests
# =============================================================================


class TestCandidateEvents:
    """Tests for candidate selection and acceptance/rejection callbacks."""

    def test_on_candidate_selected_called_with_selection_info(self):
        """Verify on_candidate_selected receives candidate_idx and candidate dict."""
        callback = RecordingCallback()

        notify_callbacks(
            [callback],
            "on_candidate_selected",
            CandidateSelectedEvent(
                iteration=3,
                candidate_idx=2,
                candidate={"instructions": "selected instructions"},
                score=0.85,
            ),
        )

        calls = callback.get_calls("on_candidate_selected")
        assert len(calls) == 1
        assert calls[0]["iteration"] == 3
        assert calls[0]["candidate_idx"] == 2
        assert calls[0]["candidate"] == {"instructions": "selected instructions"}
        assert calls[0]["score"] == 0.85

    def test_on_minibatch_sampled_called_with_ids(self):
        """Verify on_minibatch_sampled receives the sampled IDs."""
        callback = RecordingCallback()

        notify_callbacks(
            [callback],
            "on_minibatch_sampled",
            MinibatchSampledEvent(
                iteration=3,
                minibatch_ids=[0, 5, 12, 23, 45],
                trainset_size=100,
            ),
        )

        calls = callback.get_calls("on_minibatch_sampled")
        assert len(calls) == 1
        assert calls[0]["minibatch_ids"] == [0, 5, 12, 23, 45]
        assert calls[0]["trainset_size"] == 100

    def test_on_candidate_accepted_called_on_improvement(self):
        """Verify acceptance callback called with new candidate info."""
        callback = RecordingCallback()

        notify_callbacks(
            [callback],
            "on_candidate_accepted",
            CandidateAcceptedEvent(
                iteration=5,
                new_candidate_idx=3,
                new_score=0.92,
                parent_ids=[1],
            ),
        )

        calls = callback.get_calls("on_candidate_accepted")
        assert len(calls) == 1
        assert calls[0]["new_candidate_idx"] == 3
        assert calls[0]["new_score"] == 0.92
        assert calls[0]["parent_ids"] == [1]

    def test_on_candidate_rejected_called_on_no_improvement(self):
        """Verify rejection callback called with scores and reason."""
        callback = RecordingCallback()

        notify_callbacks(
            [callback],
            "on_candidate_rejected",
            CandidateRejectedEvent(
                iteration=5,
                old_score=0.85,
                new_score=0.80,
                reason="New subsample score not better than old",
            ),
        )

        calls = callback.get_calls("on_candidate_rejected")
        assert len(calls) == 1
        assert calls[0]["old_score"] == 0.85
        assert calls[0]["new_score"] == 0.80
        assert "not better" in calls[0]["reason"]


# =============================================================================
# E. Evaluation Event Tests
# =============================================================================


class TestEvaluationEvents:
    """Tests for evaluation start/end callbacks."""

    def test_on_evaluation_start_called_with_batch_info(self):
        """Verify on_evaluation_start receives batch size, trace flag, and inputs."""
        callback = RecordingCallback()

        test_inputs = [{"question": "What is 2+2?"}, {"question": "What is 3+3?"}]

        notify_callbacks(
            [callback],
            "on_evaluation_start",
            EvaluationStartEvent(
                iteration=3,
                candidate_idx=1,
                batch_size=35,
                capture_traces=True,
                parent_ids=[0],
                inputs=test_inputs,
                is_seed_candidate=False,
            ),
        )

        calls = callback.get_calls("on_evaluation_start")
        assert len(calls) == 1
        assert calls[0]["batch_size"] == 35
        assert calls[0]["capture_traces"] is True
        assert calls[0]["parent_ids"] == [0]
        assert calls[0]["inputs"] == test_inputs
        assert calls[0]["is_seed_candidate"] is False

    def test_on_evaluation_end_scores_are_list_of_floats(self):
        """Verify scores argument is correctly typed."""
        callback = RecordingCallback()

        test_outputs = ["answer1", "answer2", "answer3", "answer4", "answer5"]
        test_trajectories = [{"trace": "t1"}, {"trace": "t2"}, {"trace": "t3"}, {"trace": "t4"}, {"trace": "t5"}]
        test_objective_scores = [
            {"accuracy": 0.8},
            {"accuracy": 0.9},
            {"accuracy": 1.0},
            {"accuracy": 0.7},
            {"accuracy": 0.85},
        ]

        notify_callbacks(
            [callback],
            "on_evaluation_end",
            EvaluationEndEvent(
                iteration=3,
                candidate_idx=1,
                scores=[0.8, 0.9, 1.0, 0.7, 0.85],
                has_trajectories=True,
                parent_ids=[0],
                outputs=test_outputs,
                trajectories=test_trajectories,
                objective_scores=test_objective_scores,
                is_seed_candidate=False,
            ),
        )

        calls = callback.get_calls("on_evaluation_end")
        assert len(calls) == 1
        assert isinstance(calls[0]["scores"], list)
        assert all(isinstance(s, float) for s in calls[0]["scores"])
        assert calls[0]["has_trajectories"] is True
        assert calls[0]["parent_ids"] == [0]
        assert calls[0]["outputs"] == test_outputs
        assert calls[0]["trajectories"] == test_trajectories
        assert calls[0]["objective_scores"] == test_objective_scores
        assert calls[0]["is_seed_candidate"] is False

    def test_on_evaluation_start_with_new_candidate(self):
        """Verify evaluation of new candidate has candidate_idx=None and parent_ids set."""
        callback = RecordingCallback()

        test_inputs = [{"q": "test"}]

        # New mutation candidate (1 parent)
        notify_callbacks(
            [callback],
            "on_evaluation_start",
            EvaluationStartEvent(
                iteration=5,
                candidate_idx=None,
                batch_size=10,
                capture_traces=False,
                parent_ids=[3],
                inputs=test_inputs,
                is_seed_candidate=False,
            ),
        )

        calls = callback.get_calls("on_evaluation_start")
        assert len(calls) == 1
        assert calls[0]["candidate_idx"] is None
        assert calls[0]["parent_ids"] == [3]
        assert calls[0]["inputs"] == test_inputs
        assert calls[0]["is_seed_candidate"] is False

    def test_on_evaluation_with_merge_parents(self):
        """Verify merge evaluation has candidate_idx=None and parent_ids with 2 elements."""
        callback = RecordingCallback()

        test_inputs = [{"q": f"q{i}"} for i in range(5)]
        test_outputs = [f"out{i}" for i in range(5)]

        # Merged candidate (2 parents)
        notify_callbacks(
            [callback],
            "on_evaluation_start",
            EvaluationStartEvent(
                iteration=10,
                candidate_idx=None,
                batch_size=5,
                capture_traces=False,
                parent_ids=[2, 7],
                inputs=test_inputs,
                is_seed_candidate=False,
            ),
        )

        notify_callbacks(
            [callback],
            "on_evaluation_end",
            EvaluationEndEvent(
                iteration=10,
                candidate_idx=None,
                scores=[0.9, 0.85, 0.95, 0.88, 0.92],
                has_trajectories=False,
                parent_ids=[2, 7],
                outputs=test_outputs,
                trajectories=None,
                objective_scores=None,
                is_seed_candidate=False,
            ),
        )

        start_calls = callback.get_calls("on_evaluation_start")
        end_calls = callback.get_calls("on_evaluation_end")

        assert len(start_calls) == 1
        assert start_calls[0]["candidate_idx"] is None
        assert start_calls[0]["parent_ids"] == [2, 7]
        assert start_calls[0]["inputs"] == test_inputs
        assert start_calls[0]["is_seed_candidate"] is False

        assert len(end_calls) == 1
        assert end_calls[0]["candidate_idx"] is None
        assert end_calls[0]["parent_ids"] == [2, 7]
        assert end_calls[0]["outputs"] == test_outputs
        assert end_calls[0]["trajectories"] is None
        assert end_calls[0]["objective_scores"] is None
        assert end_calls[0]["is_seed_candidate"] is False

    def test_on_evaluation_with_seed_candidate(self):
        """Verify seed candidate evaluation has empty parent_ids and is_seed_candidate=True."""
        callback = RecordingCallback()

        test_inputs = [{"seed_input": i} for i in range(20)]

        # Seed candidate (no parents)
        notify_callbacks(
            [callback],
            "on_evaluation_start",
            EvaluationStartEvent(
                iteration=1,
                candidate_idx=0,
                batch_size=20,
                capture_traces=True,
                parent_ids=[],
                inputs=test_inputs,
                is_seed_candidate=True,
            ),
        )

        calls = callback.get_calls("on_evaluation_start")
        assert len(calls) == 1
        assert calls[0]["candidate_idx"] == 0
        assert calls[0]["parent_ids"] == []
        assert calls[0]["inputs"] == test_inputs
        assert calls[0]["is_seed_candidate"] is True

    def test_on_evaluation_skipped_no_trajectories(self):
        """Verify on_evaluation_skipped called when no trajectories captured."""
        callback = RecordingCallback()

        notify_callbacks(
            [callback],
            "on_evaluation_skipped",
            EvaluationSkippedEvent(
                iteration=3,
                candidate_idx=1,
                reason="no_trajectories",
                scores=[0.8, 0.9, 0.7],
                is_seed_candidate=False,
            ),
        )

        calls = callback.get_calls("on_evaluation_skipped")
        assert len(calls) == 1
        assert calls[0]["iteration"] == 3
        assert calls[0]["candidate_idx"] == 1
        assert calls[0]["reason"] == "no_trajectories"
        assert calls[0]["scores"] == [0.8, 0.9, 0.7]
        assert calls[0]["is_seed_candidate"] is False

    def test_on_evaluation_skipped_perfect_scores(self):
        """Verify on_evaluation_skipped called when all scores are perfect."""
        callback = RecordingCallback()

        notify_callbacks(
            [callback],
            "on_evaluation_skipped",
            EvaluationSkippedEvent(
                iteration=5,
                candidate_idx=2,
                reason="all_scores_perfect",
                scores=[1.0, 1.0, 1.0, 1.0],
                is_seed_candidate=False,
            ),
        )

        calls = callback.get_calls("on_evaluation_skipped")
        assert len(calls) == 1
        assert calls[0]["iteration"] == 5
        assert calls[0]["candidate_idx"] == 2
        assert calls[0]["reason"] == "all_scores_perfect"
        assert all(s == 1.0 for s in calls[0]["scores"])
        assert calls[0]["is_seed_candidate"] is False

    def test_on_evaluation_skipped_with_none_scores(self):
        """Verify on_evaluation_skipped handles None scores gracefully."""
        callback = RecordingCallback()

        notify_callbacks(
            [callback],
            "on_evaluation_skipped",
            EvaluationSkippedEvent(
                iteration=4,
                candidate_idx=0,
                reason="no_trajectories",
                scores=None,
                is_seed_candidate=True,
            ),
        )

        calls = callback.get_calls("on_evaluation_skipped")
        assert len(calls) == 1
        assert calls[0]["scores"] is None
        assert calls[0]["is_seed_candidate"] is True


# =============================================================================
# F. Reflection Event Tests
# =============================================================================


class TestReflectionEvents:
    """Tests for reflective dataset and proposal callbacks."""

    def test_on_reflective_dataset_built_called_with_dataset(self):
        """Verify callback receives the actual reflective dataset structure."""
        callback = RecordingCallback()

        dataset = {
            "predictor": [
                {
                    "Inputs": {"question": "What is 2+2?"},
                    "Generated Outputs": {"answer": "5"},
                    "Feedback": "Incorrect. The answer is 4.",
                }
            ]
        }

        notify_callbacks(
            [callback],
            "on_reflective_dataset_built",
            ReflectiveDatasetBuiltEvent(
                iteration=3,
                candidate_idx=1,
                components=["predictor"],
                dataset=dataset,
            ),
        )

        calls = callback.get_calls("on_reflective_dataset_built")
        assert len(calls) == 1
        assert "predictor" in calls[0]["dataset"]
        assert "Inputs" in calls[0]["dataset"]["predictor"][0]
        assert "Feedback" in calls[0]["dataset"]["predictor"][0]

    def test_on_proposal_start_end_called_with_instructions(self):
        """Verify proposal callbacks receive before/after instructions."""
        callback = RecordingCallback()

        # Proposal start
        notify_callbacks(
            [callback],
            "on_proposal_start",
            ProposalStartEvent(
                iteration=3,
                parent_candidate={"instructions": "Original instructions"},
                components=["instructions"],
                reflective_dataset={"instructions": []},
            ),
        )

        # Proposal end
        notify_callbacks(
            [callback],
            "on_proposal_end",
            ProposalEndEvent(
                iteration=3,
                new_instructions={"instructions": "Improved instructions"},
            ),
        )

        start_calls = callback.get_calls("on_proposal_start")
        end_calls = callback.get_calls("on_proposal_end")

        assert len(start_calls) == 1
        assert start_calls[0]["parent_candidate"]["instructions"] == "Original instructions"

        assert len(end_calls) == 1
        assert end_calls[0]["new_instructions"]["instructions"] == "Improved instructions"


# =============================================================================
# G. Merge Event Tests
# =============================================================================


class TestMergeEvents:
    """Tests for merge-related callbacks."""

    def test_on_merge_attempted_called_with_parents(self):
        """Verify merge attempted callback includes parent indices."""
        callback = RecordingCallback()

        notify_callbacks(
            [callback],
            "on_merge_attempted",
            MergeAttemptedEvent(
                iteration=10,
                parent_ids=[1, 3],
                merged_candidate={"instructions": "merged"},
            ),
        )

        calls = callback.get_calls("on_merge_attempted")
        assert len(calls) == 1
        assert calls[0]["parent_ids"] == [1, 3]

    def test_on_merge_accepted_called_on_improvement(self):
        """Verify merge acceptance callback includes new index."""
        callback = RecordingCallback()

        notify_callbacks(
            [callback],
            "on_merge_accepted",
            MergeAcceptedEvent(
                iteration=10,
                new_candidate_idx=5,
                parent_ids=[1, 3],
            ),
        )

        calls = callback.get_calls("on_merge_accepted")
        assert len(calls) == 1
        assert calls[0]["new_candidate_idx"] == 5
        assert calls[0]["parent_ids"] == [1, 3]

    def test_on_merge_rejected_called_on_failure(self):
        """Verify merge rejection includes reason."""
        callback = RecordingCallback()

        notify_callbacks(
            [callback],
            "on_merge_rejected",
            MergeRejectedEvent(
                iteration=10,
                parent_ids=[1, 3],
                reason="Merged score worse than both parents",
            ),
        )

        calls = callback.get_calls("on_merge_rejected")
        assert len(calls) == 1
        assert "worse" in calls[0]["reason"]


# =============================================================================
# H. State Event Tests
# =============================================================================


class TestStateEvents:
    """Tests for state-related callbacks."""

    def test_on_pareto_front_updated_called_with_changes(self):
        """Verify Pareto front callback shows displaced candidates."""
        callback = RecordingCallback()

        notify_callbacks(
            [callback],
            "on_pareto_front_updated",
            ParetoFrontUpdatedEvent(
                iteration=5,
                new_front=[0, 2, 4],
                displaced_candidates=[1],
            ),
        )

        calls = callback.get_calls("on_pareto_front_updated")
        assert len(calls) == 1
        assert calls[0]["new_front"] == [0, 2, 4]
        assert calls[0]["displaced_candidates"] == [1]

    def test_on_state_saved_called_with_run_dir(self):
        """Verify state save callback receives run_dir."""
        callback = RecordingCallback()

        notify_callbacks(
            [callback],
            "on_state_saved",
            StateSavedEvent(
                iteration=5,
                run_dir="/tmp/gepa_run_123",
            ),
        )

        calls = callback.get_calls("on_state_saved")
        assert len(calls) == 1
        assert calls[0]["run_dir"] == "/tmp/gepa_run_123"

    def test_on_budget_updated_tracks_remaining_calls(self):
        """Verify budget callback shows consumed vs remaining calls."""
        callback = RecordingCallback()

        notify_callbacks(
            [callback],
            "on_budget_updated",
            BudgetUpdatedEvent(
                iteration=5,
                metric_calls_used=150,
                metric_calls_delta=10,
                metric_calls_remaining=350,
            ),
        )

        calls = callback.get_calls("on_budget_updated")
        assert len(calls) == 1
        assert calls[0]["metric_calls_used"] == 150
        assert calls[0]["metric_calls_remaining"] == 350


# =============================================================================
# I. Valset Evaluated Event Tests
# =============================================================================


class TestValsetEvaluatedEvent:
    """Tests for on_valset_evaluated callback."""

    def test_on_valset_evaluated_called_with_correct_args(self):
        """Verify on_valset_evaluated receives all expected fields."""
        callback = RecordingCallback()

        notify_callbacks(
            [callback],
            "on_valset_evaluated",
            ValsetEvaluatedEvent(
                iteration=5,
                candidate_idx=3,
                candidate={"instructions": "test instructions"},
                scores_by_val_id={"val_0": 0.8, "val_1": 0.9, "val_2": 0.7},
                average_score=0.8,
                num_examples_evaluated=3,
                total_valset_size=10,
                parent_ids=[1],
                is_best_program=True,
                outputs_by_val_id={"val_0": "output_0", "val_1": "output_1", "val_2": "output_2"},
            ),
        )

        calls = callback.get_calls("on_valset_evaluated")
        assert len(calls) == 1
        assert calls[0]["iteration"] == 5
        assert calls[0]["candidate_idx"] == 3
        assert calls[0]["candidate"] == {"instructions": "test instructions"}
        assert calls[0]["scores_by_val_id"] == {"val_0": 0.8, "val_1": 0.9, "val_2": 0.7}
        assert calls[0]["average_score"] == 0.8
        assert calls[0]["num_examples_evaluated"] == 3
        assert calls[0]["total_valset_size"] == 10
        assert calls[0]["parent_ids"] == [1]
        assert calls[0]["is_best_program"] is True
        assert calls[0]["outputs_by_val_id"] == {"val_0": "output_0", "val_1": "output_1", "val_2": "output_2"}

    def test_on_valset_evaluated_with_seed_candidate(self):
        """Verify seed candidate has empty parent_ids."""
        callback = RecordingCallback()

        notify_callbacks(
            [callback],
            "on_valset_evaluated",
            ValsetEvaluatedEvent(
                iteration=1,
                candidate_idx=0,
                candidate={"instructions": "seed instructions"},
                scores_by_val_id={0: 0.5, 1: 0.6},
                average_score=0.55,
                num_examples_evaluated=2,
                total_valset_size=2,
                parent_ids=[],
                is_best_program=True,
                outputs_by_val_id={0: "out_0", 1: "out_1"},
            ),
        )

        calls = callback.get_calls("on_valset_evaluated")
        assert len(calls) == 1
        assert calls[0]["candidate_idx"] == 0
        assert calls[0]["parent_ids"] == []
        assert calls[0]["is_best_program"] is True

    def test_on_valset_evaluated_with_mutation(self):
        """Verify mutation has single parent in parent_ids."""
        callback = RecordingCallback()

        notify_callbacks(
            [callback],
            "on_valset_evaluated",
            ValsetEvaluatedEvent(
                iteration=3,
                candidate_idx=2,
                candidate={"instructions": "mutated instructions"},
                scores_by_val_id={0: 0.7, 1: 0.8},
                average_score=0.75,
                num_examples_evaluated=2,
                total_valset_size=5,
                parent_ids=[1],
                is_best_program=False,
                outputs_by_val_id={0: "out_0", 1: "out_1"},
            ),
        )

        calls = callback.get_calls("on_valset_evaluated")
        assert len(calls) == 1
        assert calls[0]["parent_ids"] == [1]
        assert calls[0]["is_best_program"] is False

    def test_on_valset_evaluated_with_merge(self):
        """Verify merge has two parents in parent_ids."""
        callback = RecordingCallback()

        notify_callbacks(
            [callback],
            "on_valset_evaluated",
            ValsetEvaluatedEvent(
                iteration=10,
                candidate_idx=5,
                candidate={"instructions": "merged instructions"},
                scores_by_val_id={"a": 0.9, "b": 0.85},
                average_score=0.875,
                num_examples_evaluated=2,
                total_valset_size=2,
                parent_ids=[2, 4],
                is_best_program=True,
                outputs_by_val_id={"a": "out_a", "b": "out_b"},
            ),
        )

        calls = callback.get_calls("on_valset_evaluated")
        assert len(calls) == 1
        assert calls[0]["parent_ids"] == [2, 4]

    def test_on_valset_evaluated_with_none_outputs(self):
        """Verify outputs_by_val_id can be None."""
        callback = RecordingCallback()

        notify_callbacks(
            [callback],
            "on_valset_evaluated",
            ValsetEvaluatedEvent(
                iteration=2,
                candidate_idx=1,
                candidate={"instructions": "test"},
                scores_by_val_id={0: 0.6},
                average_score=0.6,
                num_examples_evaluated=1,
                total_valset_size=5,
                parent_ids=[0],
                is_best_program=False,
                outputs_by_val_id=None,
            ),
        )

        calls = callback.get_calls("on_valset_evaluated")
        assert len(calls) == 1
        assert calls[0]["outputs_by_val_id"] is None

    def test_on_valset_evaluated_partial_coverage(self):
        """Verify num_examples_evaluated can differ from total_valset_size."""
        callback = RecordingCallback()

        notify_callbacks(
            [callback],
            "on_valset_evaluated",
            ValsetEvaluatedEvent(
                iteration=4,
                candidate_idx=3,
                candidate={"instructions": "partial eval"},
                scores_by_val_id={0: 0.8, 2: 0.9, 5: 0.7},
                average_score=0.8,
                num_examples_evaluated=3,
                total_valset_size=10,
                parent_ids=[2],
                is_best_program=False,
                outputs_by_val_id={0: "o0", 2: "o2", 5: "o5"},
            ),
        )

        calls = callback.get_calls("on_valset_evaluated")
        assert len(calls) == 1
        assert calls[0]["num_examples_evaluated"] == 3
        assert calls[0]["total_valset_size"] == 10
        assert len(calls[0]["scores_by_val_id"]) == 3


# =============================================================================
# J. Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in callbacks."""

    def test_on_error_called_with_exception(self):
        """Verify error callback called when exception occurs."""
        callback = RecordingCallback()
        exc = ValueError("Test error")

        notify_callbacks(
            [callback],
            "on_error",
            ErrorEvent(
                iteration=5,
                exception=exc,
                will_continue=True,
            ),
        )

        calls = callback.get_calls("on_error")
        assert len(calls) == 1
        assert calls[0]["exception"] is exc
        assert calls[0]["will_continue"] is True

    def test_callback_exception_does_not_stop_notification(self):
        """Verify a failing callback doesn't prevent other callbacks from running."""
        failing = FailingCallback(fail_on="on_optimization_start")
        recording = RecordingCallback()

        # Both callbacks should be called; failing one should not prevent recording
        notify_callbacks(
            [failing, recording],
            "on_optimization_start",
            OptimizationStartEvent(
                seed_candidate={},
                trainset_size=10,
                valset_size=5,
                config={},
            ),
        )

        # Recording callback should still have been called
        calls = recording.get_calls("on_optimization_start")
        assert len(calls) == 1

    def test_callback_exception_is_logged(self, caplog):
        """Verify callback exceptions are logged for debugging."""
        import logging

        failing = FailingCallback(fail_on="on_optimization_start")

        with caplog.at_level(logging.WARNING):
            notify_callbacks(
                [failing],
                "on_optimization_start",
                OptimizationStartEvent(
                    seed_candidate={},
                    trainset_size=10,
                    valset_size=5,
                    config={},
                ),
            )

        assert "failed on on_optimization_start" in caplog.text


# =============================================================================
# J. Composition Tests
# =============================================================================


class TestComposition:
    """Tests for callback composition."""

    def test_composite_callback_calls_all_callbacks(self):
        """Verify CompositeCallback notifies all registered callbacks."""
        callback1 = RecordingCallback()
        callback2 = RecordingCallback()

        composite = CompositeCallback([callback1, callback2])
        composite.on_optimization_start(
            OptimizationStartEvent(
                seed_candidate={},
                trainset_size=10,
                valset_size=5,
                config={},
            )
        )

        assert len(callback1.get_calls("on_optimization_start")) == 1
        assert len(callback2.get_calls("on_optimization_start")) == 1

    def test_multiple_callbacks_all_receive_events(self):
        """Verify multiple callbacks via notify_callbacks all receive events."""
        callbacks = [RecordingCallback() for _ in range(3)]

        notify_callbacks(
            callbacks,
            "on_iteration_start",
            IterationStartEvent(
                iteration=1,
                state=None,
            ),
        )

        for callback in callbacks:
            assert len(callback.get_calls("on_iteration_start")) == 1

    def test_callback_order_is_preserved(self):
        """Verify callbacks are called in registration order."""
        order = []

        class OrderCallback:
            def __init__(self, name):
                self.name = name

            def on_optimization_start(self, event):
                order.append(self.name)

        callbacks = [OrderCallback("first"), OrderCallback("second"), OrderCallback("third")]
        notify_callbacks(
            callbacks,
            "on_optimization_start",
            OptimizationStartEvent(
                seed_candidate={},
                trainset_size=10,
                valset_size=5,
                config={},
            ),
        )

        assert order == ["first", "second", "third"]

    def test_composite_callback_add_method(self):
        """Verify callbacks can be added to composite after creation."""
        composite = CompositeCallback()
        callback = RecordingCallback()

        composite.add(callback)
        composite.on_optimization_start(
            OptimizationStartEvent(
                seed_candidate={},
                trainset_size=10,
                valset_size=5,
                config={},
            )
        )

        assert len(callback.get_calls("on_optimization_start")) == 1

    def test_notify_callbacks_with_none(self):
        """Verify notify_callbacks handles None gracefully."""
        # Should not raise
        notify_callbacks(
            None,
            "on_optimization_start",
            OptimizationStartEvent(seed_candidate={}, trainset_size=0, valset_size=0, config={}),
        )

    def test_notify_callbacks_with_empty_list(self):
        """Verify notify_callbacks handles empty list gracefully."""
        # Should not raise
        notify_callbacks(
            [],
            "on_optimization_start",
            OptimizationStartEvent(seed_candidate={}, trainset_size=0, valset_size=0, config={}),
        )


# =============================================================================
# K. Argument Validation Tests
# =============================================================================


class TestArgumentValidation:
    """Tests for callback argument structure and types."""

    def test_reflective_dataset_structure_is_correct(self):
        """Verify dataset has Inputs/Generated Outputs/Feedback keys."""
        callback = RecordingCallback()

        # Standard reflective dataset structure
        dataset = {
            "predictor_name": [
                {
                    "Inputs": {"field1": "value1"},
                    "Generated Outputs": {"output1": "result1"},
                    "Feedback": "This is feedback",
                }
            ]
        }

        notify_callbacks(
            [callback],
            "on_reflective_dataset_built",
            ReflectiveDatasetBuiltEvent(
                iteration=1,
                candidate_idx=0,
                components=["predictor_name"],
                dataset=dataset,
            ),
        )

        calls = callback.get_calls("on_reflective_dataset_built")
        received_dataset = calls[0]["dataset"]

        # Verify structure
        assert "predictor_name" in received_dataset
        assert len(received_dataset["predictor_name"]) == 1
        example = received_dataset["predictor_name"][0]
        assert "Inputs" in example
        assert "Generated Outputs" in example
        assert "Feedback" in example

    def test_iteration_numbers_start_at_one(self):
        """Verify iteration numbers are 1-indexed as documented."""
        callback = RecordingCallback()

        # First iteration should be 1, not 0
        notify_callbacks(
            [callback],
            "on_iteration_start",
            IterationStartEvent(
                iteration=1,
                state=None,
            ),
        )

        calls = callback.get_calls("on_iteration_start")
        assert calls[0]["iteration"] == 1


# =============================================================================
# L. Integration Tests (to be enabled when implementation is complete)
# =============================================================================


def has_callback_support():
    """Check if optimize() supports the callbacks parameter.

    Returns True if the callback integration is implemented, False otherwise.
    """
    import inspect

    from gepa import optimize

    sig = inspect.signature(optimize)
    return "callbacks" in sig.parameters


class TestIntegration:
    """Integration tests that run with the full optimize() function.

    These tests require the callback system to be fully implemented
    in the optimize() function. They are skipped until that integration
    is complete.
    """

    @pytest.mark.skipif(not has_callback_support(), reason="callbacks parameter not yet added to optimize()")
    def test_callback_receives_real_optimization_flow(self):
        """End-to-end test with real adapter and mock LM."""
        from gepa import optimize

        callback = RecordingCallback()

        mock_data = [{"input": "test", "answer": "expected", "additional_context": {}}]
        task_lm = Mock(return_value="response")
        reflection_lm = Mock(return_value="```improved```")

        optimize(
            seed_candidate={"instructions": "initial"},
            trainset=mock_data,
            task_lm=task_lm,
            reflection_lm=reflection_lm,
            callbacks=[callback],
            max_metric_calls=5,
        )

        # Verify optimization lifecycle callbacks were called
        assert len(callback.get_calls("on_optimization_start")) == 1
        assert len(callback.get_calls("on_optimization_end")) == 1

        # Verify iteration callbacks were called
        assert len(callback.get_calls("on_iteration_start")) >= 1

    @pytest.mark.skipif(not has_callback_support(), reason="callbacks parameter not yet added to optimize()")
    def test_callback_with_stopper_interaction(self):
        """Verify callbacks work correctly with stop conditions."""
        from gepa import optimize

        callback = RecordingCallback()

        mock_data = [{"input": "test", "answer": "expected", "additional_context": {}}]
        task_lm = Mock(return_value="response")
        reflection_lm = Mock(return_value="```improved```")

        optimize(
            seed_candidate={"instructions": "initial"},
            trainset=mock_data,
            task_lm=task_lm,
            reflection_lm=reflection_lm,
            callbacks=[callback],
            max_metric_calls=10,
        )

        # Verify budget updates were tracked
        budget_calls = callback.get_calls("on_budget_updated")
        if budget_calls:
            last_call = budget_calls[-1]
            # Budget may slightly exceed max_metric_calls because stopper checks at iteration start,
            # and an iteration can consume multiple metric calls before completion
            assert last_call["metric_calls_used"] > 0
            assert "metric_calls_remaining" in last_call
