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

from gepa.core.callbacks import CompositeCallback, GEPACallback, notify_callbacks

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

    def _record(self, method_name, **kwargs):
        self.calls.append((method_name, kwargs))

    def get_calls(self, method_name):
        """Get all calls to a specific method."""
        return [kwargs for name, kwargs in self.calls if name == method_name]

    def on_optimization_start(self, **kwargs):
        self._record("on_optimization_start", **kwargs)

    def on_optimization_end(self, **kwargs):
        self._record("on_optimization_end", **kwargs)

    def on_iteration_start(self, **kwargs):
        self._record("on_iteration_start", **kwargs)

    def on_iteration_end(self, **kwargs):
        self._record("on_iteration_end", **kwargs)

    def on_candidate_selected(self, **kwargs):
        self._record("on_candidate_selected", **kwargs)

    def on_minibatch_sampled(self, **kwargs):
        self._record("on_minibatch_sampled", **kwargs)

    def on_evaluation_start(self, **kwargs):
        self._record("on_evaluation_start", **kwargs)

    def on_evaluation_end(self, **kwargs):
        self._record("on_evaluation_end", **kwargs)

    def on_reflective_dataset_built(self, **kwargs):
        self._record("on_reflective_dataset_built", **kwargs)

    def on_proposal_start(self, **kwargs):
        self._record("on_proposal_start", **kwargs)

    def on_proposal_end(self, **kwargs):
        self._record("on_proposal_end", **kwargs)

    def on_candidate_accepted(self, **kwargs):
        self._record("on_candidate_accepted", **kwargs)

    def on_candidate_rejected(self, **kwargs):
        self._record("on_candidate_rejected", **kwargs)

    def on_merge_attempted(self, **kwargs):
        self._record("on_merge_attempted", **kwargs)

    def on_merge_accepted(self, **kwargs):
        self._record("on_merge_accepted", **kwargs)

    def on_merge_rejected(self, **kwargs):
        self._record("on_merge_rejected", **kwargs)

    def on_pareto_front_updated(self, **kwargs):
        self._record("on_pareto_front_updated", **kwargs)

    def on_state_saved(self, **kwargs):
        self._record("on_state_saved", **kwargs)

    def on_budget_updated(self, **kwargs):
        self._record("on_budget_updated", **kwargs)

    def on_error(self, **kwargs):
        self._record("on_error", **kwargs)


class FailingCallback:
    """A callback that raises exceptions for testing error handling.

    Used to verify that callback failures are caught and logged without
    crashing the optimization.
    """

    def __init__(self, fail_on=None):
        self.fail_on = fail_on

    def on_optimization_start(self, **kwargs):
        if self.fail_on == "on_optimization_start":
            raise ValueError("Intentional failure")

    def on_iteration_start(self, **kwargs):
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
        assert hasattr(GEPACallback, "__protocol_attrs__") or hasattr(
            GEPACallback, "_is_runtime_protocol"
        )

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
        notify_callbacks([callback], "on_optimization_start", seed_candidate={})

    def test_partial_callback_implementation(self):
        """Verify callbacks with only some methods work."""

        class PartialCallback:
            def __init__(self):
                self.called = False

            def on_optimization_start(self, **kwargs):
                self.called = True

        callback = PartialCallback()
        notify_callbacks(
            [callback],
            "on_optimization_start",
            seed_candidate={},
            trainset_size=10,
            valset_size=5,
            config={},
        )
        assert callback.called

        # Calling a method that doesn't exist should not raise
        notify_callbacks([callback], "on_iteration_start", iteration=1, state=None)


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
            seed_candidate={"instructions": "test"},
            trainset_size=100,
            valset_size=20,
            config={"max_metric_calls": 500},
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
            best_candidate_idx=3,
            total_iterations=50,
            total_metric_calls=450,
            final_state=mock_state,
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
            iteration=5,
            state=mock_state,
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
            iteration=5,
            state=mock_state,
            proposal_accepted=True,
        )

        # Test rejected case
        notify_callbacks(
            [callback],
            "on_iteration_end",
            iteration=6,
            state=mock_state,
            proposal_accepted=False,
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
            iteration=3,
            candidate_idx=2,
            candidate={"instructions": "selected instructions"},
            score=0.85,
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
            iteration=3,
            minibatch_ids=[0, 5, 12, 23, 45],
            trainset_size=100,
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
            iteration=5,
            new_candidate_idx=3,
            new_score=0.92,
            parent_idx=1,
        )

        calls = callback.get_calls("on_candidate_accepted")
        assert len(calls) == 1
        assert calls[0]["new_candidate_idx"] == 3
        assert calls[0]["new_score"] == 0.92
        assert calls[0]["parent_idx"] == 1

    def test_on_candidate_rejected_called_on_no_improvement(self):
        """Verify rejection callback called with scores and reason."""
        callback = RecordingCallback()

        notify_callbacks(
            [callback],
            "on_candidate_rejected",
            iteration=5,
            old_score=0.85,
            new_score=0.80,
            reason="New subsample score not better than old",
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
        """Verify on_evaluation_start receives batch size and trace flag."""
        callback = RecordingCallback()

        notify_callbacks(
            [callback],
            "on_evaluation_start",
            iteration=3,
            candidate_idx=1,
            batch_size=35,
            capture_traces=True,
        )

        calls = callback.get_calls("on_evaluation_start")
        assert len(calls) == 1
        assert calls[0]["batch_size"] == 35
        assert calls[0]["capture_traces"] is True

    def test_on_evaluation_end_scores_are_list_of_floats(self):
        """Verify scores argument is correctly typed."""
        callback = RecordingCallback()

        notify_callbacks(
            [callback],
            "on_evaluation_end",
            iteration=3,
            candidate_idx=1,
            scores=[0.8, 0.9, 1.0, 0.7, 0.85],
            has_trajectories=True,
        )

        calls = callback.get_calls("on_evaluation_end")
        assert len(calls) == 1
        assert isinstance(calls[0]["scores"], list)
        assert all(isinstance(s, float) for s in calls[0]["scores"])
        assert calls[0]["has_trajectories"] is True


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
            iteration=3,
            candidate_idx=1,
            components=["predictor"],
            dataset=dataset,
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
            iteration=3,
            parent_candidate={"instructions": "Original instructions"},
            components=["instructions"],
            reflective_dataset={"instructions": []},
        )

        # Proposal end
        notify_callbacks(
            [callback],
            "on_proposal_end",
            iteration=3,
            new_instructions={"instructions": "Improved instructions"},
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
            iteration=10,
            parent_indices=[1, 3],
            merged_candidate={"instructions": "merged"},
        )

        calls = callback.get_calls("on_merge_attempted")
        assert len(calls) == 1
        assert calls[0]["parent_indices"] == [1, 3]

    def test_on_merge_accepted_called_on_improvement(self):
        """Verify merge acceptance callback includes new index."""
        callback = RecordingCallback()

        notify_callbacks(
            [callback],
            "on_merge_accepted",
            iteration=10,
            new_candidate_idx=5,
            parent_indices=[1, 3],
        )

        calls = callback.get_calls("on_merge_accepted")
        assert len(calls) == 1
        assert calls[0]["new_candidate_idx"] == 5
        assert calls[0]["parent_indices"] == [1, 3]

    def test_on_merge_rejected_called_on_failure(self):
        """Verify merge rejection includes reason."""
        callback = RecordingCallback()

        notify_callbacks(
            [callback],
            "on_merge_rejected",
            iteration=10,
            parent_indices=[1, 3],
            reason="Merged score worse than both parents",
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
            iteration=5,
            new_front=[0, 2, 4],
            displaced_candidates=[1],
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
            iteration=5,
            run_dir="/tmp/gepa_run_123",
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
            iteration=5,
            metric_calls_used=150,
            metric_calls_remaining=350,
        )

        calls = callback.get_calls("on_budget_updated")
        assert len(calls) == 1
        assert calls[0]["metric_calls_used"] == 150
        assert calls[0]["metric_calls_remaining"] == 350


# =============================================================================
# I. Error Handling Tests
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
            iteration=5,
            exception=exc,
            will_continue=True,
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
            seed_candidate={},
            trainset_size=10,
            valset_size=5,
            config={},
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
                seed_candidate={},
                trainset_size=10,
                valset_size=5,
                config={},
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
            seed_candidate={},
            trainset_size=10,
            valset_size=5,
            config={},
        )

        assert len(callback1.get_calls("on_optimization_start")) == 1
        assert len(callback2.get_calls("on_optimization_start")) == 1

    def test_multiple_callbacks_all_receive_events(self):
        """Verify multiple callbacks via notify_callbacks all receive events."""
        callbacks = [RecordingCallback() for _ in range(3)]

        notify_callbacks(
            callbacks,
            "on_iteration_start",
            iteration=1,
            state=None,
        )

        for callback in callbacks:
            assert len(callback.get_calls("on_iteration_start")) == 1

    def test_callback_order_is_preserved(self):
        """Verify callbacks are called in registration order."""
        order = []

        class OrderCallback:
            def __init__(self, name):
                self.name = name

            def on_optimization_start(self, **kwargs):
                order.append(self.name)

        callbacks = [OrderCallback("first"), OrderCallback("second"), OrderCallback("third")]
        notify_callbacks(
            callbacks,
            "on_optimization_start",
            seed_candidate={},
            trainset_size=10,
            valset_size=5,
            config={},
        )

        assert order == ["first", "second", "third"]

    def test_composite_callback_add_method(self):
        """Verify callbacks can be added to composite after creation."""
        composite = CompositeCallback()
        callback = RecordingCallback()

        composite.add(callback)
        composite.on_optimization_start(
            seed_candidate={},
            trainset_size=10,
            valset_size=5,
            config={},
        )

        assert len(callback.get_calls("on_optimization_start")) == 1

    def test_notify_callbacks_with_none(self):
        """Verify notify_callbacks handles None gracefully."""
        # Should not raise
        notify_callbacks(None, "on_optimization_start", seed_candidate={})

    def test_notify_callbacks_with_empty_list(self):
        """Verify notify_callbacks handles empty list gracefully."""
        # Should not raise
        notify_callbacks([], "on_optimization_start", seed_candidate={})


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
            iteration=1,
            candidate_idx=0,
            components=["predictor_name"],
            dataset=dataset,
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
            iteration=1,
            state=None,
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
            assert last_call["metric_calls_used"] <= 10
