# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Integration tests for callbacks during a real optimization run.

These tests verify that callbacks are properly invoked during the GEPA
optimization process with real data and LLM calls (mocked for determinism).

Note: This test reuses the LLM cache from test_aime_prompt_optimization to
avoid duplicating the cache data.
"""

from pathlib import Path

import pytest

# Reuse the AIME test's LLM cache directory
RECORDER_DIR = Path(__file__).parent.parent / "test_aime_prompt_optimization"


class RecordingCallback:
    """A callback that records all method calls for integration testing."""

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


@pytest.fixture(scope="module")
def recorder_dir() -> Path:
    """Provides the path to the recording directory and ensures it exists."""
    RECORDER_DIR.mkdir(parents=True, exist_ok=True)
    return RECORDER_DIR


def test_callbacks_during_optimization(mocked_lms, recorder_dir):
    """
    Tests that callbacks are properly invoked during GEPA optimization.

    This test verifies:
    1. on_optimization_start is called once at the beginning
    2. on_optimization_end is called once at the end
    3. on_iteration_start/end are called for each iteration
    4. Evaluation callbacks are called during candidate evaluation
    5. Budget callbacks are called when metric calls are made
    6. Pareto front update callbacks are called when new candidates are accepted
    """
    import gepa
    from gepa.adapters.default_adapter.default_adapter import DefaultAdapter

    # 1. Setup: Unpack fixtures and load data
    task_lm, reflection_lm = mocked_lms
    adapter = DefaultAdapter(model=task_lm)

    print("Initializing AIME dataset...")
    trainset, valset, _ = gepa.examples.aime.init_dataset()
    trainset = trainset[:10]
    valset = valset[:10]

    seed_prompt = {
        "system_prompt": "You are a helpful assistant. You are given a question and you need to answer it. The answer should be given at the end of your response in exactly the format '### <final answer>'"
    }

    # Create callback to record all events
    callback = RecordingCallback()

    # 2. Execution: Run the core optimization logic with callbacks
    print("Running GEPA optimization process with callbacks...")
    gepa_result = gepa.optimize(
        seed_candidate=seed_prompt,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        max_metric_calls=30,
        reflection_lm=reflection_lm,
        display_progress_bar=True,
        callbacks=[callback],
    )

    # 3. Assertions: Verify callbacks were invoked correctly

    # Optimization lifecycle callbacks
    opt_start_calls = callback.get_calls("on_optimization_start")
    assert len(opt_start_calls) == 1, "on_optimization_start should be called exactly once"
    assert opt_start_calls[0]["trainset_size"] == 10
    assert opt_start_calls[0]["valset_size"] == 10
    assert opt_start_calls[0]["seed_candidate"] == seed_prompt

    opt_end_calls = callback.get_calls("on_optimization_end")
    assert len(opt_end_calls) == 1, "on_optimization_end should be called exactly once"
    assert "best_candidate_idx" in opt_end_calls[0]
    assert "total_iterations" in opt_end_calls[0]
    assert "total_metric_calls" in opt_end_calls[0]
    assert opt_end_calls[0]["total_metric_calls"] > 0

    # Iteration callbacks
    iter_start_calls = callback.get_calls("on_iteration_start")
    iter_end_calls = callback.get_calls("on_iteration_end")
    assert len(iter_start_calls) >= 1, "at least one iteration should have started"
    assert len(iter_start_calls) == len(iter_end_calls), "each iteration_start should have a matching iteration_end"

    # Verify iterations are numbered correctly (1-indexed)
    for i, call in enumerate(iter_start_calls):
        assert call["iteration"] == i + 1, f"iteration should be 1-indexed, got {call['iteration']} for index {i}"

    # Candidate selection callbacks
    candidate_selected_calls = callback.get_calls("on_candidate_selected")
    assert len(candidate_selected_calls) >= 1, "at least one candidate should be selected"

    # Minibatch sampling callbacks
    minibatch_calls = callback.get_calls("on_minibatch_sampled")
    assert len(minibatch_calls) >= 1, "at least one minibatch should be sampled"
    for call in minibatch_calls:
        assert call["trainset_size"] == 10

    # Evaluation callbacks
    eval_start_calls = callback.get_calls("on_evaluation_start")
    eval_end_calls = callback.get_calls("on_evaluation_end")
    assert len(eval_start_calls) >= 1, "at least one evaluation should start"
    assert len(eval_end_calls) >= 1, "at least one evaluation should end"

    # Budget callbacks
    budget_calls = callback.get_calls("on_budget_updated")
    assert len(budget_calls) >= 1, "budget should be updated at least once"
    # Budget should increase monotonically
    prev_used = 0
    for call in budget_calls:
        assert call["metric_calls_used"] >= prev_used, "budget should increase monotonically"
        prev_used = call["metric_calls_used"]

    # Pareto front callbacks
    pareto_calls = callback.get_calls("on_pareto_front_updated")
    assert len(pareto_calls) >= 1, "pareto front should be updated at least once"
    for call in pareto_calls:
        assert "new_front" in call
        assert "displaced_candidates" in call

    # Valset evaluated callbacks
    valset_calls = callback.get_calls("on_valset_evaluated")
    assert len(valset_calls) >= 1, "at least one valset evaluation should occur"

    # Verify seed candidate valset evaluation is included (iteration 0)
    seed_valset_calls = [c for c in valset_calls if c["iteration"] == 0]
    assert len(seed_valset_calls) == 1, "seed candidate valset evaluation should be called at iteration 0"
    seed_call = seed_valset_calls[0]
    assert seed_call["candidate_idx"] == 0, "seed candidate should have index 0"
    assert seed_call["parent_ids"] == [], "seed candidate should have no parents"
    assert seed_call["is_best_program"] is True, "seed candidate should be best at iteration 0"

    for call in valset_calls:
        assert "candidate_idx" in call
        assert "average_score" in call
        assert "is_best_program" in call

    # Proposal callbacks - core to the optimization loop
    proposal_start_calls = callback.get_calls("on_proposal_start")
    proposal_end_calls = callback.get_calls("on_proposal_end")
    assert len(proposal_start_calls) >= 1, "at least one proposal should start"
    assert len(proposal_start_calls) == len(proposal_end_calls), (
        "each proposal_start should have a matching proposal_end"
    )
    for call in proposal_start_calls:
        assert "parent_candidate" in call
        assert "iteration" in call
        assert "components" in call
    for call in proposal_end_calls:
        assert "new_instructions" in call
        assert "iteration" in call

    # Candidate acceptance/rejection callbacks - verify optimization decisions are tracked
    accepted_calls = callback.get_calls("on_candidate_accepted")
    rejected_calls = callback.get_calls("on_candidate_rejected")
    # At least some candidates should be processed (accepted or rejected)
    total_decisions = len(accepted_calls) + len(rejected_calls)
    assert total_decisions >= 1, "at least one candidate should be accepted or rejected"
    # Verify accepted callbacks have required fields
    for call in accepted_calls:
        assert "new_candidate_idx" in call
        assert "new_score" in call
        assert "parent_ids" in call
    # Verify rejected callbacks have required fields
    for call in rejected_calls:
        assert "new_score" in call
        assert "old_score" in call
        assert "reason" in call

    # State saved callbacks
    # Note: state_saved may not be called if no run_dir is provided
    # This is expected behavior

    # Verify the result is still correct
    best_prompt = gepa_result.best_candidate["system_prompt"]
    assert isinstance(best_prompt, str) and len(best_prompt) > 0

    print(f"Total callback events recorded: {len(callback.calls)}")
    print(f"Iterations completed: {len(iter_start_calls)}")
    print(f"Budget updates: {len(budget_calls)}")
    print(f"Final metric calls: {opt_end_calls[0]['total_metric_calls']}")


def test_multiple_callbacks_all_receive_events(mocked_lms, recorder_dir):
    """
    Tests that multiple callbacks all receive the same events.
    """
    import gepa
    from gepa.adapters.default_adapter.default_adapter import DefaultAdapter

    task_lm, reflection_lm = mocked_lms
    adapter = DefaultAdapter(model=task_lm)

    trainset, valset, _ = gepa.examples.aime.init_dataset()
    trainset = trainset[:10]
    valset = valset[:10]

    # Must use the same seed prompt as the cached test to hit the LLM cache
    seed_prompt = {
        "system_prompt": "You are a helpful assistant. You are given a question and you need to answer it. The answer should be given at the end of your response in exactly the format '### <final answer>'"
    }

    # Create two separate callbacks
    callback1 = RecordingCallback()
    callback2 = RecordingCallback()

    gepa.optimize(
        seed_candidate=seed_prompt,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        max_metric_calls=30,
        reflection_lm=reflection_lm,
        callbacks=[callback1, callback2],
    )

    # Both callbacks should have received the same events
    assert len(callback1.get_calls("on_optimization_start")) == 1
    assert len(callback2.get_calls("on_optimization_start")) == 1

    assert len(callback1.get_calls("on_optimization_end")) == 1
    assert len(callback2.get_calls("on_optimization_end")) == 1

    # Iteration counts should match
    assert len(callback1.get_calls("on_iteration_start")) == len(callback2.get_calls("on_iteration_start"))
    assert len(callback1.get_calls("on_iteration_end")) == len(callback2.get_calls("on_iteration_end"))

    print("Both callbacks received identical event counts")


def test_partial_callback_implementation(mocked_lms, recorder_dir):
    """
    Tests that callbacks with only some methods implemented work correctly.
    """
    import gepa
    from gepa.adapters.default_adapter.default_adapter import DefaultAdapter

    task_lm, reflection_lm = mocked_lms
    adapter = DefaultAdapter(model=task_lm)

    trainset, valset, _ = gepa.examples.aime.init_dataset()
    trainset = trainset[:10]
    valset = valset[:10]

    # Must use the same seed prompt as the cached test to hit the LLM cache
    seed_prompt = {
        "system_prompt": "You are a helpful assistant. You are given a question and you need to answer it. The answer should be given at the end of your response in exactly the format '### <final answer>'"
    }

    # Create a callback that only implements some methods
    class PartialCallback:
        def __init__(self):
            self.iterations = []
            self.budget_updates = []

        def on_iteration_start(self, event):
            self.iterations.append(event["iteration"])

        def on_budget_updated(self, event):
            self.budget_updates.append(event["metric_calls_used"])

    callback = PartialCallback()

    # This should not raise any errors
    gepa.optimize(
        seed_candidate=seed_prompt,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        max_metric_calls=30,
        reflection_lm=reflection_lm,
        callbacks=[callback],
    )

    assert len(callback.iterations) >= 1, "partial callback should receive iteration events"
    assert len(callback.budget_updates) >= 1, "partial callback should receive budget events"

    print(f"Partial callback received {len(callback.iterations)} iteration events")
    print(f"Partial callback received {len(callback.budget_updates)} budget updates")
