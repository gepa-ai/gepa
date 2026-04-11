# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""End-to-end tests for reflection LM cost tracking persisted in GEPAState.

Exercises three layers:

1. ``LM``-level unit tests: per-thread cost accumulation, reset, and thread
   isolation under ``threading.local``.
2. Sequential end-to-end: ``optimize_anything`` run with an ``LM`` whose
   ``litellm`` calls are mocked, verifying state fields fill correctly.
3. Parallel end-to-end: same pipeline with ``num_parallel_proposals > 1``,
   verifying per-proposal cost attribution survives thread-pool execution.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from gepa.core.state import GEPAState
from gepa.lm import LM
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    optimize_anything,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_completion_response(content: str = "```\nimproved candidate\n```") -> MagicMock:
    """Build a ModelResponse-shaped mock that satisfies ``LM.__call__``."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.choices[0].finish_reason = "stop"
    return resp


def _good_evaluator(_candidate):
    """Evaluator for optimize_anything — returns a fixed score and empty side info."""
    return 0.5, {}


class _StateCapture:
    """GEPACallback that stashes the final GEPAState for inspection."""

    def __init__(self) -> None:
        self.final_state: GEPAState | None = None

    def on_optimization_end(self, event) -> None:
        self.final_state = event["final_state"]


# ---------------------------------------------------------------------------
# LM-level unit tests
# ---------------------------------------------------------------------------


class TestLMCostTracking:
    """Verify per-thread cost accumulation, reset, and isolation on ``LM``."""

    @patch("litellm.completion_cost")
    @patch("litellm.completion")
    def test_initial_cost_is_zero(self, mock_completion, mock_cost):
        mock_completion.return_value = _fake_completion_response()
        mock_cost.return_value = 0.01
        lm = LM("openai/gpt-4.1-mini")
        assert lm.get_call_cost() == 0.0

    @patch("litellm.completion_cost")
    @patch("litellm.completion")
    def test_single_call_accumulates(self, mock_completion, mock_cost):
        mock_completion.return_value = _fake_completion_response()
        mock_cost.return_value = 0.007
        lm = LM("openai/gpt-4.1-mini")
        lm("hello")
        assert lm.get_call_cost() == pytest.approx(0.007)

    @patch("litellm.completion_cost")
    @patch("litellm.completion")
    def test_multiple_calls_accumulate(self, mock_completion, mock_cost):
        mock_completion.return_value = _fake_completion_response()
        mock_cost.return_value = 0.01
        lm = LM("openai/gpt-4.1-mini")
        for _ in range(5):
            lm("prompt")
        assert lm.get_call_cost() == pytest.approx(0.05)

    @patch("litellm.completion_cost")
    @patch("litellm.completion")
    def test_reset_zeros_counter(self, mock_completion, mock_cost):
        mock_completion.return_value = _fake_completion_response()
        mock_cost.return_value = 0.01
        lm = LM("openai/gpt-4.1-mini")
        lm("a")
        lm("b")
        assert lm.get_call_cost() == pytest.approx(0.02)
        lm.reset_call_cost()
        assert lm.get_call_cost() == 0.0
        lm("c")
        assert lm.get_call_cost() == pytest.approx(0.01)

    @patch("litellm.completion_cost")
    @patch("litellm.completion")
    def test_thread_local_isolation(self, mock_completion, mock_cost):
        """Two threads sharing one LM must not see each other's cost."""
        mock_completion.return_value = _fake_completion_response()
        mock_cost.return_value = 0.005

        lm = LM("openai/gpt-4.1-mini")
        results: dict[str, float] = {}
        barrier = threading.Barrier(2)

        def worker(name: str, n_calls: int) -> None:
            lm.reset_call_cost()
            barrier.wait()  # force both threads to run concurrently
            for _ in range(n_calls):
                lm("prompt")
            results[name] = lm.get_call_cost()

        t1 = threading.Thread(target=worker, args=("A", 3))
        t2 = threading.Thread(target=worker, args=("B", 5))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results["A"] == pytest.approx(0.015)  # 3 * 0.005
        assert results["B"] == pytest.approx(0.025)  # 5 * 0.005

    @patch("litellm.completion_cost")
    @patch("litellm.completion")
    def test_completion_cost_exception_handled(self, mock_completion, mock_cost):
        """Models where litellm.completion_cost raises still complete calls with 0.0."""
        mock_completion.return_value = _fake_completion_response()
        mock_cost.side_effect = Exception("unknown model pricing")
        lm = LM("unknown/my-self-hosted-model")
        lm("hello")
        assert lm.get_call_cost() == 0.0

    @patch("litellm.completion_cost")
    @patch("litellm.completion")
    def test_completion_cost_none_handled(self, mock_completion, mock_cost):
        """litellm sometimes returns None for cost — treated as 0.0."""
        mock_completion.return_value = _fake_completion_response()
        mock_cost.return_value = None
        lm = LM("openai/gpt-4.1-mini")
        lm("hello")
        assert lm.get_call_cost() == 0.0


# ---------------------------------------------------------------------------
# End-to-end tests: optimize_anything with real LM + mocked litellm
# ---------------------------------------------------------------------------


def _assert_cost_invariants(state, cost_per_call: float) -> None:
    """Shared invariants for both sequential and parallel runs."""
    by_candidate: list[float] = list(state.reflection_cost_by_candidate)
    rejected: list[float] = list(state.reflection_cost_rejected)
    rejected_calls: list[int] = list(state.num_metric_calls_at_rejection)
    program_candidates = list(state.program_candidates)

    # Parallel-list invariants.
    assert len(by_candidate) == len(program_candidates)
    assert len(rejected) == len(rejected_calls)

    # Seed candidate has zero reflection cost.
    assert by_candidate[0] == 0.0

    # At least one proposal was attempted.
    total_proposals = (len(by_candidate) - 1) + len(rejected)
    assert total_proposals >= 1, "expected at least one reflection proposal to have run"

    # Each proposal (seedless) makes at least one reflection call, so its
    # recorded cost must be exactly an integer multiple of cost_per_call.
    # Single-component seed_candidate means exactly one call per proposal.
    for cost in by_candidate[1:]:
        assert cost == pytest.approx(cost_per_call), (
            f"accepted proposal cost {cost} != single-call cost {cost_per_call}"
        )
    for cost in rejected:
        assert cost == pytest.approx(cost_per_call), (
            f"rejected proposal cost {cost} != single-call cost {cost_per_call}"
        )

    # total_reflection_cost property matches the sum of its parts.
    expected_total = sum(by_candidate) + sum(rejected)
    assert state.total_reflection_cost == pytest.approx(expected_total)
    assert state.total_reflection_cost > 0.0


class TestE2EReflectionCostTracking:
    """Run a full optimize_anything pipeline and verify GEPAState cost fields."""

    @patch("litellm.completion_cost")
    @patch("litellm.completion")
    def test_sequential_cost_populates_state(self, mock_completion, mock_cost):
        """Sequential mode: every accepted + rejected proposal contributes a cost entry."""
        mock_completion.return_value = _fake_completion_response("```\nbetter candidate\n```")
        cost_per_call = 0.003
        mock_cost.return_value = cost_per_call

        lm = LM("openai/gpt-4.1-mini")
        capture = _StateCapture()

        _ = optimize_anything(
            seed_candidate="seed text",
            evaluator=_good_evaluator,
            config=GEPAConfig(
                callbacks=[capture],  # type: ignore[list-item]
                engine=EngineConfig(max_metric_calls=15, num_parallel_proposals=1),
                reflection=ReflectionConfig(reflection_lm=lm),
            ),
        )

        assert capture.final_state is not None
        state = capture.final_state

        _assert_cost_invariants(state, cost_per_call)

        # Grand-total conservation: with sequential mode, cost attribution has
        # no races, so the total equals cost_per_call * number of LM calls.
        assert state.total_reflection_cost == pytest.approx(cost_per_call * mock_cost.call_count)

    @patch("litellm.completion_cost")
    @patch("litellm.completion")
    def test_parallel_cost_populates_state(self, mock_completion, mock_cost):
        """Parallel mode: cost attribution must survive ThreadPoolExecutor dispatch.

        With num_parallel_proposals > 1, multiple workers share one LM instance
        but each worker's reflection cost must be attributed to its own proposal,
        not leaked across workers. thread-local gives us this isolation; this
        test is the end-to-end witness that the chain (LM → proposer → engine
        → state) holds under concurrent execution.
        """
        mock_completion.return_value = _fake_completion_response("```\nbetter candidate\n```")
        cost_per_call = 0.004
        mock_cost.return_value = cost_per_call

        lm = LM("openai/gpt-4.1-mini")
        capture = _StateCapture()

        _ = optimize_anything(
            seed_candidate="seed text",
            evaluator=_good_evaluator,
            config=GEPAConfig(
                callbacks=[capture],  # type: ignore[list-item]
                engine=EngineConfig(max_metric_calls=20, num_parallel_proposals=2),
                reflection=ReflectionConfig(reflection_lm=lm),
            ),
        )

        assert capture.final_state is not None
        state = capture.final_state

        _assert_cost_invariants(state, cost_per_call)

        # Conservation: sum of all per-proposal costs equals cost_per_call * total
        # LM calls. If one worker had overwritten another's thread-local slot,
        # some proposal would report 0.0 and this equality would fail (the
        # per-entry `>= cost_per_call` assertions in `_assert_cost_invariants`
        # catch that case too).
        assert state.total_reflection_cost == pytest.approx(cost_per_call * mock_cost.call_count)
