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
from concurrent.futures import ThreadPoolExecutor
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


class TestLMCallWithCost:
    """Verify ``LM.call_with_cost`` returns (text, cost) and degrades gracefully."""

    @patch("litellm.completion_cost")
    @patch("litellm.completion")
    def test_returns_text_and_cost(self, mock_completion, mock_cost):
        mock_completion.return_value = _fake_completion_response("```\nresult\n```")
        mock_cost.return_value = 0.007
        lm = LM("openai/gpt-4.1-mini")
        text, cost = lm.call_with_cost("hello")
        assert text == "```\nresult\n```"
        assert cost == pytest.approx(0.007)

    @patch("litellm.completion_cost")
    @patch("litellm.completion")
    def test_call_still_returns_plain_str(self, mock_completion, mock_cost):
        """Backward compat: ``lm("prompt")`` still returns a bare string."""
        mock_completion.return_value = _fake_completion_response("plain text")
        mock_cost.return_value = 0.01
        lm = LM("openai/gpt-4.1-mini")
        result = lm("hello")
        assert result == "plain text"
        assert isinstance(result, str)

    @patch("litellm.completion_cost")
    @patch("litellm.completion")
    def test_parallel_calls_each_return_own_cost(self, mock_completion, mock_cost):
        """Each call_with_cost invocation is self-contained.

        With a return-value-driven contract, there's no shared mutable state
        between workers, so each call's cost is the cost of *that* call. We
        exercise the concurrency path to witness no cross-worker leakage.
        """
        mock_completion.return_value = _fake_completion_response()
        # Alternate cost per call so we can spot any cross-attribution.
        cost_sequence = iter([0.001, 0.002, 0.003, 0.004, 0.005] * 4)
        cost_lock = threading.Lock()

        def _next_cost(*_args, **_kwargs):
            with cost_lock:
                return next(cost_sequence)

        mock_cost.side_effect = _next_cost

        lm = LM("openai/gpt-4.1-mini")
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(lm.call_with_cost, f"prompt-{i}") for i in range(20)]
            results = [f.result() for f in futures]

        costs = [cost for _text, cost in results]
        # All 20 call sites got a cost; every cost is one of our expected values.
        assert len(costs) == 20
        assert all(c in (0.001, 0.002, 0.003, 0.004, 0.005) for c in costs)
        # Grand-total conservation.
        assert sum(costs) == pytest.approx(4 * (0.001 + 0.002 + 0.003 + 0.004 + 0.005))

    @patch("litellm.completion_cost")
    @patch("litellm.completion")
    def test_completion_cost_exception_handled(self, mock_completion, mock_cost):
        """Models where litellm.completion_cost raises still complete calls with 0.0."""
        mock_completion.return_value = _fake_completion_response("ok")
        mock_cost.side_effect = Exception("unknown model pricing")
        lm = LM("unknown/my-self-hosted-model")
        text, cost = lm.call_with_cost("hello")
        assert text == "ok"
        assert cost == 0.0

    @patch("litellm.completion_cost")
    @patch("litellm.completion")
    def test_completion_cost_none_handled(self, mock_completion, mock_cost):
        """litellm sometimes returns None for cost — treated as 0.0."""
        mock_completion.return_value = _fake_completion_response("ok")
        mock_cost.return_value = None
        lm = LM("openai/gpt-4.1-mini")
        text, cost = lm.call_with_cost("hello")
        assert text == "ok"
        assert cost == 0.0

    def test_plain_callable_bypasses_call_with_cost(self):
        """A plain-callable reflection_lm (no ``call_with_cost``) still works.

        ``run_with_metadata`` falls back to ``lm(prompt)`` with cost 0.0 when
        the callable doesn't expose the method. This preserves the public
        ``LanguageModel`` protocol for users who pass a bare function.
        """
        from gepa.strategies.instruction_proposal import InstructionProposalSignature

        calls: list[str] = []

        def plain_lm(prompt):
            calls.append(prompt)
            return "```\nnew_instruction: better text\n```"

        assert not hasattr(plain_lm, "call_with_cost")

        result, _full_prompt, _raw, cost = InstructionProposalSignature.run_with_metadata(
            lm=plain_lm,
            input_dict={
                "current_instruction_doc": "old",
                "dataset_with_feedback": [{"input": "x", "output": "y"}],
                "prompt_template": None,
            },
        )
        assert len(calls) == 1
        assert cost == 0.0  # no cost data available from plain callable
        assert result is not None


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
