# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for reflection LM cost tracking persisted in GEPAState.

Three scopes, four tests:

- Unit: ``LM.call_with_cost`` returns ``(text, cost_usd)`` and falls back
  to ``0.0`` gracefully when litellm can't price the model.
- Protocol: ``run_with_metadata`` works with plain-callable reflection LMs
  that don't expose ``call_with_cost`` — the session-friendly duck-typed
  fallback path.
- End-to-end: ``optimize_anything`` runs with a mocked litellm backend,
  both sequential (``num_parallel_proposals=1``) and parallel
  (``num_parallel_proposals=2``), verifying per-proposal cost attribution
  lands correctly in ``GEPAState`` parallel-list fields.
"""

from __future__ import annotations

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
# Unit: LM.call_with_cost
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cost_value,cost_side_effect,expected_cost",
    [
        pytest.param(0.007, None, 0.007, id="normal"),
        pytest.param(None, None, 0.0, id="litellm_returns_none"),
        pytest.param(None, Exception("unknown model"), 0.0, id="litellm_raises"),
    ],
)
@patch("litellm.completion_cost")
@patch("litellm.completion")
def test_call_with_cost_returns_tuple(mock_completion, mock_cost, cost_value, cost_side_effect, expected_cost):
    """``LM.call_with_cost`` returns ``(text, cost)``. Unknown/unpriced models
    (where litellm raises or returns None) record ``0.0`` and the call
    still succeeds."""
    mock_completion.return_value = _fake_completion_response("result text")
    if cost_side_effect is not None:
        mock_cost.side_effect = cost_side_effect
    else:
        mock_cost.return_value = cost_value

    lm = LM("openai/gpt-4.1-mini")
    text, cost = lm.call_with_cost("hello")

    assert text == "result text"
    assert cost == pytest.approx(expected_cost)


# ---------------------------------------------------------------------------
# Protocol: duck-typed fallback for plain callables
# ---------------------------------------------------------------------------


def test_plain_callable_bypasses_call_with_cost():
    """A plain-callable reflection_lm (no ``call_with_cost``) still works.

    ``run_with_metadata`` falls back to ``lm(prompt)`` with cost ``0.0`` when
    the callable doesn't expose the method. This is the session-friendly
    path: when ``make_session_lm`` eventually attaches its own
    ``call_with_cost`` closure, the exact same ``getattr`` dispatch picks
    it up, no proposer changes needed.
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
# End-to-end: optimize_anything with LM + mocked litellm
# ---------------------------------------------------------------------------


def _assert_cost_invariants(state, cost_per_call: float) -> None:
    """Shared invariants for E2E runs: parallel-list lengths, per-entry
    equality to ``cost_per_call``, and total-cost conservation."""
    by_candidate: list[float] = list(state.reflection_cost_by_candidate)
    rejected: list[float] = list(state.reflection_cost_rejected)
    rejected_calls: list[int] = list(state.num_metric_calls_at_rejection)

    assert len(by_candidate) == len(state.program_candidates)
    assert len(rejected) == len(rejected_calls)
    assert by_candidate[0] == 0.0  # seed has no reflection cost

    total_proposals = (len(by_candidate) - 1) + len(rejected)
    assert total_proposals >= 1, "expected at least one reflection proposal to have run"

    # Single-component seed -> exactly one reflection call per proposal.
    for cost in by_candidate[1:]:
        assert cost == pytest.approx(cost_per_call)
    for cost in rejected:
        assert cost == pytest.approx(cost_per_call)

    expected_total = sum(by_candidate) + sum(rejected)
    assert state.total_reflection_cost == pytest.approx(expected_total)
    assert state.total_reflection_cost > 0.0


@pytest.mark.parametrize(
    "num_parallel_proposals,cost_per_call",
    [
        pytest.param(1, 0.003, id="sequential"),
        pytest.param(2, 0.004, id="parallel"),
    ],
)
@patch("litellm.completion_cost")
@patch("litellm.completion")
def test_cost_populates_state_e2e(mock_completion, mock_cost, num_parallel_proposals, cost_per_call):
    """Full ``optimize_anything`` pipeline in both sequential and parallel
    modes. Verifies the chain ``LM.call_with_cost`` → ``run_with_metadata``
    → ``propose_new_texts`` → ``CandidateProposal`` → engine accept/reject
    → ``GEPAState`` parallel lists."""
    mock_completion.return_value = _fake_completion_response("```\nbetter candidate\n```")
    mock_cost.return_value = cost_per_call

    lm = LM("openai/gpt-4.1-mini")
    capture = _StateCapture()

    result = optimize_anything(
        seed_candidate="seed text",
        evaluator=_good_evaluator,
        config=GEPAConfig(
            callbacks=[capture],  # type: ignore[list-item]
            engine=EngineConfig(
                max_metric_calls=15 if num_parallel_proposals == 1 else 20,
                num_parallel_proposals=num_parallel_proposals,
            ),
            reflection=ReflectionConfig(reflection_lm=lm),
        ),
    )

    assert capture.final_state is not None
    state = capture.final_state

    _assert_cost_invariants(state, cost_per_call)

    # Grand-total conservation: total cost equals cost_per_call * total LM calls.
    # Under parallel execution this is the strong witness — if any worker's
    # cost got mis-attributed to another, either the per-entry equality above
    # or this sum would fail.
    assert state.total_reflection_cost == pytest.approx(cost_per_call * mock_cost.call_count)

    # GEPAResult exposes the total cost as a scalar.
    assert result is not None
    assert result.total_reflection_cost == pytest.approx(state.total_reflection_cost)
