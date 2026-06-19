# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for the Proposer ABC and its template method pipeline."""

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from gepa.proposer.base import CandidateProposal, Proposer
from gepa.strategies.proposal_sampling import ProposalTask, SingleMutationSampling
from gepa.strategies.proposal_selection import AllImprovements, BestImprovement
from gepa.strategies.acceptance import StrictImprovementAcceptance


# ---------------------------------------------------------------------------
# Minimal concrete subclass for testing
# ---------------------------------------------------------------------------

class EchoProposer(Proposer):
    """Returns each task's parent_candidate as a proposal with configurable scores."""

    def __init__(self, scores_before, scores_after, **kwargs):
        self.scores_before = scores_before
        self.scores_after = scores_after
        super().__init__(**kwargs)

    def _mutate(self, tasks, state):
        return [
            CandidateProposal(
                candidate=task.parent_candidate,
                parent_program_ids=[task.parent_idx],
                subsample_scores_before=self.scores_before,
                subsample_scores_after=self.scores_after,
            )
            for task in tasks
        ]


def _make_state(num_candidates=1):
    state = MagicMock()
    state.program_candidates = [{"prompt": f"candidate_{i}"} for i in range(num_candidates)]
    state.program_full_scores_val_set = [0.5] * num_candidates
    state.parent_program_for_candidate = [[None]] * num_candidates
    state.full_program_trace = [{}]
    state.i = 0
    state.total_num_evals = 0
    return state


def _make_proposer(scores_before, scores_after, **kwargs):
    candidate_selector = MagicMock()
    candidate_selector.select_candidate_idx.return_value = 0
    batch_sampler = MagicMock()
    batch_sampler.next_minibatch_ids.return_value = [0, 1]
    trainset = MagicMock()
    trainset.__len__ = MagicMock(return_value=10)
    trainset.fetch.return_value = [{"x": 1}, {"x": 2}]
    return EchoProposer(
        scores_before=scores_before,
        scores_after=scores_after,
        trainset=trainset,
        candidate_selector=candidate_selector,
        batch_sampler=batch_sampler,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_proposer_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        Proposer(trainset=[], candidate_selector=MagicMock(), batch_sampler=MagicMock())


def test_propose_returns_list_when_proposal_improves():
    proposer = _make_proposer(scores_before=[0.4], scores_after=[0.6])
    state = _make_state()
    result = proposer.propose(state)
    assert len(result) == 1
    assert isinstance(result[0], CandidateProposal)


def test_propose_returns_empty_when_proposal_does_not_improve():
    proposer = _make_proposer(scores_before=[0.6], scores_after=[0.4])
    state = _make_state()
    result = proposer.propose(state)
    assert result == []


def test_propose_uses_default_strict_improvement_acceptance():
    # Equal score should be rejected by StrictImprovementAcceptance
    proposer = _make_proposer(scores_before=[0.5], scores_after=[0.5])
    state = _make_state()
    result = proposer.propose(state)
    assert result == []


def test_custom_selection_strategy_best_improvement():
    proposer = _make_proposer(
        scores_before=[0.4],
        scores_after=[0.9],
        selection_strategy=BestImprovement(),
    )
    state = _make_state()
    result = proposer.propose(state)
    assert len(result) == 1


def test_override_select_bypasses_strategy():
    class CustomSelectProposer(EchoProposer):
        def _select(self, proposals, state):
            # Always return all proposals regardless of scores
            return proposals

    proposer = CustomSelectProposer(
        scores_before=[0.9],
        scores_after=[0.1],  # would normally be rejected
        trainset=MagicMock(__len__=MagicMock(return_value=10), fetch=MagicMock(return_value=[])),
        candidate_selector=MagicMock(select_candidate_idx=MagicMock(return_value=0)),
        batch_sampler=MagicMock(next_minibatch_ids=MagicMock(return_value=[])),
    )
    state = _make_state()
    result = proposer.propose(state)
    assert len(result) == 1


def test_default_sampling_strategy_is_single_mutation():
    proposer = _make_proposer(scores_before=[0.3], scores_after=[0.7])
    assert isinstance(proposer.sampling_strategy, SingleMutationSampling)


def test_default_selection_strategy_is_all_improvements():
    proposer = _make_proposer(scores_before=[0.3], scores_after=[0.7])
    assert isinstance(proposer.selection_strategy, AllImprovements)


def test_default_acceptance_criterion_is_strict_improvement():
    proposer = _make_proposer(scores_before=[0.3], scores_after=[0.7])
    assert isinstance(proposer.acceptance_criterion, StrictImprovementAcceptance)
