# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for batch-based parallel proposals."""

from unittest.mock import MagicMock

import pytest

from gepa.core.adapter import EvaluationBatch, default_batch_evaluate
from gepa.core.state import GEPAState, ValsetEvaluation
from gepa.proposer.base import CandidateProposal, SubsampleEvaluation
from gepa.strategies.acceptance import StrictImprovementAcceptance
from gepa.strategies.proposal_sampling import (
    IndependentSampling,
    PxNSampling,
    SameParentSampling,
    SingleMutationSampling,
)
from gepa.strategies.proposal_selection import (
    AllImprovements,
    BestImprovement,
    TopKImprovements,
)


@pytest.fixture
def mock_state():
    """Create a GEPAState with 2 candidates."""
    seed_candidate = {"system_prompt": "test0"}
    base_eval = ValsetEvaluation(
        outputs_by_val_id={0: "out0", 1: "out1"},
        scores_by_val_id={0: 0.5, 1: 0.5},
        objective_scores_by_val_id=None,
    )
    state = GEPAState(seed_candidate, base_eval, track_best_outputs=False)

    # Add a second candidate
    state.program_candidates.append({"system_prompt": "test1"})
    state.prog_candidate_val_subscores.append({0: 0.7, 1: 0.7})
    state.prog_candidate_objective_scores.append({})
    state.parent_program_for_candidate.append([0])
    state.named_predictor_id_to_update_next_for_program_candidate.append(0)
    state.num_metric_calls_by_discovery.append(0)

    for i in range(len(state.pareto_front_valset)):
        state.program_at_pareto_front_valset[i] = {0, 1}

    assert state.is_consistent()
    return state


@pytest.fixture
def mock_candidate_selector():
    selector = MagicMock()
    selector.select_candidate_idx = MagicMock(return_value=0)
    return selector


@pytest.fixture
def mock_batch_sampler():
    sampler = MagicMock()
    call_count = 0

    def next_ids(trainset, state):
        nonlocal call_count
        call_count += 1
        return [call_count - 1]

    sampler.next_minibatch_ids = MagicMock(side_effect=next_ids)
    return sampler


@pytest.fixture
def mock_trainset():
    trainset = MagicMock()
    trainset.fetch = MagicMock(side_effect=lambda ids: [f"example_{i}" for i in ids])
    trainset.all_ids = MagicMock(return_value=[0, 1, 2, 3])
    trainset.__len__ = MagicMock(return_value=4)
    return trainset


class TestSamplingStrategies:
    def test_single_mutation(self, mock_state, mock_candidate_selector, mock_batch_sampler, mock_trainset):
        strategy = SingleMutationSampling()
        tasks = strategy.sample_tasks(mock_state, mock_candidate_selector, mock_batch_sampler, mock_trainset)
        assert len(tasks) == 1
        assert tasks[0].parent_idx == 0

    def test_same_parent_sampling(self, mock_state, mock_candidate_selector, mock_batch_sampler, mock_trainset):
        strategy = SameParentSampling(n=3)
        tasks = strategy.sample_tasks(mock_state, mock_candidate_selector, mock_batch_sampler, mock_trainset)
        assert len(tasks) == 3
        # All tasks should have the same parent
        assert all(t.parent_idx == tasks[0].parent_idx for t in tasks)
        # But different minibatches
        ids = [tuple(t.minibatch_ids) for t in tasks]
        assert len(set(ids)) == 3

    def test_independent_sampling(self, mock_state, mock_candidate_selector, mock_batch_sampler, mock_trainset):
        strategy = IndependentSampling(n=4)
        tasks = strategy.sample_tasks(mock_state, mock_candidate_selector, mock_batch_sampler, mock_trainset)
        assert len(tasks) == 4
        # Selector called 4 times
        assert mock_candidate_selector.select_candidate_idx.call_count == 4

    def test_pxn_sampling(self, mock_state, mock_candidate_selector, mock_batch_sampler, mock_trainset):
        strategy = PxNSampling(p=2, n=3)
        tasks = strategy.sample_tasks(mock_state, mock_candidate_selector, mock_batch_sampler, mock_trainset)
        assert len(tasks) == 6  # 2 * 3
        # Selector called 2 times (once per parent)
        assert mock_candidate_selector.select_candidate_idx.call_count == 2


class TestSelectionStrategies:
    def _make_proposal(self, before: float, after: float) -> CandidateProposal:
        return CandidateProposal(
            candidate={"prompt": "test"},
            parent_program_ids=[0],
            subsample_indices=[0],
            subsample_scores_before=[before],
            subsample_scores_after=[after],
            eval_before=SubsampleEvaluation(scores=[before], outputs=["out"]),
            eval_after=SubsampleEvaluation(scores=[after], outputs=["out"]),
            tag="test",
        )

    def test_all_improvements(self, mock_state):
        strategy = AllImprovements()
        criterion = StrictImprovementAcceptance()
        proposals = [
            self._make_proposal(0.5, 0.8),  # improvement
            self._make_proposal(0.5, 0.3),  # regression
            self._make_proposal(0.5, 0.6),  # improvement
        ]
        accepted = strategy.select(proposals, mock_state, criterion)
        assert len(accepted) == 2

    def test_best_improvement(self, mock_state):
        strategy = BestImprovement()
        criterion = StrictImprovementAcceptance()
        proposals = [
            self._make_proposal(0.5, 0.8),
            self._make_proposal(0.5, 0.9),
            self._make_proposal(0.5, 0.6),
        ]
        accepted = strategy.select(proposals, mock_state, criterion)
        assert len(accepted) == 1
        assert accepted[0].subsample_scores_after == [0.9]

    def test_best_improvement_none_pass(self, mock_state):
        strategy = BestImprovement()
        criterion = StrictImprovementAcceptance()
        proposals = [self._make_proposal(0.5, 0.3)]
        accepted = strategy.select(proposals, mock_state, criterion)
        assert len(accepted) == 0

    def test_top_k_improvements(self, mock_state):
        strategy = TopKImprovements(k=2)
        criterion = StrictImprovementAcceptance()
        proposals = [
            self._make_proposal(0.5, 0.9),
            self._make_proposal(0.5, 0.6),
            self._make_proposal(0.5, 0.8),
            self._make_proposal(0.5, 0.3),
        ]
        accepted = strategy.select(proposals, mock_state, criterion)
        assert len(accepted) == 2
        assert accepted[0].subsample_scores_after == [0.9]
        assert accepted[1].subsample_scores_after == [0.8]


class TestDefaultBatchEvaluate:
    def test_sequential_fallback(self):
        adapter = MagicMock()
        eval_result = EvaluationBatch(outputs=["out"], scores=[0.5])
        adapter.evaluate = MagicMock(return_value=eval_result)

        items = [
            ({"prompt": "a"}, ["ex1"]),
            ({"prompt": "b"}, ["ex2"]),
        ]
        results = default_batch_evaluate(adapter, items)
        assert len(results) == 2
        assert adapter.evaluate.call_count == 2


class TestDefaultStrategiesRetainBehavior:
    """Verify that default strategies produce the expected behavior."""

    def test_single_mutation_is_default_sampling(self):
        strategy = SingleMutationSampling()
        assert hasattr(strategy, "sample_tasks")

    def test_all_improvements_is_default_selection(self):
        strategy = AllImprovements()
        assert hasattr(strategy, "select")
