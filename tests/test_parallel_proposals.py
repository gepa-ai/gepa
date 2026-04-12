# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for batch-based parallel proposals."""

from unittest.mock import MagicMock

import pytest

from gepa.core.adapter import EvaluationBatch, default_batch_evaluate
from gepa.core.state import GEPAState, ValsetEvaluation
from gepa.proposer.base import CandidateProposal, SubsampleEvaluation
from gepa.proposer.parallel import (
    AllImprovements,
    BestImprovement,
    IndependentSampling,
    ParallelConfig,
    ProposalTask,
    PxNSampling,
    SameParentSampling,
    SingleMutationSampling,
    TopKImprovements,
    propose_batch,
)
from gepa.strategies.acceptance import StrictImprovementAcceptance


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
        results = default_batch_evaluate(adapter, items, capture_traces=True)
        assert len(results) == 2
        assert adapter.evaluate.call_count == 2


class TestParallelConfig:
    def test_dataclass_creation(self):
        config = ParallelConfig(
            sampling_strategy=SingleMutationSampling(),
            selection_strategy=AllImprovements(),
        )
        assert config.sampling_strategy is not None
        assert config.selection_strategy is not None


class TestProposeBatch:
    def test_propose_batch_no_tasks(self):
        """Empty tasks -> empty result."""
        proposer = MagicMock()
        adapter = MagicMock()
        state = MagicMock()
        strategy = MagicMock()
        strategy.sample_tasks = MagicMock(return_value=[])
        selection = AllImprovements()
        criterion = StrictImprovementAcceptance()

        result = propose_batch(
            proposer, adapter, state, strategy, selection, criterion,
            MagicMock(), MagicMock()
        )
        assert result == []

    def test_propose_batch_no_trajectories(self):
        """When parent eval has no trajectories, children list is empty."""
        proposer = MagicMock()
        proposer.candidate_selector = MagicMock()
        proposer.batch_sampler = MagicMock()
        proposer.trainset = MagicMock()
        proposer.skip_perfect_score = False
        proposer.perfect_score = None

        adapter = MagicMock()
        # Return eval with no trajectories
        eval_no_traces = EvaluationBatch(outputs=["out"], scores=[0.5], trajectories=None)
        adapter.batch_evaluate = MagicMock(return_value=[eval_no_traces])

        state = MagicMock()
        state.increment_evals = MagicMock()

        task = ProposalTask(
            parent_idx=0,
            parent_candidate={"prompt": "test"},
            minibatch_ids=[0],
            minibatch=["example"],
        )
        strategy = MagicMock()
        strategy.sample_tasks = MagicMock(return_value=[task])
        selection = AllImprovements()
        criterion = StrictImprovementAcceptance()

        result = propose_batch(
            proposer, adapter, state, strategy, selection, criterion,
            MagicMock(), MagicMock()
        )
        assert result == []

    def test_propose_batch_with_improvement(self):
        """Full pipeline with a proposal that improves."""
        proposer = MagicMock()
        proposer.candidate_selector = MagicMock()
        proposer.batch_sampler = MagicMock()
        proposer.trainset = MagicMock()
        proposer.skip_perfect_score = False
        proposer.perfect_score = None
        proposer.module_selector = MagicMock(return_value=["system_prompt"])
        proposer.propose_new_texts = MagicMock(
            return_value=({"system_prompt": "improved"}, {"system_prompt": "prompt"}, {"system_prompt": "raw"})
        )

        adapter = MagicMock()
        parent_eval = EvaluationBatch(
            outputs=["out"], scores=[0.5],
            trajectories=["trace"], objective_scores=None,
        )
        child_eval = EvaluationBatch(
            outputs=["out2"], scores=[0.9],
            trajectories=["trace2"], objective_scores=None,
        )
        adapter.batch_evaluate = MagicMock(side_effect=[[parent_eval], [child_eval]])
        adapter.make_reflective_dataset = MagicMock(return_value={"system_prompt": [{"data": "x"}]})

        state = MagicMock()
        state.increment_evals = MagicMock()

        task = ProposalTask(
            parent_idx=0,
            parent_candidate={"system_prompt": "test"},
            minibatch_ids=[0],
            minibatch=["example"],
        )
        strategy = MagicMock()
        strategy.sample_tasks = MagicMock(return_value=[task])
        selection = AllImprovements()
        criterion = StrictImprovementAcceptance()

        result = propose_batch(
            proposer, adapter, state, strategy, selection, criterion,
            MagicMock(), MagicMock()
        )
        assert len(result) == 1
        assert result[0].candidate == {"system_prompt": "improved"}
        assert result[0].subsample_scores_after == [0.9]
        assert result[0].tag == "parallel_mutation"

    def test_deduplication(self):
        """Same parent + same minibatch should only be evaluated once."""
        proposer = MagicMock()
        proposer.candidate_selector = MagicMock()
        proposer.batch_sampler = MagicMock()
        proposer.trainset = MagicMock()
        proposer.skip_perfect_score = False
        proposer.perfect_score = None
        proposer.module_selector = MagicMock(return_value=["prompt"])
        proposer.propose_new_texts = MagicMock(
            return_value=({"prompt": "new"}, {"prompt": "p"}, {"prompt": "r"})
        )

        adapter = MagicMock()
        parent_eval = EvaluationBatch(
            outputs=["out"], scores=[0.5],
            trajectories=["trace"], objective_scores=None,
        )
        child_eval_1 = EvaluationBatch(
            outputs=["out1"], scores=[0.6], trajectories=["t1"], objective_scores=None,
        )
        child_eval_2 = EvaluationBatch(
            outputs=["out2"], scores=[0.7], trajectories=["t2"], objective_scores=None,
        )
        # First call: parent evals (should be 1 due to dedup). Second call: child evals (2).
        adapter.batch_evaluate = MagicMock(side_effect=[[parent_eval], [child_eval_1, child_eval_2]])
        adapter.make_reflective_dataset = MagicMock(return_value={"prompt": [{"data": "x"}]})

        state = MagicMock()
        state.increment_evals = MagicMock()

        # Two tasks with same parent and same minibatch_ids
        task1 = ProposalTask(0, {"prompt": "test"}, [0], ["example"])
        task2 = ProposalTask(0, {"prompt": "test"}, [0], ["example"])
        strategy = MagicMock()
        strategy.sample_tasks = MagicMock(return_value=[task1, task2])
        selection = AllImprovements()
        criterion = StrictImprovementAcceptance()

        propose_batch(
            proposer, adapter, state, strategy, selection, criterion,
            MagicMock(), MagicMock()
        )
        # Parent eval should have been called with 1 item (deduplicated)
        parent_call_items = adapter.batch_evaluate.call_args_list[0][0][0]
        assert len(parent_call_items) == 1
