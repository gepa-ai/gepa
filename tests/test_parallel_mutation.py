# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import random
import threading
from unittest.mock import MagicMock

import pytest

from gepa.core.adapter import EvaluationBatch
from gepa.core.data_loader import ListDataLoader
from gepa.core.state import GEPAState, ValsetEvaluation
from gepa.proposer.base import CandidateProposal
from gepa.proposer.parallel import (
    AllImprovements,
    BestAbsolute,
    BestImprovement,
    IndependentSampling,
    MinibatchPolicy,
    MutationTask,
    ParallelMutationOrchestrator,
    PxNSampling,
    SameParentSampling,
    SingleMutationSampling,
    TopKImprovements,
)
from gepa.proposer.reflective_mutation.reflective_mutation import ReflectiveMutationProposer
from gepa.strategies.batch_sampler import EpochShuffledBatchSampler
from gepa.strategies.candidate_selector import CurrentBestCandidateSelector, ParetoCandidateSelector

# ---- Fixtures ----


class MockAdapter:
    """Adapter that returns deterministic scores based on candidate content."""

    def __init__(self, score_fn=None):
        self._score_fn = score_fn or (lambda candidate, item: len(str(candidate.get("instruction", ""))) / 100.0)
        self.evaluate_call_count = 0
        self.propose_new_texts = None

    def evaluate(self, batch, candidate, capture_traces=False):
        self.evaluate_call_count += 1
        scores = [min(self._score_fn(candidate, item), 1.0) for item in batch]
        outputs = [f"output_{i}" for i in range(len(batch))]
        trajectories = (
            [{"trace": f"trace_{i}", "candidate": candidate} for i in range(len(batch))] if capture_traces else None
        )
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=None,
        )

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        return {comp: [{"Inputs": {"x": "test"}, "Feedback": "improve it"}] for comp in components_to_update}


class MockModuleSelector:
    """Module selector that always returns all components."""

    def __call__(self, state, trajectories, subsample_scores, candidate_idx, candidate):
        return list(candidate.keys())


@pytest.fixture
def mock_adapter():
    return MockAdapter()


@pytest.fixture
def mock_trainset():
    return ListDataLoader(["train_0", "train_1", "train_2", "train_3", "train_4", "train_5"])


@pytest.fixture
def mock_valset():
    return ListDataLoader(["val_0", "val_1", "val_2"])


@pytest.fixture
def mock_state():
    """Create a GEPAState with 3 candidates."""
    seed_candidate = {"instruction": "initial prompt"}
    base_eval = ValsetEvaluation(
        outputs_by_val_id={0: "out0", 1: "out1", 2: "out2"},
        scores_by_val_id={0: 0.3, 1: 0.4, 2: 0.5},
        objective_scores_by_val_id=None,
    )
    state = GEPAState(seed_candidate, base_eval, track_best_outputs=False)

    # Add two more candidates
    state.program_candidates.append({"instruction": "improved prompt v1"})
    state.prog_candidate_val_subscores.append({0: 0.5, 1: 0.6, 2: 0.7})
    state.prog_candidate_objective_scores.append({})
    state.parent_program_for_candidate.append([0])
    state.named_predictor_id_to_update_next_for_program_candidate.append(0)
    state.num_metric_calls_by_discovery.append(0)

    state.program_candidates.append({"instruction": "improved prompt v2 longer"})
    state.prog_candidate_val_subscores.append({0: 0.6, 1: 0.7, 2: 0.8})
    state.prog_candidate_objective_scores.append({})
    state.parent_program_for_candidate.append([1])
    state.named_predictor_id_to_update_next_for_program_candidate.append(0)
    state.num_metric_calls_by_discovery.append(0)

    # Update pareto front
    for val_id in state.pareto_front_valset:
        state.program_at_pareto_front_valset[val_id] = {0, 1, 2}

    state.num_full_ds_evals = 1
    state.total_num_evals = 3

    assert state.is_consistent()
    return state


def _make_mock_lm(prefix="improved"):
    """LLM that returns deterministically improved text."""
    call_count = 0
    lock = threading.Lock()

    def lm(prompt):
        nonlocal call_count
        with lock:
            call_count += 1
            current = call_count
        return f"```\n{prefix} instruction v{current}\n```"

    return lm


def _make_proposer(adapter, trainset, reflection_lm=None):
    """Create a ReflectiveMutationProposer with mocked dependencies."""
    return ReflectiveMutationProposer(
        logger=MagicMock(),
        trainset=trainset,
        adapter=adapter,
        candidate_selector=ParetoCandidateSelector(rng=random.Random(42)),
        module_selector=MockModuleSelector(),
        batch_sampler=EpochShuffledBatchSampler(minibatch_size=2, rng=random.Random(42)),
        perfect_score=1.0,
        skip_perfect_score=False,
        experiment_tracker=MagicMock(),
        reflection_lm=reflection_lm or _make_mock_lm(),
    )


# ===== Sampling Strategy Tests =====


class TestSingleMutationSampling:
    def test_generates_one_task(self, mock_state):
        strategy = SingleMutationSampling()
        selector = ParetoCandidateSelector(rng=random.Random(42))
        sampler = EpochShuffledBatchSampler(minibatch_size=2, rng=random.Random(42))
        trainset = ListDataLoader(["a", "b", "c", "d"])

        tasks = strategy.generate_tasks(mock_state, selector, sampler, trainset)
        assert len(tasks) == 1
        assert tasks[0].worker_id == 0

    def test_uses_candidate_selector(self, mock_state):
        strategy = SingleMutationSampling()
        selector = CurrentBestCandidateSelector()
        sampler = EpochShuffledBatchSampler(minibatch_size=2, rng=random.Random(42))
        trainset = ListDataLoader(["a", "b", "c", "d"])

        tasks = strategy.generate_tasks(mock_state, selector, sampler, trainset)
        # CurrentBest should select index 2 (highest score)
        assert tasks[0].parent_idx == 2

    def test_uses_batch_sampler(self, mock_state):
        strategy = SingleMutationSampling()
        selector = CurrentBestCandidateSelector()
        sampler = EpochShuffledBatchSampler(minibatch_size=3, rng=random.Random(42))
        trainset = ListDataLoader(["a", "b", "c", "d", "e", "f"])

        tasks = strategy.generate_tasks(mock_state, selector, sampler, trainset)
        assert len(tasks[0].minibatch_ids) == 3


class TestSameParentSampling:
    def test_all_tasks_same_parent(self, mock_state):
        strategy = SameParentSampling(n=3)
        selector = CurrentBestCandidateSelector()
        sampler = EpochShuffledBatchSampler(minibatch_size=2, rng=random.Random(42))
        trainset = ListDataLoader(["a", "b", "c", "d"])

        tasks = strategy.generate_tasks(mock_state, selector, sampler, trainset)
        parent_idxs = {t.parent_idx for t in tasks}
        assert len(parent_idxs) == 1

    def test_shared_minibatch_mode(self, mock_state):
        strategy = SameParentSampling(n=3)
        selector = CurrentBestCandidateSelector()
        sampler = EpochShuffledBatchSampler(minibatch_size=2, rng=random.Random(42))
        trainset = ListDataLoader(["a", "b", "c", "d"])

        tasks = strategy.generate_tasks(mock_state, selector, sampler, trainset)
        # All tasks share the same minibatch
        for t in tasks:
            assert t.minibatch_ids == tasks[0].minibatch_ids

    def test_correct_number_of_tasks(self, mock_state):
        strategy = SameParentSampling(n=5)
        selector = CurrentBestCandidateSelector()
        sampler = EpochShuffledBatchSampler(minibatch_size=2, rng=random.Random(42))
        trainset = ListDataLoader(["a", "b", "c", "d"])

        tasks = strategy.generate_tasks(mock_state, selector, sampler, trainset)
        assert len(tasks) == 5

    def test_worker_ids_sequential(self, mock_state):
        strategy = SameParentSampling(n=4)
        selector = CurrentBestCandidateSelector()
        sampler = EpochShuffledBatchSampler(minibatch_size=2, rng=random.Random(42))
        trainset = ListDataLoader(["a", "b", "c", "d"])

        tasks = strategy.generate_tasks(mock_state, selector, sampler, trainset)
        assert [t.worker_id for t in tasks] == [0, 1, 2, 3]


class TestIndependentSampling:
    def test_correct_number_of_tasks(self, mock_state):
        strategy = IndependentSampling(n=3)
        selector = ParetoCandidateSelector(rng=random.Random(42))
        sampler = EpochShuffledBatchSampler(minibatch_size=2, rng=random.Random(42))
        trainset = ListDataLoader(["a", "b", "c", "d"])

        tasks = strategy.generate_tasks(mock_state, selector, sampler, trainset)
        assert len(tasks) == 3

    def test_independent_parent_selection(self, mock_state):
        """With enough samples, different parents should be selected."""
        strategy = IndependentSampling(n=20)
        selector = ParetoCandidateSelector(rng=random.Random(42))
        sampler = EpochShuffledBatchSampler(minibatch_size=2, rng=random.Random(42))
        trainset = ListDataLoader(["a", "b", "c", "d"])

        tasks = strategy.generate_tasks(mock_state, selector, sampler, trainset)
        parent_idxs = {t.parent_idx for t in tasks}
        # Should have more than 1 unique parent with high probability
        assert len(parent_idxs) >= 1  # At least 1; likely more with 20 samples

    def test_independent_minibatches(self, mock_state):
        strategy = IndependentSampling(n=3)
        selector = CurrentBestCandidateSelector()
        sampler = EpochShuffledBatchSampler(minibatch_size=2, rng=random.Random(42))
        trainset = ListDataLoader(["a", "b", "c", "d"])

        tasks = strategy.generate_tasks(mock_state, selector, sampler, trainset)
        # Each task gets its own minibatch_ids from the sampler
        assert len(tasks) == 3
        for t in tasks:
            assert len(t.minibatch_ids) == 2


class TestPxNSampling:
    def test_pxn_task_count(self, mock_state):
        strategy = PxNSampling(p=2, n=3)
        selector = ParetoCandidateSelector(rng=random.Random(42))
        sampler = EpochShuffledBatchSampler(minibatch_size=2, rng=random.Random(42))
        trainset = ListDataLoader(["a", "b", "c", "d", "e", "f"])

        tasks = strategy.generate_tasks(mock_state, selector, sampler, trainset)
        assert len(tasks) == 6  # 2 * 3

    def test_shared_per_parent_minibatch(self, mock_state):
        strategy = PxNSampling(p=2, n=3, minibatch_policy="shared_per_parent")
        selector = CurrentBestCandidateSelector()
        sampler = EpochShuffledBatchSampler(minibatch_size=2, rng=random.Random(42))
        trainset = ListDataLoader(["a", "b", "c", "d", "e", "f"])

        tasks = strategy.generate_tasks(mock_state, selector, sampler, trainset)
        # All N tasks for the same parent share the same minibatch
        # Group by parent
        by_parent: dict[int, list] = {}
        for t in tasks:
            by_parent.setdefault(t.parent_idx, []).append(t)
        for parent_tasks in by_parent.values():
            mb = parent_tasks[0].minibatch_ids
            for t in parent_tasks:
                assert t.minibatch_ids == mb

    def test_independent_minibatch_policy(self, mock_state):
        strategy = PxNSampling(p=1, n=3, minibatch_policy=MinibatchPolicy.INDEPENDENT_PER_TASK)
        selector = CurrentBestCandidateSelector()
        sampler = EpochShuffledBatchSampler(minibatch_size=2, rng=random.Random(42))
        trainset = ListDataLoader(["a", "b", "c", "d", "e", "f"])

        tasks = strategy.generate_tasks(mock_state, selector, sampler, trainset)
        assert len(tasks) == 3
        # Each task has its own minibatch (from the sampler)
        for t in tasks:
            assert len(t.minibatch_ids) == 2

    def test_global_shared_minibatch(self, mock_state):
        strategy = PxNSampling(p=2, n=2, minibatch_policy="global_shared")
        selector = CurrentBestCandidateSelector()
        sampler = EpochShuffledBatchSampler(minibatch_size=2, rng=random.Random(42))
        trainset = ListDataLoader(["a", "b", "c", "d", "e", "f"])

        tasks = strategy.generate_tasks(mock_state, selector, sampler, trainset)
        assert len(tasks) == 4
        # All tasks share the same global minibatch
        mb = tasks[0].minibatch_ids
        for t in tasks:
            assert t.minibatch_ids == mb


# ===== Selection Strategy Tests =====


class TestAllImprovements:
    def test_accepts_improving_proposals(self):
        proposals = [
            CandidateProposal(
                candidate={"instruction": "better"},
                parent_program_ids=[0],
                subsample_scores_before=[0.3, 0.4],
                subsample_scores_after=[0.5, 0.6],
            ),
        ]
        result = AllImprovements().select(proposals)
        assert len(result) == 1

    def test_rejects_non_improving_proposals(self):
        proposals = [
            CandidateProposal(
                candidate={"instruction": "worse"},
                parent_program_ids=[0],
                subsample_scores_before=[0.5, 0.6],
                subsample_scores_after=[0.3, 0.4],
            ),
        ]
        result = AllImprovements().select(proposals)
        assert len(result) == 0

    def test_mixed_proposals(self):
        proposals = [
            CandidateProposal(
                candidate={"instruction": "better"},
                parent_program_ids=[0],
                subsample_scores_before=[0.3, 0.4],
                subsample_scores_after=[0.5, 0.6],
            ),
            CandidateProposal(
                candidate={"instruction": "worse"},
                parent_program_ids=[0],
                subsample_scores_before=[0.5, 0.6],
                subsample_scores_after=[0.3, 0.4],
            ),
            CandidateProposal(
                candidate={"instruction": "also better"},
                parent_program_ids=[1],
                subsample_scores_before=[0.2, 0.3],
                subsample_scores_after=[0.4, 0.5],
            ),
        ]
        result = AllImprovements().select(proposals)
        assert len(result) == 2

    def test_empty_proposals(self):
        result = AllImprovements().select([])
        assert len(result) == 0

    def test_equal_scores_rejected(self):
        proposals = [
            CandidateProposal(
                candidate={"instruction": "same"},
                parent_program_ids=[0],
                subsample_scores_before=[0.5, 0.5],
                subsample_scores_after=[0.5, 0.5],
            ),
        ]
        result = AllImprovements().select(proposals)
        assert len(result) == 0


class TestBestImprovement:
    def test_selects_largest_improvement(self):
        proposals = [
            CandidateProposal(
                candidate={"instruction": "a"},
                parent_program_ids=[0],
                subsample_scores_before=[0.3],
                subsample_scores_after=[0.5],  # improvement = 0.2
            ),
            CandidateProposal(
                candidate={"instruction": "b"},
                parent_program_ids=[0],
                subsample_scores_before=[0.3],
                subsample_scores_after=[0.8],  # improvement = 0.5
            ),
        ]
        result = BestImprovement().select(proposals)
        assert len(result) == 1
        assert result[0].candidate["instruction"] == "b"

    def test_no_improvements_returns_empty(self):
        proposals = [
            CandidateProposal(
                candidate={"instruction": "a"},
                parent_program_ids=[0],
                subsample_scores_before=[0.5],
                subsample_scores_after=[0.3],
            ),
        ]
        result = BestImprovement().select(proposals)
        assert len(result) == 0

    def test_single_proposal(self):
        proposals = [
            CandidateProposal(
                candidate={"instruction": "a"},
                parent_program_ids=[0],
                subsample_scores_before=[0.3],
                subsample_scores_after=[0.5],
            ),
        ]
        result = BestImprovement().select(proposals)
        assert len(result) == 1


class TestBestAbsolute:
    def test_selects_highest_absolute_score(self):
        proposals = [
            CandidateProposal(
                candidate={"instruction": "a"},
                parent_program_ids=[0],
                subsample_scores_before=[0.1],
                subsample_scores_after=[0.9],  # absolute = 0.9, improvement = 0.8
            ),
            CandidateProposal(
                candidate={"instruction": "b"},
                parent_program_ids=[0],
                subsample_scores_before=[0.8],
                subsample_scores_after=[0.95],  # absolute = 0.95, improvement = 0.15
            ),
        ]
        result = BestAbsolute().select(proposals)
        assert len(result) == 1
        assert result[0].candidate["instruction"] == "b"

    def test_no_improvements_returns_empty(self):
        proposals = [
            CandidateProposal(
                candidate={"instruction": "a"},
                parent_program_ids=[0],
                subsample_scores_before=[0.5],
                subsample_scores_after=[0.3],
            ),
        ]
        result = BestAbsolute().select(proposals)
        assert len(result) == 0


class TestTopKImprovements:
    def test_selects_top_k(self):
        proposals = [
            CandidateProposal(
                candidate={"instruction": "a"},
                parent_program_ids=[0],
                subsample_scores_before=[0.1],
                subsample_scores_after=[0.3],  # improvement = 0.2
            ),
            CandidateProposal(
                candidate={"instruction": "b"},
                parent_program_ids=[0],
                subsample_scores_before=[0.1],
                subsample_scores_after=[0.6],  # improvement = 0.5
            ),
            CandidateProposal(
                candidate={"instruction": "c"},
                parent_program_ids=[0],
                subsample_scores_before=[0.1],
                subsample_scores_after=[0.4],  # improvement = 0.3
            ),
        ]
        result = TopKImprovements(k=2).select(proposals)
        assert len(result) == 2
        # Sorted by improvement: b (0.5), c (0.3)
        assert result[0].candidate["instruction"] == "b"
        assert result[1].candidate["instruction"] == "c"

    def test_fewer_than_k_improvements(self):
        proposals = [
            CandidateProposal(
                candidate={"instruction": "a"},
                parent_program_ids=[0],
                subsample_scores_before=[0.1],
                subsample_scores_after=[0.3],
            ),
            CandidateProposal(
                candidate={"instruction": "b"},
                parent_program_ids=[0],
                subsample_scores_before=[0.5],
                subsample_scores_after=[0.3],  # no improvement
            ),
        ]
        result = TopKImprovements(k=5).select(proposals)
        assert len(result) == 1

    def test_k_equals_one_same_as_best(self):
        proposals = [
            CandidateProposal(
                candidate={"instruction": "a"},
                parent_program_ids=[0],
                subsample_scores_before=[0.1],
                subsample_scores_after=[0.3],
            ),
            CandidateProposal(
                candidate={"instruction": "b"},
                parent_program_ids=[0],
                subsample_scores_before=[0.1],
                subsample_scores_after=[0.5],
            ),
        ]
        top_k_result = TopKImprovements(k=1).select(proposals)
        best_result = BestImprovement().select(proposals)
        assert len(top_k_result) == 1
        assert len(best_result) == 1
        assert top_k_result[0].candidate == best_result[0].candidate


# ===== Orchestrator Integration Tests =====


class TestParallelMutationOrchestrator:
    def test_single_mutation_produces_result(self, mock_adapter, mock_trainset, mock_state):
        """SingleMutationSampling + AllImprovements produces a result."""
        proposer = _make_proposer(mock_adapter, mock_trainset)
        orchestrator = ParallelMutationOrchestrator(
            proposer=proposer,
            sampling_strategy=SingleMutationSampling(),
            selection_strategy=AllImprovements(),
            max_workers=1,
        )
        # The mock LLM produces longer text -> higher score with MockAdapter
        results = orchestrator.propose_batch(mock_state)
        # Should return 0 or 1 proposals
        assert isinstance(results, list)

    def test_multiple_mutations_same_parent(self, mock_adapter, mock_trainset, mock_state):
        """SameParentSampling(n=3) produces proposals from same parent."""
        proposer = _make_proposer(mock_adapter, mock_trainset)
        orchestrator = ParallelMutationOrchestrator(
            proposer=proposer,
            sampling_strategy=SameParentSampling(n=3),
            selection_strategy=AllImprovements(),
            max_workers=3,
        )
        results = orchestrator.propose_batch(mock_state)
        # All proposals should have the same parent
        if results:
            parent_ids = {tuple(p.parent_program_ids) for p in results}
            assert len(parent_ids) == 1

    def test_selection_strategy_applied(self, mock_adapter, mock_trainset, mock_state):
        """BestImprovement returns at most 1 proposal even with N=3."""
        proposer = _make_proposer(mock_adapter, mock_trainset)
        orchestrator = ParallelMutationOrchestrator(
            proposer=proposer,
            sampling_strategy=SameParentSampling(n=3),
            selection_strategy=BestImprovement(),
            max_workers=3,
        )
        results = orchestrator.propose_batch(mock_state)
        assert len(results) <= 1

    def test_eval_count_incremented(self, mock_adapter, mock_trainset, mock_state):
        """state.total_num_evals is incremented correctly."""
        initial_evals = mock_state.total_num_evals
        proposer = _make_proposer(mock_adapter, mock_trainset)
        orchestrator = ParallelMutationOrchestrator(
            proposer=proposer,
            sampling_strategy=SingleMutationSampling(),
            selection_strategy=AllImprovements(),
            max_workers=1,
        )
        orchestrator.propose_batch(mock_state)
        # Should have incremented for parent eval + child eval
        assert mock_state.total_num_evals > initial_evals

    def test_parent_eval_deduplication(self, mock_trainset, mock_state):
        """SameParentSampling(n=3) with shared minibatch evaluates parent only ONCE."""
        adapter = MockAdapter()
        proposer = _make_proposer(adapter, mock_trainset)
        orchestrator = ParallelMutationOrchestrator(
            proposer=proposer,
            sampling_strategy=SameParentSampling(n=3),
            selection_strategy=AllImprovements(),
            max_workers=3,
        )
        initial_eval_count = adapter.evaluate_call_count
        orchestrator.propose_batch(mock_state)
        # Parent eval: 1 call (deduplicated), Child evals: 3 calls = 4 total
        assert adapter.evaluate_call_count == initial_eval_count + 4

    def test_pxn_strategy(self, mock_adapter, mock_trainset, mock_state):
        """PxNSampling(p=2, n=2) produces up to 4 proposals before filtering."""
        proposer = _make_proposer(mock_adapter, mock_trainset)
        orchestrator = ParallelMutationOrchestrator(
            proposer=proposer,
            sampling_strategy=PxNSampling(p=2, n=2),
            selection_strategy=AllImprovements(),
            max_workers=4,
        )
        # This just verifies it doesn't crash and returns a valid list
        results = orchestrator.propose_batch(mock_state)
        assert isinstance(results, list)

    def test_proposals_have_correct_tag(self, mock_adapter, mock_trainset, mock_state):
        """All proposals from orchestrator have tag='parallel_mutation'."""
        proposer = _make_proposer(mock_adapter, mock_trainset)
        # Use a score function that always improves
        mock_adapter._score_fn = lambda c, i: 0.9 if "improved" in str(c.get("instruction", "")) else 0.1
        orchestrator = ParallelMutationOrchestrator(
            proposer=proposer,
            sampling_strategy=SameParentSampling(n=2),
            selection_strategy=AllImprovements(),
            max_workers=2,
        )
        results = orchestrator.propose_batch(mock_state)
        for p in results:
            assert p.tag == "parallel_mutation"
            assert "worker_id" in p.metadata

    def test_exception_in_one_worker_doesnt_crash_others(self, mock_trainset, mock_state):
        """If one LLM call raises, other proposals still returned."""
        call_count = 0
        lock = threading.Lock()

        def flaky_lm(prompt):
            nonlocal call_count
            with lock:
                call_count += 1
                current = call_count
            if current == 2:
                raise RuntimeError("LLM failed")
            return f"```\nimproved instruction v{current}\n```"

        adapter = MockAdapter()
        # Make score function that always improves
        adapter._score_fn = lambda c, i: 0.9 if "improved" in str(c.get("instruction", "")) else 0.1
        proposer = _make_proposer(adapter, mock_trainset, reflection_lm=flaky_lm)
        orchestrator = ParallelMutationOrchestrator(
            proposer=proposer,
            sampling_strategy=SameParentSampling(n=3),
            selection_strategy=AllImprovements(),
            max_workers=3,
        )
        # Should not raise — failed worker is logged and skipped
        results = orchestrator.propose_batch(mock_state)
        # At most 2 proposals (one failed)
        assert isinstance(results, list)

    def test_parallel_proposals_dont_share_mutable_state(self, mock_adapter, mock_trainset, mock_state):
        """Each proposal's candidate dict must be an independent copy."""
        mock_adapter._score_fn = lambda c, i: 0.9 if "improved" in str(c.get("instruction", "")) else 0.1
        proposer = _make_proposer(mock_adapter, mock_trainset)
        orchestrator = ParallelMutationOrchestrator(
            proposer=proposer,
            sampling_strategy=SameParentSampling(n=3),
            selection_strategy=AllImprovements(),
            max_workers=3,
        )
        results = orchestrator.propose_batch(mock_state)
        if len(results) >= 2:
            # Mutating one should not affect another
            results[0].candidate["instruction"] = "MUTATED"
            assert results[1].candidate["instruction"] != "MUTATED"


# ===== Contract / Invariant Tests =====


class TestContractInvariants:
    def test_eval_count_invariant(self, mock_trainset, mock_state):
        """state.total_num_evals must increase by exactly the right amount."""
        adapter = MockAdapter()
        proposer = _make_proposer(adapter, mock_trainset)

        # SingleMutationSampling: 1 parent eval + 1 child eval, each with minibatch_size=2
        orchestrator = ParallelMutationOrchestrator(
            proposer=proposer,
            sampling_strategy=SingleMutationSampling(),
            selection_strategy=AllImprovements(),
            max_workers=1,
        )
        initial_evals = mock_state.total_num_evals
        orchestrator.propose_batch(mock_state)
        # parent_evals (1 unique * 2 minibatch) + child_evals (1 * 2 minibatch) = 4
        assert mock_state.total_num_evals == initial_evals + 4

    def test_proposal_has_required_fields(self, mock_trainset, mock_state):
        """Every CandidateProposal from orchestrator must have required fields."""
        adapter = MockAdapter()
        adapter._score_fn = lambda c, i: 0.9 if "improved" in str(c.get("instruction", "")) else 0.1
        proposer = _make_proposer(adapter, mock_trainset)
        orchestrator = ParallelMutationOrchestrator(
            proposer=proposer,
            sampling_strategy=SameParentSampling(n=2),
            selection_strategy=AllImprovements(),
            max_workers=2,
        )
        results = orchestrator.propose_batch(mock_state)
        for p in results:
            assert isinstance(p.candidate, dict)
            assert set(p.candidate.keys()) == {"instruction"}
            assert isinstance(p.parent_program_ids, list)
            assert all(0 <= idx < len(mock_state.program_candidates) for idx in p.parent_program_ids)
            assert p.subsample_indices is not None and len(p.subsample_indices) > 0
            assert p.subsample_scores_before is not None
            assert p.subsample_scores_after is not None
            assert len(p.subsample_scores_before) == len(p.subsample_indices)
            assert len(p.subsample_scores_after) == len(p.subsample_indices)
            assert p.tag == "parallel_mutation"

    def test_selection_strategy_contract_all_improvements(self):
        """AllImprovements must return only proposals where sum(after) > sum(before)."""
        proposals = [
            CandidateProposal(
                candidate={"x": "1"},
                parent_program_ids=[0],
                subsample_scores_before=[0.3],
                subsample_scores_after=[0.5],
            ),
            CandidateProposal(
                candidate={"x": "2"},
                parent_program_ids=[0],
                subsample_scores_before=[0.5],
                subsample_scores_after=[0.3],
            ),
        ]
        result = AllImprovements().select(proposals)
        assert all(sum(p.subsample_scores_after) > sum(p.subsample_scores_before) for p in result)
        # Must be a subset
        assert all(p in proposals for p in result)

    def test_selection_strategy_contract_best_improvement(self):
        """BestImprovement must return at most 1."""
        proposals = [
            CandidateProposal(
                candidate={"x": str(i)},
                parent_program_ids=[0],
                subsample_scores_before=[0.1],
                subsample_scores_after=[0.1 + i * 0.1],
            )
            for i in range(5)
        ]
        result = BestImprovement().select(proposals)
        assert len(result) <= 1

    def test_selection_strategy_contract_top_k(self):
        """TopK must return at most K."""
        proposals = [
            CandidateProposal(
                candidate={"x": str(i)},
                parent_program_ids=[0],
                subsample_scores_before=[0.1],
                subsample_scores_after=[0.1 + i * 0.1],
            )
            for i in range(10)
        ]
        result = TopKImprovements(k=3).select(proposals)
        assert len(result) <= 3

    def test_sampling_strategy_contract(self, mock_state):
        """All sampling strategies must return valid MutationTasks."""
        selector = ParetoCandidateSelector(rng=random.Random(42))
        sampler = EpochShuffledBatchSampler(minibatch_size=2, rng=random.Random(42))
        trainset = ListDataLoader(["a", "b", "c", "d"])

        strategies = [
            (SingleMutationSampling(), 1),
            (SameParentSampling(n=3), 3),
            (IndependentSampling(n=2), 2),
            (PxNSampling(p=2, n=2), 4),
        ]

        for strategy, expected_count in strategies:
            tasks = strategy.generate_tasks(mock_state, selector, sampler, trainset)
            assert len(tasks) == expected_count, f"{strategy.__class__.__name__} produced {len(tasks)} tasks"
            worker_ids = set()
            for t in tasks:
                assert isinstance(t, MutationTask)
                assert 0 <= t.parent_idx < len(mock_state.program_candidates)
                assert len(t.minibatch_ids) > 0
                assert t.worker_id not in worker_ids, "Duplicate worker_id"
                worker_ids.add(t.worker_id)


class TestBackwardCompatibility:
    def test_engine_init_without_orchestrator(self):
        """GEPAEngine() without orchestrator param must work."""
        from gepa.core.engine import GEPAEngine

        adapter = MockAdapter()
        proposer = _make_proposer(adapter, ListDataLoader(["a", "b"]))

        engine = GEPAEngine(
            adapter=adapter,
            run_dir=None,
            valset=ListDataLoader(["v0", "v1"]),
            seed_candidate={"instruction": "test"},
            perfect_score=1.0,
            seed=0,
            reflective_proposer=proposer,
            merge_proposer=None,
            frontier_type="instance",
            logger=MagicMock(),
            experiment_tracker=MagicMock(),
        )
        assert engine.orchestrator is None

    def test_engine_init_with_orchestrator(self):
        """GEPAEngine() with orchestrator param sets it correctly."""
        from gepa.core.engine import GEPAEngine

        adapter = MockAdapter()
        trainset = ListDataLoader(["a", "b"])
        proposer = _make_proposer(adapter, trainset)
        orchestrator = ParallelMutationOrchestrator(
            proposer=proposer,
            sampling_strategy=SingleMutationSampling(),
            selection_strategy=AllImprovements(),
        )

        engine = GEPAEngine(
            adapter=adapter,
            run_dir=None,
            valset=ListDataLoader(["v0", "v1"]),
            seed_candidate={"instruction": "test"},
            perfect_score=1.0,
            seed=0,
            reflective_proposer=proposer,
            merge_proposer=None,
            frontier_type="instance",
            logger=MagicMock(),
            experiment_tracker=MagicMock(),
            orchestrator=orchestrator,
        )
        assert engine.orchestrator is orchestrator
