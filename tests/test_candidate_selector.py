# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import random

import pytest

from gepa.core.state import GEPAState, ValsetEvaluation
from gepa.strategies.candidate_selector import (
    BatchLexicaseCandidateSelector,
    CurrentBestCandidateSelector,
    EpsilonGreedyCandidateSelector,
    ParetoCandidateSelector,
    UnprunedParetoCandidateSelector,
)


@pytest.fixture
def mock_state():
    """Create a mock GEPAState with 3 candidates for testing."""
    seed_candidate = {"system_prompt": "test"}
    # GEPAState expects ValsetEvaluation with dicts for outputs and scores keyed by data IDs
    base_valset_eval_output = ValsetEvaluation(
        outputs_by_val_id={0: "out1", 1: "out2", 2: "out3"},
        scores_by_val_id={0: 0.5, 1: 0.3, 2: 0.7},
        objective_scores_by_val_id=None,
    )
    state = GEPAState(seed_candidate, base_valset_eval_output, track_best_outputs=False)

    # Add two more candidates with different scores
    state.program_candidates.append({"system_prompt": "test2"})
    state.program_candidates.append({"system_prompt": "test3"})

    # prog_candidate_val_subscores should be dicts, not lists
    state.prog_candidate_val_subscores.append({0: 0.6, 1: 0.6, 2: 0.6})
    state.prog_candidate_val_subscores.append({0: 0.8, 1: 0.8, 2: 0.8})

    # Add entries to prog_candidate_objective_scores to maintain consistency
    state.prog_candidate_objective_scores.append({})
    state.prog_candidate_objective_scores.append({})

    state.parent_program_for_candidate.append([0])
    state.parent_program_for_candidate.append([1])

    state.named_predictor_id_to_update_next_for_program_candidate.append(0)
    state.named_predictor_id_to_update_next_for_program_candidate.append(0)

    state.num_metric_calls_by_discovery.append(0)
    state.num_metric_calls_by_discovery.append(0)

    state.iteration_ids_by_candidate_idx.append("1")
    state.iteration_ids_by_candidate_idx.append("2")

    # Update pareto front to include all candidates
    for i in range(len(state.pareto_front_valset)):
        state.program_at_pareto_front_valset[i] = {0, 1, 2}

    assert state.is_consistent()
    return state


class TestCurrentBestCandidateSelector:
    def test_selects_best_candidate(self, mock_state):
        """Test that CurrentBestCandidateSelector always selects the candidate with highest score."""
        selector = CurrentBestCandidateSelector()

        # Best candidate is at index 2 with score 0.8
        selected_idx = selector.select_candidate_idx(mock_state)
        assert selected_idx == 2
        assert mock_state.program_full_scores_val_set[selected_idx] == pytest.approx(0.8)

    def test_deterministic(self, mock_state):
        """Test that CurrentBestCandidateSelector is deterministic."""
        selector = CurrentBestCandidateSelector()

        results = [selector.select_candidate_idx(mock_state) for _ in range(10)]
        assert all(r == results[0] for r in results)


class TestParetoCandidateSelector:
    def test_samples_from_pareto_front(self, mock_state):
        """Test that ParetoCandidateSelector samples from Pareto front candidates."""
        rng = random.Random(42)
        selector = ParetoCandidateSelector(rng)

        # Sample multiple times and ensure all are from Pareto front
        samples = [selector.select_candidate_idx(mock_state) for _ in range(20)]

        # All samples should be valid indices
        for idx in samples:
            assert 0 <= idx < len(mock_state.program_candidates)

    def test_seeding_produces_deterministic_sequence(self, mock_state):
        """Test that same seed produces same sequence of selections."""
        selector1 = ParetoCandidateSelector(random.Random(42))
        selector2 = ParetoCandidateSelector(random.Random(42))

        results1 = [selector1.select_candidate_idx(mock_state) for _ in range(10)]
        results2 = [selector2.select_candidate_idx(mock_state) for _ in range(10)]

        assert results1 == results2

    def test_default_rng_with_seed_zero(self, mock_state):
        """Test that None rng defaults to Random(0)."""
        selector = ParetoCandidateSelector(None)

        # Should be deterministic with default seed
        results = [selector.select_candidate_idx(mock_state) for _ in range(5)]

        # Compare with explicit Random(0)
        selector_explicit = ParetoCandidateSelector(random.Random(0))
        results_explicit = [selector_explicit.select_candidate_idx(mock_state) for _ in range(5)]

        assert results == results_explicit


class TestEpsilonGreedyCandidateSelector:
    def test_epsilon_zero_always_exploits(self, mock_state):
        """Test that epsilon=0 always selects the best candidate (pure exploitation)."""
        rng = random.Random(42)
        selector = EpsilonGreedyCandidateSelector(epsilon=0.0, rng=rng)

        # Should always select best candidate (index 2)
        results = [selector.select_candidate_idx(mock_state) for _ in range(20)]
        assert all(r == 2 for r in results)

    def test_epsilon_one_always_explores(self, mock_state):
        """Test that epsilon=1.0 always explores randomly."""
        rng = random.Random(42)
        selector = EpsilonGreedyCandidateSelector(epsilon=1.0, rng=rng)

        # Should select randomly, so we expect variation
        results = [selector.select_candidate_idx(mock_state) for _ in range(50)]

        # Should have at least 2 different selections in 50 trials
        assert len(set(results)) > 1

        # All selections should be valid
        for idx in results:
            assert 0 <= idx < len(mock_state.program_candidates)

    def test_seeding_produces_deterministic_sequence(self, mock_state):
        """Test that same seed produces same sequence of selections."""
        selector1 = EpsilonGreedyCandidateSelector(epsilon=0.3, rng=random.Random(42))
        selector2 = EpsilonGreedyCandidateSelector(epsilon=0.3, rng=random.Random(42))

        results1 = [selector1.select_candidate_idx(mock_state) for _ in range(10)]
        results2 = [selector2.select_candidate_idx(mock_state) for _ in range(10)]

        assert results1 == results2

    def test_invalid_epsilon_raises_assertion(self):
        """Test that invalid epsilon values raise AssertionError."""
        rng = random.Random(42)

        with pytest.raises(AssertionError):
            EpsilonGreedyCandidateSelector(epsilon=-0.1, rng=rng)

        with pytest.raises(AssertionError):
            EpsilonGreedyCandidateSelector(epsilon=1.5, rng=rng)

    def test_boundary_epsilon_values(self, mock_state):
        """Test that boundary epsilon values (0.0 and 1.0) work correctly."""
        # Test epsilon=0.0
        selector_zero = EpsilonGreedyCandidateSelector(epsilon=0.0, rng=random.Random(42))
        assert selector_zero.epsilon == 0.0

        # Test epsilon=1.0
        selector_one = EpsilonGreedyCandidateSelector(epsilon=1.0, rng=random.Random(42))
        assert selector_one.epsilon == 1.0

    def test_selects_best_when_not_exploring(self, mock_state):
        """Test that when not exploring, the best candidate is selected."""
        # Use a very small epsilon to mostly exploit
        selector = EpsilonGreedyCandidateSelector(epsilon=0.01, rng=random.Random(42))

        # Most selections should be the best (index 2)
        results = [selector.select_candidate_idx(mock_state) for _ in range(100)]
        best_count = sum(1 for r in results if r == 2)

        # With epsilon=0.01, expect ~99% to be best
        assert best_count >= 90


@pytest.fixture
def dominated_champion_state():
    """State reproducing the binary-metric pathology: candidate 3 has the best
    aggregate score, but every instance it wins is shared with an irremovable
    sole-solver specialist, so dominance pruning excludes it from Pareto sampling."""
    seed_candidate = {"system_prompt": "seed"}
    base_valset_eval_output = ValsetEvaluation(
        outputs_by_val_id=dict.fromkeys(range(6), "o"),
        scores_by_val_id=dict.fromkeys(range(6), 0.0),
        objective_scores_by_val_id=None,
    )
    state = GEPAState(seed_candidate, base_valset_eval_output, track_best_outputs=False)

    # specialist A uniquely solves instance 0, specialist B uniquely solves instance 1:
    # both are unremovable. The champion solves the most instances (2..5), each shared
    # with one of the specialists, so it holds no unique slot.
    subscores = [
        {0: 1.0, 1: 0.0, 2: 1.0, 3: 0.0, 4: 1.0, 5: 0.0},  # specialist A
        {0: 0.0, 1: 1.0, 2: 0.0, 3: 1.0, 4: 0.0, 5: 1.0},  # specialist B
        {0: 0.0, 1: 0.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0},  # champion
    ]
    for i, scores in enumerate(subscores, start=1):
        state.program_candidates.append({"system_prompt": f"cand{i}"})
        state.prog_candidate_val_subscores.append(scores)
        state.prog_candidate_objective_scores.append({})
        state.parent_program_for_candidate.append([0])
        state.named_predictor_id_to_update_next_for_program_candidate.append(0)
        state.num_metric_calls_by_discovery.append(0)
        state.iteration_ids_by_candidate_idx.append(str(i))

    state.pareto_front_valset = dict.fromkeys(range(6), 1.0)
    state.program_at_pareto_front_valset = {
        0: {1},
        1: {2},
        2: {1, 3},
        3: {2, 3},
        4: {1, 3},
        5: {2, 3},
    }
    assert state.is_consistent()
    return state


class TestBatchLexicaseCandidateSelector:
    def test_selects_valid_indices(self, mock_state):
        selector = BatchLexicaseCandidateSelector(batch_size=2, rng=random.Random(42))
        for _ in range(20):
            idx = selector.select_candidate_idx(mock_state)
            assert 0 <= idx < len(mock_state.program_candidates)

    def test_full_batch_recovers_argmax_aggregate(self, mock_state):
        """A batch spanning the whole valset degenerates to argmax by aggregate score."""
        selector = BatchLexicaseCandidateSelector(batch_size=3, rng=random.Random(42))
        results = [selector.select_candidate_idx(mock_state) for _ in range(20)]
        assert all(r == 2 for r in results)

    def test_seeding_produces_deterministic_sequence(self, mock_state):
        selector1 = BatchLexicaseCandidateSelector(batch_size=2, rng=random.Random(42))
        selector2 = BatchLexicaseCandidateSelector(batch_size=2, rng=random.Random(42))
        results1 = [selector1.select_candidate_idx(mock_state) for _ in range(10)]
        results2 = [selector2.select_candidate_idx(mock_state) for _ in range(10)]
        assert results1 == results2

    def test_batch_size_one_is_plain_lexicase(self, mock_state):
        selector = BatchLexicaseCandidateSelector(batch_size=1, rng=random.Random(42))
        for _ in range(10):
            assert 0 <= selector.select_candidate_idx(mock_state) < 3

    def test_invalid_batch_size_raises_assertion(self):
        with pytest.raises(AssertionError):
            BatchLexicaseCandidateSelector(batch_size=0)

    def test_dominated_champion_keeps_selection_mass(self, dominated_champion_state):
        """The best-aggregate candidate is excluded by Pareto dominance pruning but
        must remain selectable under batch lexicase."""
        pareto = ParetoCandidateSelector(random.Random(42))
        pareto_samples = {pareto.select_candidate_idx(dominated_champion_state) for _ in range(200)}
        assert 3 not in pareto_samples  # documents the pathology being fixed

        lexicase = BatchLexicaseCandidateSelector(batch_size=2, rng=random.Random(42))
        lexicase_samples = [lexicase.select_candidate_idx(dominated_champion_state) for _ in range(200)]
        champion_share = sum(1 for s in lexicase_samples if s == 3) / len(lexicase_samples)
        assert champion_share > 0.5


class TestUnprunedParetoCandidateSelector:
    def test_samples_all_front_members(self, dominated_champion_state):
        """Every front member, including the dominance-pruned champion, gets mass."""
        selector = UnprunedParetoCandidateSelector(rng=random.Random(42))
        samples = {selector.select_candidate_idx(dominated_champion_state) for _ in range(300)}
        assert 3 in samples

    def test_seeding_produces_deterministic_sequence(self, mock_state):
        selector1 = UnprunedParetoCandidateSelector(rng=random.Random(42))
        selector2 = UnprunedParetoCandidateSelector(rng=random.Random(42))
        results1 = [selector1.select_candidate_idx(mock_state) for _ in range(10)]
        results2 = [selector2.select_candidate_idx(mock_state) for _ in range(10)]
        assert results1 == results2


class TestLatestCandidateSelector:
    def test_selects_newest_candidate(self, mock_state):
        from gepa.strategies.candidate_selector import LatestCandidateSelector

        selector = LatestCandidateSelector()
        assert selector.select_candidate_idx(mock_state) == 2

        mock_state.program_candidates.append({"system_prompt": "test4"})
        mock_state.prog_candidate_val_subscores.append({0: 0.1, 1: 0.1, 2: 0.1})
        mock_state.prog_candidate_objective_scores.append({})
        mock_state.parent_program_for_candidate.append([2])
        mock_state.named_predictor_id_to_update_next_for_program_candidate.append(0)
        mock_state.num_metric_calls_by_discovery.append(0)
        mock_state.iteration_ids_by_candidate_idx.append("3")
        assert selector.select_candidate_idx(mock_state) == 3


class TestAlwaysAcceptance:
    def test_accepts_everything(self):
        from gepa.strategies.acceptance import AlwaysAcceptance

        class FakeProposal:
            def __init__(self):
                self.subsample_scores_before = [1.0, 1.0, 1.0]
                self.subsample_scores_after = [0.0, 0.0, 0.0]

        assert AlwaysAcceptance().should_accept(FakeProposal(), state=None) is True
