# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for LearnabilityBatchSampler."""

import random

import pytest

from gepa.core.data_loader import ensure_loader
from gepa.core.state import GEPAState, ValsetEvaluation
from gepa.strategies.learnability_sampler import LearnabilityBatchSampler


@pytest.fixture
def state_with_candidates():
    """GEPAState with 4 candidates showing different score patterns."""
    seed = {"prompt": "seed"}
    # 5 val examples with deliberate score patterns:
    #   example 0: all candidates score ~0.5 (low variance)
    #   example 1: scores vary widely 0.1-0.9 (high variance)
    #   example 2: all candidates score ~0.9 (low variance, high)
    #   example 3: scores vary moderately (medium variance)
    #   example 4: all candidates score ~0.0 (low variance, low)
    base_eval = ValsetEvaluation(
        outputs_by_val_id={0: "a", 1: "b", 2: "c", 3: "d", 4: "e"},
        scores_by_val_id={0: 0.5, 1: 0.1, 2: 0.9, 3: 0.3, 4: 0.0},
        objective_scores_by_val_id=None,
    )
    state = GEPAState(seed, base_eval)
    state.total_num_evals = 20
    state.num_full_ds_evals = 4

    # Add 3 more candidates with different score patterns
    for scores in [
        {0: 0.5, 1: 0.9, 2: 0.9, 3: 0.7, 4: 0.0},
        {0: 0.4, 1: 0.5, 2: 0.8, 3: 0.4, 4: 0.1},
        {0: 0.6, 1: 0.8, 2: 0.9, 3: 0.6, 4: 0.0},
    ]:
        state.program_candidates.append({"prompt": f"cand_{len(state.program_candidates)}"})
        state.prog_candidate_val_subscores.append(scores)
        state.prog_candidate_objective_scores.append({})
        state.parent_program_for_candidate.append([0])
        state.named_predictor_id_to_update_next_for_program_candidate.append(0)
        state.num_metric_calls_by_discovery.append(0)

    for val_id in state.pareto_front_valset:
        state.program_at_pareto_front_valset[val_id] = {0, 1, 2, 3}

    assert state.is_consistent()
    return state


class TestLearnabilityComputation:
    def test_high_variance_example_has_highest_learnability(self, state_with_candidates):
        sampler = LearnabilityBatchSampler(minibatch_size=2, min_candidates=3)
        learnability = sampler._compute_learnability(state_with_candidates)

        # Example 1 has scores [0.1, 0.9, 0.5, 0.8] — highest variance
        # Example 0 has scores [0.5, 0.5, 0.4, 0.6] — low variance
        # Example 2 has scores [0.9, 0.9, 0.8, 0.9] — low variance
        assert learnability[1] > learnability[0]
        assert learnability[1] > learnability[2]

    def test_min_candidates_threshold(self, state_with_candidates):
        # With min_candidates=10, nothing should qualify
        sampler = LearnabilityBatchSampler(minibatch_size=2, min_candidates=10)
        learnability = sampler._compute_learnability(state_with_candidates)
        assert len(learnability) == 0


class TestSamplingBehavior:
    def test_temperature_zero_picks_highest_variance(self, state_with_candidates):
        loader = ensure_loader(["ex0", "ex1", "ex2", "ex3", "ex4"])
        sampler = LearnabilityBatchSampler(
            minibatch_size=2, min_candidates=3, temperature=0.0, rng=random.Random(42)
        )
        ids = sampler.next_minibatch_ids(loader, state_with_candidates)
        assert len(ids) == 2
        # Example 1 (highest variance) should always be included
        assert 1 in ids

    def test_fallback_when_too_few_candidates(self):
        """Falls back to EpochShuffledBatchSampler when min_candidates not met."""
        seed = {"prompt": "seed"}
        base_eval = ValsetEvaluation(
            outputs_by_val_id={0: "a", 1: "b"},
            scores_by_val_id={0: 0.5, 1: 0.5},
            objective_scores_by_val_id=None,
        )
        state = GEPAState(seed, base_eval)
        state.total_num_evals = 2
        state.num_full_ds_evals = 1
        state.i = 0

        loader = ensure_loader(["ex0", "ex1"])
        sampler = LearnabilityBatchSampler(minibatch_size=2, min_candidates=3)
        ids = sampler.next_minibatch_ids(loader, state)
        assert len(ids) == 2

    def test_deterministic_with_seed(self, state_with_candidates):
        loader = ensure_loader(["ex0", "ex1", "ex2", "ex3", "ex4"])
        sampler1 = LearnabilityBatchSampler(minibatch_size=3, min_candidates=3, rng=random.Random(42))
        sampler2 = LearnabilityBatchSampler(minibatch_size=3, min_candidates=3, rng=random.Random(42))

        ids1 = sampler1.next_minibatch_ids(loader, state_with_candidates)
        ids2 = sampler2.next_minibatch_ids(loader, state_with_candidates)
        assert ids1 == ids2

    def test_high_temperature_approaches_uniform(self, state_with_candidates):
        """High temperature should make sampling more uniform."""
        loader = ensure_loader(["ex0", "ex1", "ex2", "ex3", "ex4"])
        counts: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        for seed in range(200):
            sampler = LearnabilityBatchSampler(
                minibatch_size=1, min_candidates=3, temperature=100.0, rng=random.Random(seed)
            )
            ids = sampler.next_minibatch_ids(loader, state_with_candidates)
            counts[ids[0]] += 1

        # With very high temperature, all examples should be sampled somewhat often
        for eid, count in counts.items():
            assert count > 10, f"Example {eid} sampled only {count}/200 times at high temperature"

    def test_low_temperature_concentrates(self, state_with_candidates):
        """Low temperature should concentrate on high-variance examples."""
        loader = ensure_loader(["ex0", "ex1", "ex2", "ex3", "ex4"])
        counts: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        for seed in range(200):
            sampler = LearnabilityBatchSampler(
                minibatch_size=1, min_candidates=3, temperature=0.1, rng=random.Random(seed)
            )
            ids = sampler.next_minibatch_ids(loader, state_with_candidates)
            counts[ids[0]] += 1

        # Example 1 (highest variance) should dominate
        assert counts[1] > counts[0]
        assert counts[1] > counts[2]
        assert counts[1] > counts[4]
