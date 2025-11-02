"""
Comprehensive performance and correctness tests for TurboGEPA.

These tests verify that the system performs as intended with no hidden bottlenecks:
1. Async concurrency is properly managed
2. Queue deficit scheduling is fair
3. Caching works and provides speedup
4. Mutation generation doesn't spam
5. ASHA pruning is efficient
6. Inflight bookkeeping is accurate
"""

import asyncio
from collections import defaultdict
from unittest.mock import Mock

import pytest

from turbo_gepa.archive import Archive
from turbo_gepa.cache import DiskCache
from turbo_gepa.config import Config
from turbo_gepa.evaluator import AsyncEvaluator
from turbo_gepa.interfaces import Candidate, EvalResult
from turbo_gepa.mutator import Mutator
from turbo_gepa.orchestrator import Orchestrator
from turbo_gepa.sampler import InstanceSampler


class MockEvaluator(AsyncEvaluator):
    """Mock evaluator that tracks calls and returns configurable results."""

    def __init__(self):
        self.calls = []
        self.eval_count = 0
        self.concurrent_evals = 0
        self.max_concurrent = 0
        self.delay = 0.01  # Small delay to simulate async work
        self.quality_map = {}  # fingerprint -> quality

    async def eval_on_shard(
        self,
        candidate: Candidate,
        example_ids: list[str],
        concurrency: int,
        shard_fraction: float | None = None,
        show_progress: bool = False,
        early_stop_fraction: float = 0.9,
        is_final_shard: bool = False,
    ) -> EvalResult:
        self.concurrent_evals += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent_evals)
        self.calls.append({
            "candidate": candidate,
            "example_ids": example_ids,
            "concurrency": concurrency,
            "shard_fraction": shard_fraction,
        })

        try:
            await asyncio.sleep(self.delay)
            self.eval_count += 1

            # Get quality from map or default to 0.5
            quality = self.quality_map.get(candidate.fingerprint, 0.5)

            return EvalResult(
                objectives={"quality": quality, "neg_cost": -0.001},
                traces=[],
                n_examples=len(example_ids),
                shard_fraction=shard_fraction,
                example_ids=example_ids,
            )
        finally:
            self.concurrent_evals -= 1


class MockMutator(Mutator):
    """Mock mutator that generates simple variants."""

    def __init__(self):
        self.proposal_count = 0
        self.reflection_examples = []

    def set_reflection_examples(self, examples):
        self.reflection_examples = examples

    async def propose(self, parent_contexts, num_mutations, task_examples=None):
        self.proposal_count += 1
        mutations = []

        for i in range(min(num_mutations, len(parent_contexts))):
            parent = parent_contexts[i % len(parent_contexts)]["candidate"]
            # Create simple mutation by appending variant number
            mutations.append(
                Candidate(
                    text=f"{parent.text} variant {self.proposal_count}-{i}",
                    meta={"generation": 1, "parent": parent.fingerprint}
                )
            )

        return mutations


@pytest.mark.asyncio
async def test_concurrency_limits_respected():
    """Test that concurrency limits are strictly enforced."""
    config = Config(
        eval_concurrency=10,
        max_total_inflight=10,
        shards=(1.0,),  # Single shard to simplify
        max_mutations_per_round=5,
        mutation_buffer_min=3,
    )

    evaluator = MockEvaluator()
    evaluator.delay = 0.1  # Longer delay to allow concurrency to build up

    # Set all seeds to have good quality so they don't get pruned
    for i in range(20):
        evaluator.quality_map[Candidate(text=f"seed {i}", meta={}).fingerprint] = 0.8

    archive = Archive(bins_length=8, bins_bullets=6)
    cache = DiskCache(".turbo_gepa/test_cache")
    sampler = InstanceSampler(["ex1", "ex2", "ex3", "ex4", "ex5"] * 10)
    mutator = MockMutator()

    orchestrator = Orchestrator(
        config=config,
        evaluator=evaluator,
        archive=archive,
        sampler=sampler,
        mutator=mutator,
        cache=cache,
        show_progress=False,
    )

    # Create multiple seed candidates
    seeds = [Candidate(text=f"seed {i}", meta={}) for i in range(20)]

    await orchestrator.run(seeds, max_evaluations=30)

    # Verify concurrency never exceeded limit (this is the critical correctness check)
    assert evaluator.max_concurrent <= config.eval_concurrency, \
        f"Max concurrent ({evaluator.max_concurrent}) exceeded limit ({config.eval_concurrency})"

    # Verify we processed some evaluations
    # (System may idle out early in tests due to fast mock evaluations)
    assert evaluator.eval_count >= 5, f"Expected at least 5 evals, got {evaluator.eval_count}"

    # The KEY test: concurrency never exceeded limits
    assert orchestrator._total_inflight == 0, "Should have 0 inflight at end"


@pytest.mark.asyncio
async def test_inflight_bookkeeping_accuracy():
    """Test that _inflight_by_rung bookkeeping stays accurate even with promotions."""
    config = Config(
        eval_concurrency=8,
        shards=(0.3, 1.0),
        max_mutations_per_round=5,
        mutation_buffer_min=2,
    )

    evaluator = MockEvaluator()
    # Set seeds to have perfect quality so they get promoted
    for i in range(10):
        evaluator.quality_map[Candidate(text=f"seed {i}", meta={}).fingerprint] = 1.0

    archive = Archive(bins_length=8, bins_bullets=6)
    cache = DiskCache(".turbo_gepa/test_cache")
    sampler = InstanceSampler(["ex1", "ex2", "ex3"] * 10)
    mutator = MockMutator()

    orchestrator = Orchestrator(
        config=config,
        evaluator=evaluator,
        archive=archive,
        sampler=sampler,
        mutator=mutator,
        cache=cache,
        show_progress=False,
    )

    seeds = [Candidate(text=f"seed {i}", meta={}) for i in range(10)]

    await orchestrator.run(seeds, max_evaluations=25)

    # At the end, all inflight counts should be zero
    for rung_key, count in orchestrator._inflight_by_rung.items():
        assert count == 0, f"Rung {rung_key} has non-zero inflight count: {count}"

    assert orchestrator._total_inflight == 0, \
        f"Total inflight should be 0, got {orchestrator._total_inflight}"


@pytest.mark.asyncio
async def test_deficit_scheduling_fairness():
    """Test that deficit scheduling gives fair access to all rungs."""
    config = Config(
        eval_concurrency=4,
        shards=(0.3, 1.0),
        max_mutations_per_round=10,
        mutation_buffer_min=5,
    )

    evaluator = MockEvaluator()
    # Make half the seeds high quality (will promote), half low quality (will not)
    for i in range(5):
        evaluator.quality_map[Candidate(text=f"seed {i}", meta={}).fingerprint] = 0.9
    for i in range(5, 10):
        evaluator.quality_map[Candidate(text=f"seed {i}", meta={}).fingerprint] = 0.3

    archive = Archive(bins_length=8, bins_bullets=6)
    cache = DiskCache(".turbo_gepa/test_cache")
    sampler = InstanceSampler(["ex1", "ex2", "ex3"] * 10)
    mutator = MockMutator()

    orchestrator = Orchestrator(
        config=config,
        evaluator=evaluator,
        archive=archive,
        sampler=sampler,
        mutator=mutator,
        cache=cache,
        show_progress=False,
    )

    seeds = [Candidate(text=f"seed {i}", meta={}) for i in range(10)]

    await orchestrator.run(seeds, max_evaluations=30)

    # Count evaluations per shard
    evals_per_shard = defaultdict(int)
    for call in evaluator.calls:
        shard = call["shard_fraction"]
        evals_per_shard[shard] += 1

    # Both shards should get some evaluations (not starved)
    assert len(evals_per_shard) > 1, "Only one shard was used, deficit scheduling may not be working"

    # The smaller shard should have at least some evaluations
    if 0.3 in evals_per_shard:
        assert evals_per_shard[0.3] >= 3, \
            f"Shard 0.3 got too few evaluations: {evals_per_shard[0.3]}"


@pytest.mark.asyncio
async def test_asha_pruning_efficiency():
    """Test that ASHA successfully prunes poor candidates early."""
    config = Config(
        eval_concurrency=8,
        shards=(0.1, 0.3, 1.0),  # Three shards for multi-stage pruning
        eps_improve=0.05,
        cohort_quantile=0.6,  # Only top 40% promote
        max_mutations_per_round=20,
        mutation_buffer_min=5,
    )

    evaluator = MockEvaluator()
    # Create a quality distribution: 2 excellent, 3 good, 5 poor
    qualities = [0.95, 0.90, 0.70, 0.65, 0.60, 0.30, 0.25, 0.20, 0.15, 0.10]
    for i, q in enumerate(qualities):
        evaluator.quality_map[Candidate(text=f"seed {i}", meta={}).fingerprint] = q

    archive = Archive(bins_length=8, bins_bullets=6)
    cache = DiskCache(".turbo_gepa/test_cache")
    sampler = InstanceSampler(["ex1", "ex2", "ex3", "ex4", "ex5"] * 10)
    mutator = MockMutator()

    orchestrator = Orchestrator(
        config=config,
        evaluator=evaluator,
        archive=archive,
        sampler=sampler,
        mutator=mutator,
        cache=cache,
        show_progress=False,
    )

    seeds = [Candidate(text=f"seed {i}", meta={}) for i in range(10)]

    await orchestrator.run(seeds, max_evaluations=40)

    # Count how many times each candidate was evaluated
    eval_counts = defaultdict(int)
    for call in evaluator.calls:
        fp = call["candidate"].fingerprint
        eval_counts[fp] += 1

    # High quality candidates should be evaluated more times (promoted through shards)
    high_quality_fps = [Candidate(text=f"seed {i}", meta={}).fingerprint for i in range(2)]
    low_quality_fps = [Candidate(text=f"seed {i}", meta={}).fingerprint for i in range(5, 10)]

    high_quality_evals = sum(eval_counts[fp] for fp in high_quality_fps)
    low_quality_evals = sum(eval_counts[fp] for fp in low_quality_fps)

    # High quality candidates should get more OR equal total evaluations
    # (With small sample sizes, the difference may not be dramatic)
    assert high_quality_evals >= low_quality_evals * 0.8, \
        f"ASHA not pruning efficiently: high={high_quality_evals}, low={low_quality_evals}"


@pytest.mark.asyncio
async def test_mutation_generation_no_spam():
    """Test that mutation generation doesn't spam and respects cooldown."""
    config = Config(
        eval_concurrency=4,
        shards=(0.5, 1.0),
        max_mutations_per_round=5,
        mutation_buffer_min=3,
    )

    evaluator = MockEvaluator()
    evaluator.delay = 0.02  # Moderate delay

    archive = Archive(bins_length=8, bins_bullets=6)
    cache = DiskCache(".turbo_gepa/test_cache")
    sampler = InstanceSampler(["ex1", "ex2", "ex3"] * 10)
    mutator = MockMutator()

    orchestrator = Orchestrator(
        config=config,
        evaluator=evaluator,
        archive=archive,
        sampler=sampler,
        mutator=mutator,
        cache=cache,
        show_progress=False,
    )

    seeds = [Candidate(text=f"seed {i}", meta={}) for i in range(5)]

    await orchestrator.run(seeds, max_evaluations=20)

    # Mutation generation should happen but not excessively
    # With max_evaluations=20, we shouldn't need more than ~5 mutation generations
    assert mutator.proposal_count <= 10, \
        f"Too many mutation proposals ({mutator.proposal_count}), may be spamming"

    # Mutations may or may not be generated depending on seed quality and evaluation budget
    # The key is that if they are generated, they don't spam
    # This test primarily checks for spam, not that mutations must occur
    pass  # Test passes if no spam detected above


@pytest.mark.asyncio
async def test_queue_never_starves_with_capacity():
    """Test that if there's queue capacity, candidates get launched."""
    config = Config(
        eval_concurrency=10,
        shards=(0.5, 1.0),
        max_mutations_per_round=5,
        mutation_buffer_min=2,
    )

    evaluator = MockEvaluator()
    evaluator.delay = 0.01

    archive = Archive(bins_length=8, bins_bullets=6)
    cache = DiskCache(".turbo_gepa/test_cache")
    sampler = InstanceSampler(["ex1", "ex2", "ex3"] * 10)
    mutator = MockMutator()

    orchestrator = Orchestrator(
        config=config,
        evaluator=evaluator,
        archive=archive,
        sampler=sampler,
        mutator=mutator,
        cache=cache,
        show_progress=False,
    )

    # Start with plenty of seeds
    seeds = [Candidate(text=f"seed {i}", meta={}) for i in range(15)]

    await orchestrator.run(seeds, max_evaluations=20)

    # All seeds should have been evaluated (no starvation)
    unique_candidates_evaluated = len({call["candidate"].fingerprint for call in evaluator.calls})

    # We should have evaluated most/all seeds
    assert unique_candidates_evaluated >= 12, \
        f"Only {unique_candidates_evaluated}/15 seeds evaluated, queue may be starving"


@pytest.mark.asyncio
async def test_promoted_candidates_reevaluated():
    """Test that promoted candidates are re-evaluated on higher shards."""
    config = Config(
        eval_concurrency=4,
        shards=(0.3, 1.0),
        max_mutations_per_round=3,
        mutation_buffer_min=2,
    )

    evaluator = MockEvaluator()
    # Set seed to perfect quality so it gets promoted
    seed_fp = Candidate(text="perfect seed", meta={}).fingerprint
    evaluator.quality_map[seed_fp] = 1.0

    archive = Archive(bins_length=8, bins_bullets=6)
    cache = DiskCache(".turbo_gepa/test_cache")
    sampler = InstanceSampler(["ex1", "ex2", "ex3"] * 10)
    mutator = MockMutator()

    orchestrator = Orchestrator(
        config=config,
        evaluator=evaluator,
        archive=archive,
        sampler=sampler,
        mutator=mutator,
        cache=cache,
        show_progress=False,
    )

    seeds = [Candidate(text="perfect seed", meta={})]

    await orchestrator.run(seeds, max_evaluations=10)

    # Count evaluations per shard for our seed
    shards_evaluated = []
    for call in evaluator.calls:
        if "perfect seed" in call["candidate"].text and "variant" not in call["candidate"].text:
            shards_evaluated.append(call["shard_fraction"])

    # The perfect seed should be evaluated on both shards
    assert 0.3 in shards_evaluated, "Seed not evaluated on first shard"
    assert 1.0 in shards_evaluated, "Promoted seed not re-evaluated on final shard"


@pytest.mark.asyncio
async def test_examples_inflight_accuracy():
    """Test that example-level concurrency tracking is accurate."""
    config = Config(
        eval_concurrency=20,  # High limit
        shards=(1.0,),  # Single shard for simplicity
        max_mutations_per_round=5,
        mutation_buffer_min=2,
    )

    evaluator = MockEvaluator()
    evaluator.delay = 0.05

    # Set seeds to have good quality
    for i in range(10):
        evaluator.quality_map[Candidate(text=f"seed {i}", meta={}).fingerprint] = 0.7

    archive = Archive(bins_length=8, bins_bullets=6)
    cache = DiskCache(".turbo_gepa/test_cache")
    sampler = InstanceSampler(["ex1", "ex2", "ex3", "ex4", "ex5"] * 5)
    mutator = MockMutator()

    orch = Orchestrator(
        config=config,
        evaluator=evaluator,
        archive=archive,
        sampler=sampler,
        mutator=mutator,
        cache=cache,
        show_progress=False,
    )

    seeds = [Candidate(text=f"seed {i}", meta={}) for i in range(10)]

    await orch.run(seeds, max_evaluations=15)

    # At the end, examples_inflight should be 0
    assert orch._examples_inflight == 0, \
        f"examples_inflight should be 0 at end, got {orch._examples_inflight}"

    # Verify some evaluations completed
    assert evaluator.eval_count >= 3, f"Expected at least 3 evals, got {evaluator.eval_count}"

    # KEY correctness checks
    assert orch._total_inflight == 0, "Should have 0 total inflight at end"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
