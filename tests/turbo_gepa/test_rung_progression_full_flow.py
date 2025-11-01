"""Test the full rung progression flow simulating orchestrator behavior."""

import pytest

from turbo_gepa.interfaces import Candidate, EvalResult
from turbo_gepa.scheduler import BudgetedScheduler, SchedulerConfig


def simulate_ingest_result(candidate: Candidate, result: EvalResult, promote_objective: str) -> Candidate:
    """Simulate what orchestrator._ingest_result does to candidate metadata."""
    shard_fraction = result.shard_fraction or 0.0
    quality = result.objectives.get(promote_objective, 0.0)

    original_fingerprint = candidate.fingerprint
    meta = dict(candidate.meta)

    # Add _sched_key if not present (line 994-995 in orchestrator.py)
    if "_sched_key" not in meta:
        meta["_sched_key"] = original_fingerprint

    prev_fraction = meta.get("quality_shard_fraction", 0.0)
    prev_quality = meta.get("quality", float("-inf"))

    if shard_fraction > prev_fraction or (shard_fraction == prev_fraction and quality >= prev_quality):
        meta["quality"] = quality
        meta["quality_shard_fraction"] = shard_fraction

    return Candidate(text=candidate.text, meta=meta)


def test_full_orchestrator_flow():
    """
    Simulate the full orchestrator flow:
    1. Seed candidate evaluated on shard 0
    2. Promoted to shard 1
    3. Re-evaluated on shard 1
    4. Completed at final rung
    """
    config = SchedulerConfig(
        shards=[0.3, 1.0],
        eps_improve=0.0,
        quantile=0.6,
        enable_convergence=False,
        lineage_patience=0,
        lineage_min_improve=0.01,
    )
    scheduler = BudgetedScheduler(config)

    # Step 1: Seed candidate (no metadata)
    seed = Candidate(text="seed prompt", meta={})

    # Step 2: Evaluate on shard 0
    result_shard0 = EvalResult(
        objectives={"quality": 1.0, "neg_cost": -0.1},
        traces=[],
        n_examples=3,
        shard_fraction=0.3,
    )

    # Step 3: Simulate _ingest_result (adds _sched_key)
    seed_with_meta = simulate_ingest_result(seed, result_shard0, "quality")
    print(f"After ingest_result: meta={seed_with_meta.meta}")
    print(f"Original fingerprint: {seed.fingerprint}")
    print(f"After ingest fingerprint: {seed_with_meta.fingerprint}")

    # Step 4: Scheduler records the result
    decision1 = scheduler.record(seed_with_meta, result_shard0, "quality")
    print(f"Shard 0 decision: {decision1}")
    assert decision1 == "promoted"

    # Step 5: Get promoted candidates
    promotions = scheduler.promote_ready()
    assert len(promotions) == 1
    promoted = promotions[0]

    print(f"Promoted candidate meta: {promoted.meta}")
    print(f"Promoted candidate fingerprint: {promoted.fingerprint}")
    print(f"Scheduler shard index for promoted: {scheduler.current_shard_index(promoted)}")

    # Step 6: Orchestrator enqueues the promoted candidate (no changes to metadata)
    # Then it gets evaluated on shard 1

    # Step 7: Evaluate on shard 1
    result_shard1 = EvalResult(
        objectives={"quality": 1.0, "neg_cost": -0.1},
        traces=[],
        n_examples=10,
        shard_fraction=1.0,
    )

    # Step 8: Simulate _ingest_result again (should preserve _sched_key)
    promoted_with_meta = simulate_ingest_result(promoted, result_shard1, "quality")
    print(f"After second ingest_result: meta={promoted_with_meta.meta}")
    print(f"After second ingest fingerprint: {promoted_with_meta.fingerprint}")
    print(f"Scheduler shard index before record: {scheduler.current_shard_index(promoted_with_meta)}")

    # Step 9: Scheduler records the result on shard 1
    decision2 = scheduler.record(promoted_with_meta, result_shard1, "quality")
    print(f"Shard 1 decision: {decision2}")
    print(f"Scheduler shard index after record: {scheduler.current_shard_index(promoted_with_meta)}")

    assert decision2 == "completed", f"Expected 'completed', got '{decision2}'"


if __name__ == "__main__":
    test_full_orchestrator_flow()
    print("\n✅ All tests passed!")
