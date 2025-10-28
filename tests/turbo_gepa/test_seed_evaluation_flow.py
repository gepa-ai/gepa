#!/usr/bin/env python3
"""
Test complete seed evaluation flow to reproduce hang.
Run with: python tests/turbo_gepa/test_seed_evaluation_flow.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from turbo_gepa.orchestrator import Orchestrator
from turbo_gepa.config import Config
from turbo_gepa.interfaces import Candidate, EvalResult
from turbo_gepa.scheduler import BudgetedScheduler, SchedulerConfig
from turbo_gepa.archive import Archive
from turbo_gepa.sampler import InstanceSampler
from turbo_gepa.evaluator import AsyncEvaluator
from turbo_gepa.mutator import Mutator
from turbo_gepa.cache import DiskCache
import asyncio


async def test_seed_flow_minimal():
    """Minimal reproduction of seed evaluation → promotion → re-queue flow."""
    print("\n" + "=" * 80)
    print("MINIMAL SEED FLOW TEST")
    print("=" * 80)

    # Create minimal config
    config = Config(
        shards=(0.3, 1.0),
        eval_concurrency=64,
        max_total_inflight=64,
        queue_limit=64,
        batch_size=4,
        mutation_buffer_min=2,
    )

    # Create components
    scheduler = BudgetedScheduler(
        SchedulerConfig(
            shards=config.shards,
            eps_improve=config.eps_improve,
            quantile=config.cohort_quantile,
        )
    )

    # Create seed
    seed = Candidate(text="You are a helpful assistant.", meta={"source": "seed"})

    print(f"\n1. INITIAL STATE")
    print(f"   Seed text: {seed.text}")
    print(f"   Seed fingerprint: {seed.fingerprint[:12]}...")
    print(f"   Seed rung: {scheduler.current_shard_index(seed)}")

    # Simulate shard 0 evaluation
    result_shard0 = EvalResult(
        objectives={"quality": 0.67},
        traces=[],
        n_examples=1,
        shard_fraction=0.3
    )

    print(f"\n2. SHARD 0 EVALUATION")
    print(f"   Result: {result_shard0.objectives['quality']:.0%} quality")
    print(f"   Shard: {result_shard0.shard_fraction}")

    # Record in scheduler
    decision = scheduler.record(seed, result_shard0, "quality")
    print(f"   Decision: {decision}")
    print(f"   Seed rung after record: {scheduler.current_shard_index(seed)}")

    # Get promotions
    promotions = scheduler.promote_ready()
    print(f"\n3. PROMOTIONS")
    print(f"   Promotions returned: {len(promotions)}")

    if promotions:
        promoted_seed = promotions[0]
        print(f"   Promoted seed fingerprint: {promoted_seed.fingerprint[:12]}...")
        print(f"   Same object as original? {promoted_seed is seed}")
        print(f"   Same fingerprint? {promoted_seed.fingerprint == seed.fingerprint}")
        print(f"   Promoted seed rung: {scheduler.current_shard_index(promoted_seed)}")

        # Check if metadata changed
        print(f"\n4. METADATA CHECK")
        print(f"   Original seed meta: {seed.meta}")
        print(f"   Promoted seed meta: {promoted_seed.meta}")

        return promoted_seed, scheduler
    else:
        print("   ❌ NO PROMOTIONS - THIS IS THE BUG!")
        return None, scheduler


async def test_orchestrator_enqueue():
    """Test that promoted seed gets enqueued correctly."""
    print("\n" + "=" * 80)
    print("ORCHESTRATOR ENQUEUE TEST")
    print("=" * 80)

    promoted_seed, scheduler = await test_seed_flow_minimal()
    if not promoted_seed:
        print("❌ Can't test enqueue without promoted seed")
        return

    # Create mock orchestrator components
    config = Config(shards=(0.3, 1.0), eval_concurrency=64)

    # Mock dataset
    dataset = [{"input": "test", "answer": "test"}]

    # Create minimal orchestrator (won't work fully but we can test enqueue)
    class MockOrchestrator:
        def __init__(self, config, scheduler):
            from collections import deque
            self.config = config
            self.scheduler = scheduler
            self.queue = deque(maxlen=config.queue_limit)
            self._per_shard_queue = [deque() for _ in range(len(config.shards))]
            self._pending_fingerprints = set()
            self._inflight_fingerprints = set()

        def enqueue(self, candidates):
            for candidate in candidates:
                if not candidate.text.strip():
                    continue
                fingerprint = candidate.fingerprint
                if fingerprint in self._pending_fingerprints or fingerprint in self._inflight_fingerprints:
                    print(f"      Skipping (already pending/inflight): {fingerprint[:12]}...")
                    continue
                idx = self.scheduler.current_shard_index(candidate)
                shard_idx = min(idx, len(self._per_shard_queue) - 1)
                self._per_shard_queue[shard_idx].append(candidate)
                self.queue.append(candidate)
                self._pending_fingerprints.add(fingerprint)
                print(f"      ✅ Enqueued to shard {shard_idx} (rung {idx})")

    orch = MockOrchestrator(config, scheduler)

    print(f"\n5. ENQUEUE PROMOTED SEED")
    print(f"   Queue size before: {len(orch.queue)}")
    print(f"   Per-shard queues before: {[len(q) for q in orch._per_shard_queue]}")

    orch.enqueue([promoted_seed])

    print(f"   Queue size after: {len(orch.queue)}")
    print(f"   Per-shard queues after: {[len(q) for q in orch._per_shard_queue]}")

    if len(orch.queue) > 0:
        queued_cand = orch.queue[0]
        print(f"   Queued candidate rung: {scheduler.current_shard_index(queued_cand)}")
        print(f"   Expected rung: 1 (shard {config.shards[1]})")

        # Check which shard queue it's in
        for i, q in enumerate(orch._per_shard_queue):
            if queued_cand in q:
                print(f"   ✅ Found in per-shard queue {i}")
                break
        else:
            print(f"   ❌ NOT in any per-shard queue!")
    else:
        print(f"   ❌ Queue is empty!")


async def test_launch_conditions():
    """Test if promoted seed can be launched."""
    print("\n" + "=" * 80)
    print("LAUNCH CONDITIONS TEST")
    print("=" * 80)

    promoted_seed, scheduler = await test_seed_flow_minimal()
    if not promoted_seed:
        print("❌ Can't test launch without promoted seed")
        return

    config = Config(shards=(0.3, 1.0), eval_concurrency=64, max_total_inflight=64)

    # Simulate orchestrator state
    _max_total_inflight = 64
    _shard_capacity = [32, 32]  # Split evenly between 2 shards
    _shard_inflight = [0, 0]
    _total_inflight = 0

    print(f"\n6. LAUNCH CHECK")
    print(f"   max_total_inflight: {_max_total_inflight}")
    print(f"   shard_capacity: {_shard_capacity}")
    print(f"   shard_inflight: {_shard_inflight}")
    print(f"   total_inflight: {_total_inflight}")

    # Check if can launch
    shard_idx = scheduler.current_shard_index(promoted_seed)
    actual_shard_idx = min(shard_idx, len(_shard_capacity) - 1)

    print(f"\n   Promoted seed rung: {shard_idx}")
    print(f"   Actual shard index: {actual_shard_idx}")

    can_launch = (
        _shard_inflight[actual_shard_idx] < _shard_capacity[actual_shard_idx]
        and _total_inflight < _max_total_inflight
    )

    print(f"   Can launch? {can_launch}")

    if can_launch:
        print(f"   ✅ Promoted seed CAN be launched")
    else:
        print(f"   ❌ Promoted seed CANNOT be launched - capacity issue!")


if __name__ == "__main__":
    try:
        asyncio.run(test_seed_flow_minimal())
        asyncio.run(test_orchestrator_enqueue())
        asyncio.run(test_launch_conditions())

        print("\n" + "=" * 80)
        print("DIAGNOSIS COMPLETE")
        print("=" * 80)
        print("\nIf all checks passed, the hang must be in the orchestrator loop logic.")
        print("Next: Check if _stream_launch_ready is being called and why it's not launching.")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
