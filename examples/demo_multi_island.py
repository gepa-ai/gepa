#!/usr/bin/env python3
"""
Multi-Island Parallelism Demo

This example demonstrates TurboGEPA's island-based parallel optimization,
where multiple independent populations evolve in parallel and periodically
exchange their best candidates.

Island Model Benefits:
1. True parallelism across CPU cores (multiprocessing)
2. Diverse exploration (each island evolves independently)
3. Periodic migration shares discoveries between islands
4. Near-linear speedup with number of islands

Architecture:
- Each island runs in its own process
- Islands connected in a ring topology
- Every N rounds, top-K candidates migrate to next island
- Final result merges Pareto frontiers from all islands

Usage:
    python examples/demo_multi_island.py

Expected output:
    - Shows 2-4 islands running in parallel
    - Displays migration events
    - Demonstrates speedup vs single island
"""

import time

import gepa
from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config

print("=" * 80)
print("MULTI-ISLAND PARALLELISM DEMO")
print("=" * 80)
print()

# Configure LLMs
task_lm = "openrouter/openai/gpt-oss-20b:nitro"
reflection_lm = "openrouter/x-ai/grok-4-fast"

# Load dataset
print("📊 Loading AIME dataset...")
trainset, _, _ = gepa.examples.aime.init_dataset()
DATASET_SIZE = 10  # Moderate size for island demo
dataset = [
    DefaultDataInst(
        input=ex["input"],
        answer=ex["answer"],
        id=f"aime_{i}",
        additional_context=ex.get("additional_context"),
    )
    for i, ex in enumerate(trainset[:DATASET_SIZE])
]
print(f"✓ Loaded {len(dataset)} problems")
print()

# Configure multi-island optimization
import os
cpu_count = os.cpu_count() or 4
n_islands = min(4, max(2, cpu_count // 2))  # Use 2-4 islands

print(f"⚙️  Configuring {n_islands}-island parallel optimization...")
print()

config = Config(
    shards=(1.0,),  # Single shard for simplicity
    eval_concurrency=12,  # Moderate concurrency per island
    max_mutations_per_round=8,
    mutation_buffer_min=4,
    queue_limit=128,
    batch_size=10,
    n_islands=n_islands,  # Multiple islands!
)

# Migration settings
config.migration_period = 2  # Migrate every 2 rounds
config.migration_k = 2  # Send top 2 candidates
config.cohort_quantile = 0.5

print("📋 Island Configuration:")
print(f"   Number of islands: {n_islands}")
print(f"   Concurrency per island: {config.eval_concurrency}")
print(f"   Total parallel capacity: {n_islands * config.eval_concurrency}")
print()
print("🌉 Migration Configuration:")
print(f"   Migration period: every {config.migration_period} rounds")
print(f"   Candidates per migration: {config.migration_k}")
print(f"   Topology: Ring (Island i → Island (i+1) % {n_islands})")
print()
print("💡 What to watch for:")
print(f"   - {n_islands} processes running in parallel")
print("   - Islands evolving independently between migrations")
print("   - Best candidates migrating between islands")
print(f"   - Expected speedup: ~{n_islands * 0.7:.1f}x vs single island")
print()

# Create adapter
adapter = DefaultAdapter(
    dataset=dataset,
    task_lm=task_lm,
    reflection_lm=reflection_lm,
    config=config,
    auto_config=False,
)

seed = "You are a helpful assistant. Answer in the format '### <final answer>'"
print("🚀 Starting multi-island optimization...")
print(f"   Seed: {seed[:50]}...")
print()
print("=" * 80)
print()
print(f"📺 Progress from {n_islands} islands will be displayed...")
print("   (Note: Progress bars show aggregated stats across all islands)")
print()

start_time = time.time()
result = adapter.optimize(
    seeds=[seed],
    enable_auto_stop=False,
    max_rounds=6,
    display_progress=True,
    optimize_temperature_after_convergence=False,
)
elapsed = time.time() - start_time

print()
print("=" * 80)
print("MULTI-ISLAND ANALYSIS")
print("=" * 80)
print()

# Analyze results
pareto = result.get("pareto_entries", [])
stats = result.get("evolution_stats", {})

if pareto:
    best = max(pareto, key=lambda e: e.result.objectives.get("quality", 0.0))
    quality = best.result.objectives.get("quality", 0.0)

    print("✅ Results:")
    print(f"   Best quality: {quality:.1%}")
    print(f"   Pareto size: {len(pareto)} (merged from all islands)")
    print(f"   Total evaluations: {stats.get('total_evaluations', 0)}")
    print(f"   Time: {elapsed:.1f}s")
    print(f"   Throughput: {stats.get('total_evaluations', 0) / elapsed:.1f} evals/sec")
    print()

    # Island-specific stats (if available)
    island_stats = result.get("island_stats", {})
    if island_stats:
        print("📊 Per-Island Statistics:")
        for island_id, istats in sorted(island_stats.items()):
            evals = istats.get("evaluations", 0)
            mutations = istats.get("mutations_generated", 0)
            print(f"   Island {island_id}: {evals} evals, {mutations} mutations")
        print()

    print("🌉 Migration Impact:")
    migrations_sent = stats.get("migrations_sent", 0)
    migrations_received = stats.get("migrations_received", 0)
    print(f"   Total migrations: {migrations_sent} sent, {migrations_received} received")
    print(f"   Migrations per island: ~{migrations_sent / n_islands if n_islands > 0 else 0:.1f}")
    print()

    print("📝 Best prompt:")
    print("-" * 80)
    print(best.candidate.text[:200] + "..." if len(best.candidate.text) > 200 else best.candidate.text)
    print("-" * 80)
else:
    print("❌ No results generated")

print()
print("=" * 80)
print("🎓 Key Takeaways:")
print(f"   ✓ {n_islands} islands evolved populations in parallel")
print("   ✓ True multiprocessing across CPU cores")
print("   ✓ Ring topology shares discoveries between islands")
print("   ✓ Merged Pareto frontier from all islands")
print()
print("💡 Tuning Tips:")
print("   - More islands = more parallelism, but migration overhead")
print("   - migration_period: 2-4 rounds is typical")
print("   - migration_k: 1-3 candidates usually sufficient")
print("   - Sweet spot: 2-4 islands for most systems")
print()
print("⚡ Performance Notes:")
print(f"   - Used {n_islands} processes (vs 1 for single-island)")
print(f"   - Total parallel capacity: {n_islands * config.eval_concurrency} concurrent evals")
print("   - Speedup depends on LLM latency (higher latency = better speedup)")
print("=" * 80)
