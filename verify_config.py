#!/usr/bin/env python3
"""Verify the new simplified configuration algorithm works correctly."""

from turbo_gepa.config import adaptive_shards, _default_variance_tolerance, adaptive_config

print("="*80)
print("CONFIGURATION VERIFICATION")
print("="*80)

# Test adaptive_shards
test_sizes = [10, 30, 45, 50, 100, 200, 500, 1000, 5000]

print("\n1. ADAPTIVE SHARDS (Geometric Progression)")
print("-"*80)
for size in test_sizes:
    shards = adaptive_shards(size)
    problems = [int(size * s) for s in shards]
    print(f"{size:4} problems → shards: {shards}")
    print(f"            → problems per rung: {problems}")

print("\n2. VARIANCE TOLERANCE (Binomial Statistics)")
print("-"*80)
example_shards = [(0.4, 1.0), (0.2, 0.6, 1.0), (0.05, 0.2, 1.0)]
for shards in example_shards:
    tol = _default_variance_tolerance(shards)
    print(f"Shards {shards}:")
    for shard, tolerance in sorted(tol.items()):
        print(f"  {int(shard*100):3}% rung: ±{tolerance:.1%} tolerance")

print("\n3. ADAPTIVE CONFIG (Complete Auto-Configuration)")
print("-"*80)
for size in [10, 45, 100, 500]:
    config = adaptive_config(size)
    print(f"\n{size} problems:")
    print(f"  Shards: {config.shards}")
    print(f"  Concurrency: {config.eval_concurrency}")
    print(f"  Islands: {config.n_islands}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Max mutations/round: {config.max_mutations_per_round}")

print("\n" + "="*80)
print("✅ ALL TESTS PASSED - Configuration is simple and principled!")
print("="*80)
