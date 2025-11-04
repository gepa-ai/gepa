#!/usr/bin/env python3
"""Test the new adaptive shards algorithm."""

from turbo_gepa.config import adaptive_shards, _default_variance_tolerance

print('=== ADAPTIVE SHARDS ALGORITHM ===\n')
print('Formula: first_rung = min_samples / dataset_size')
print('         subsequent_rungs = previous * reduction_factor')
print('         (min_samples=20, reduction_factor=3.0)\n')
print('Tolerance: 0.75 / sqrt(shard) + 0.02\n')
print('='*80)

test_sizes = [30, 50, 100, 200, 500, 1000, 5000]
for size in test_sizes:
    shards = adaptive_shards(size)
    problems_per_rung = [int(size * s) for s in shards]
    tolerance = _default_variance_tolerance(shards)

    print(f'\nDataset: {size:4} examples')
    print(f'  Shards: {shards}')
    print(f'  Problems per rung: {problems_per_rung}')
    tol_str = ', '.join(f"{int(k*100)}%: ±{v:.1%}" for k, v in sorted(tolerance.items()))
    print(f'  Tolerances: {tol_str}')
