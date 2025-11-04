#!/usr/bin/env python3
"""Debug why promotions are failing."""

from turbo_gepa.config import _default_variance_tolerance, _default_shrinkage_alpha

# Test case: parent scores 100% at final rung, child scores 100% at first rung
shards = (0.2, 0.5, 1.0)
tolerance = _default_variance_tolerance(shards)
alpha = _default_shrinkage_alpha(shards)

print('Shards:', shards)
print('\nVariance tolerance:', tolerance)
print('Shrinkage alpha:', alpha)

# Simulate: Parent scored 100% on full dataset
# Child scores 100% on 20% rung (1 problem)
parent_final = 1.0  # 100%
rung_fraction = 0.2
alpha_val = alpha[rung_fraction]
global_baseline = 0.5

parent_at_rung = (1 - alpha_val) * global_baseline + alpha_val * parent_final
tolerance_val = tolerance[rung_fraction]

child_score = 1.0  # 100% on 1 problem

print(f'\n--- Promotion Decision (child = 100%) ---')
print(f'Parent final score: {parent_final:.1%}')
print(f'Estimated parent @ rung {rung_fraction:.1%}: {parent_at_rung:.1%} (shrinkage α={alpha_val:.2f})')
print(f'Child score @ rung {rung_fraction:.1%}: {child_score:.1%}')
print(f'Tolerance: ±{tolerance_val:.1%}')
print(f'Threshold: {parent_at_rung - tolerance_val:.1%}')
print(f'Promote if: {child_score:.1%} >= {parent_at_rung - tolerance_val:.1%}')
result = 'PROMOTE' if child_score >= parent_at_rung - tolerance_val else 'PRUNE'
print(f'Result: {result}')

# What if child scores 0%?
child_score_fail = 0.0
print(f'\n--- What if child fails (0%)? ---')
print(f'Child score: {child_score_fail:.1%}')
print(f'Promote if: {child_score_fail:.1%} >= {parent_at_rung - tolerance_val:.1%}')
result2 = 'PROMOTE' if child_score_fail >= parent_at_rung - tolerance_val else 'PRUNE'
print(f'Result: {result2}')

# THE BUG: Check what if parent didn't reach final rung yet?
print(f'\n--- BUG SCENARIO: Parent never evaluated at final rung ---')
print(f'If parent was pruned before reaching final rung:')
print(f'  - parent_sched_key exists in candidate.meta')
print(f'  - BUT parent never got final score')
print(f'  - parent_objectives contains PARTIAL score from earlier rung')
print(f'  - Shrinkage uses PARTIAL score as if it were final score')
print(f'  - This creates INVALID comparison baseline!')

# Test with parent at 60% rung scoring 100%
print(f'\n--- Real scenario: Parent at 60% rung ---')
parent_at_60 = 1.0  # Parent scored 100% on 60% of data
# Shrinkage estimates this parent's score at 20% rung
parent_est_at_20 = (1 - alpha_val) * global_baseline + alpha_val * parent_at_60
print(f'Parent score @ 60% rung: {parent_at_60:.1%}')
print(f'Estimated parent @ 20% rung: {parent_est_at_20:.1%}')
print(f'Child must score >= {parent_est_at_20 - tolerance_val:.1%} to promote')
print(f'If child scores 100%: {child_score:.1%} >= {parent_est_at_20 - tolerance_val:.1%} = {"PROMOTE" if child_score >= parent_est_at_20 - tolerance_val else "PRUNE"}')
