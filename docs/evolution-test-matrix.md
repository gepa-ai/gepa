# Evolution Regression Matrix

This document captures the real-model scenarios we run to validate that TurboGEPA’s orchestration loop behaves correctly under common failure modes. All scenarios use the approved models:

- `task_lm="openrouter/openai/gpt-oss-20b:nitro"`
- `reflection_lm="openrouter/x-ai/grok-4-fast"`

Set `OPENROUTER_API_KEY` (and any other provider credentials) before running.

## How to Run

```bash
python scripts/run_evolution_matrix.py --all
# or run a subset
python scripts/run_evolution_matrix.py --scenario seed-baseline high-concurrency
```

Each scenario writes its cache/logs under `./regression_runs/<scenario-name>/`.

## Scenario Matrix

| Scenario | Purpose | Key Config |
| --- | --- | --- |
| `seed-baseline` | Pure promotion path (mutations disabled). Surfaces rung tolerances and final-rung throughput. | `max_mutations_per_round=0`, shards `(0.5, 1.0)` |
| `high-concurrency` | Stress queue/inflight coordination when many evaluators target the final rung. | `eval_concurrency=32`, `max_final_shard_inflight=3`, shards `(0.2, 0.6, 1.0)` |
| `mixed-parent-pool` | Ensures rung-0 parents remain in mutation pool until ≥3 higher-rung parents exist. | `max_mutations_per_round=24`, shards `(0.1, 0.6, 1.0)` |
| `neg-cost-objective` | Regression test for alternate `promote_objective` values. | `promote_objective="neg_cost"` |
| `resume-drill` | Stop mid-run and resume from cache to validate persistence hygiene. | Same cache/log paths reused; script runs twice automatically. |
| `multi-island-migration` | Verifies migration quotas and avoids reintroducing queue saturation. | `n_islands=2`, `migration_k=2`, `migration_period=2` |
| `straggler-tail` | Injects slow evaluations to exercise straggler detaching and replay. | Adds 20s delay to ~15% of task-runner calls. |

Each scenario logs a summary line with total run time, Pareto size, and best objective so we can compare across builds. Use the scripts’ per-scenario log directories plus the evolution snapshots produced by the orchestrator for deeper analysis.
