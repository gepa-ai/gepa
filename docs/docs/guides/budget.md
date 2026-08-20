# Choosing `max_metric_calls`

There are two ways to control when a GEPA run stops:

1. **Set `max_metric_calls`** — a hard upper bound on the metric-call budget. Simple, but you have to pick a number upfront.
2. **Skip `max_metric_calls` and pass `stop_callbacks` instead** — let GEPA run until a substantive condition fires (no improvement for N iterations, score threshold reached, wall-clock deadline). You don't have to predict the budget; the run stops when the optimization has actually converged.

You can also do both: GEPA stops as soon as any stopper fires.

## Recommended: use `NoImprovementStopper` instead of a fixed budget

For most real workloads, the right answer is to **not** set `max_metric_calls` and use `NoImprovementStopper` instead:

```python
from gepa.utils import NoImprovementStopper

result = gepa.optimize(
    seed_candidate=...,
    trainset=...,
    valset=...,
    stop_callbacks=[NoImprovementStopper(max_iterations_without_improvement=10)],
    reflection_lm="openai/gpt-5",
)
```

This gives the optimizer license to keep proposing as long as it's making progress, and stops it when it actually stalls. No need to guess the budget.

Other useful stoppers:

- `ScoreThresholdStopper(threshold=...)` — stop once a target validation score is reached.
- `TimeoutStopCondition(timeout_seconds=...)` — wall-clock deadline (useful in CI / agent loops).
- `SignalStopper` / `FileStopper` — graceful external stop signals.
- `CompositeStopper(...)` — combine several.

## When you *do* set `max_metric_calls`

If you'd rather pin the budget explicitly, **set it to `> 15 × len(valset)`**.

That gives GEPA room for ~15 proposal attempts, the recommended minimum for meaningful evolutionary search. Higher is usually better — many production deployments use 200-2000+ total calls.

### The exact formula

Each metric call costs roughly one rollout. GEPA spends them in two places:

| Cost | Per occurrence | Frequency |
|------|----------------|-----------|
| Baseline full validation | `len(valset)` | Once at the start |
| Each proposal cycle | `reflection_minibatch_size + len(valset)` | Per proposal attempt |

To support **N proposal attempts**, you need at least:

```
floor(N) = len(valset) + N * (reflection_minibatch_size + len(valset))
```

With the recommended `N = 15` and the default `reflection_minibatch_size = 3`, this simplifies (when `len(valset) >> minibatch`) to roughly `> 15 × len(valset)`.

### Recommended floors

| `len(valset)` | minibatch | **Recommended (N=15)** | Generous (N=30) |
|---:|---:|---:|---:|
| 6 | 3 | **141** | 276 |
| 10 | 3 | **205** | 400 |
| 20 | 3 | **365** | 710 |
| 50 | 3 | **815** | 1,610 |
| 100 | 3 | **1,645** | 3,200 |
| 200 | 5 | **3,275** | 6,350 |

`gepa.optimize` uses the N=15 column as the threshold for the pre-flight warning.

## The bug this prevents

This guide exists because of a real incident ([#375](https://github.com/gepa-ai/gepa/issues/375)). A user ran:

```python
gepa.optimize(
    ...,
    valset=valset,           # 6 examples
    max_metric_calls=8,      # ← only enough for baseline + 0 proposals
)
```

With `len(valset) = 6` and the default minibatch of 3, the baseline full-validation alone burned 6 of the 8 budgeted calls, leaving budget for less than one proposal cycle. The agent driving the run reported the baseline-improved candidate as a successful optimization, when in fact GEPA had done one proposal step and stopped.

The recommended budget for that configuration is `> 15 × 6 = 90` (or precisely `141` if you compute exactly), or — better — use `NoImprovementStopper` and skip the budget entirely.

## How to read the warnings

**At the start of the run (pre-flight, only when `max_metric_calls` is set below the floor):**

```
GEPABudgetWarning: max_metric_calls=8 is below the recommended floor of 141
for valset of size 6 and reflection_minibatch_size=3. At this budget GEPA can
attempt at most ~0 proposal step(s), which typically produces a short trajectory
rather than real optimization. Recommended: set max_metric_calls to at least 141
(gives ~15 proposal attempts). Rule of thumb: ~16 x len(valset).
```

**At the end of the run (post-run summary + warning):**

```
GEPA finished: 1 proposal(s) accepted over 8 metric call(s); final candidate pool size = 2.
GEPABudgetWarning: GEPA finished with only 1 accepted proposal(s) ...
```

A pool size of 2 means baseline + 1 candidate — the optimizer didn't get to explore. If you were using a non-`max_metric_calls` stopper, the warning means *that* stopper fired too early: loosen `NoImprovementStopper.max_iterations_without_improvement`, raise `ScoreThresholdStopper.threshold`, or extend the timeout.

## Silencing the warning

For deliberate small-budget runs (smoke tests, integration checks, deterministic CI), suppress with the standard `warnings` filter:

```python
import warnings
import gepa

warnings.filterwarnings("ignore", category=gepa.GEPABudgetWarning)
```

Or temporarily:

```python
with warnings.catch_warnings():
    warnings.simplefilter("ignore", gepa.GEPABudgetWarning)
    gepa.optimize(...)
```

## Inspecting a finished run

```python
result = gepa.optimize(...)
print(f"Accepted candidates: {result.num_candidates - 1}")   # excludes baseline
print(f"Total metric calls: {result.total_metric_calls}")
print(f"Best val score:     {result.val_aggregate_scores[result.best_idx]}")
```

If `num_candidates - 1` is much smaller than what you expected, the run was under-budgeted or your stopper fired too early — regardless of whether the warning fired (e.g., the optimizer accepted very few proposals despite a generous budget — a sign your evaluator's feedback signal is too weak).

## Where to set the budget for autonomous coding agents

See the dedicated guide: [Running GEPA inside autonomous coding agents](coding-agents.md).
