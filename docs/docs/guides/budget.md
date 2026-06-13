# Choosing `max_metric_calls`

`max_metric_calls` is the **only** knob that controls how many proposal iterations GEPA gets to attempt. There is no separate `iterations` flag. If you set it too low, GEPA will produce a 1- or 2-point trajectory (baseline → one accepted candidate) that looks like optimization but isn't.

GEPA 0.x will now warn you when this happens — both before the run (if the budget is below the recommended floor) and after the run (if too few candidates were accepted). This guide explains how to size it correctly.

## The budget formula

Each metric call costs roughly one rollout. GEPA spends them in two places:

| Cost | Per occurrence | Frequency |
|------|----------------|-----------|
| Baseline full validation | `len(valset)` | Once at the start |
| Each proposal cycle | `reflection_minibatch_size + len(valset)` | Per proposal attempt |

So to support **N proposal attempts**, you need at least:

```
floor(N) = len(valset) + N * (reflection_minibatch_size + len(valset))
```

`reflection_minibatch_size` defaults to **3**.

## Recommended floors

`gepa.optimize` uses `N = 3` as the recommended-minimum threshold. Below that, the warning fires. Here are concrete numbers:

| `len(valset)` | minibatch | Floor (N=3) | Floor for "real" runs (N=10) | Floor for production (N=30) |
|---:|---:|---:|---:|---:|
| 6 | 3 | **33** | 96 | 276 |
| 20 | 3 | **89** | 250 | 710 |
| 50 | 3 | **209** | 580 | 1640 |
| 100 | 3 | **409** | 1130 | 3190 |
| 200 | 5 | **815** | 2250 | 6350 |

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

The floor formula above would have warned: `floor(3) = 6 + 3 * (3 + 6) = 33`.

## How to read the warnings

**At the start of the run:**

```
GEPABudgetWarning: max_metric_calls=8 is below the recommended floor of 33
for valset of size 6 and reflection_minibatch_size=3. At this budget GEPA can
attempt at most ~0 proposal step(s), which typically produces a 2-point
trajectory rather than real optimization. Recommended: set max_metric_calls
to at least 33 (gives ≥3 proposal attempts).
```

**At the end of the run:**

```
GEPA finished: 1 proposal(s) accepted over 8 metric call(s); final candidate pool size = 2.
GEPABudgetWarning: GEPA finished with only 1 accepted proposal(s) ...
```

A pool size of 2 means baseline + 1 candidate — the optimizer didn't get to explore.

## Silencing the warning

If you have a deliberate reason to run with a small budget (smoke tests, integration checks, deterministic CI), suppress it with the standard `warnings` filter:

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

If `num_candidates - 1` is much smaller than what you expected, the run was under-budgeted regardless of whether the warning fired (e.g., the optimizer accepted very few proposals despite a generous budget — a sign your evaluator's feedback signal is too weak).

## Where to set the budget for autonomous coding agents

See the dedicated guide: [Running GEPA inside autonomous coding agents](coding-agents.md).
