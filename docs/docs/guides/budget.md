# Choosing `max_metric_calls`

`max_metric_calls` is the **ceiling** on the metric-call budget for a GEPA run. Other stop conditions (`NoImprovementStopper`, `TimeoutStopCondition`, `ScoreThresholdStopper`, …) can stop the run earlier when their condition fires, but none of them give GEPA *more* iterations than `max_metric_calls` allows. So this single value is the upper bound on optimization depth — setting it too low caps the optimizer regardless of any other stoppers you configure.

If you set it too low, GEPA will produce a 1- or 2-point trajectory (baseline → one accepted candidate) that looks like optimization but isn't. GEPA will warn you when this happens — both before the run (if the budget is below the recommended floor) and after the run (if too few candidates were accepted). This guide explains how to size it correctly.

## Rule of thumb

**Set `max_metric_calls` to more than `15 × len(valset)`.**

That gives GEPA room for ~15 proposal attempts, which is the recommended minimum for meaningful evolutionary search. For real workloads, **higher is usually better** — many production deployments use 200-2000+ total calls. Pick the largest budget you can afford in wall-clock time.

## The exact formula

Each metric call costs roughly one rollout. GEPA spends them in two places:

| Cost | Per occurrence | Frequency |
|------|----------------|-----------|
| Baseline full validation | `len(valset)` | Once at the start |
| Each proposal cycle | `reflection_minibatch_size + len(valset)` | Per proposal attempt |

To support **N proposal attempts**, you need at least:

```
floor(N) = len(valset) + N * (reflection_minibatch_size + len(valset))
```

With the recommended `N = 15` and the default `reflection_minibatch_size = 3`, this simplifies (when `len(valset) >> minibatch`) to roughly `> 15 × len(valset)` — the rule of thumb above.

## Recommended floors

| `len(valset)` | minibatch | **Recommended (N=15)** | Generous (N=30) |
|---:|---:|---:|---:|
| 6 | 3 | **141** | 276 |
| 10 | 3 | **205** | 400 |
| 20 | 3 | **365** | 710 |
| 50 | 3 | **815** | 1,610 |
| 100 | 3 | **1,645** | 3,200 |
| 200 | 5 | **3,275** | 6,350 |

`gepa.optimize` uses the N=15 column as the threshold for the pre-flight warning.

## Other stop conditions

`max_metric_calls` is the ceiling, but you can configure additional stoppers via `stop_callbacks` so GEPA halts as soon as any condition fires:

- `NoImprovementStopper(max_iterations_without_improvement=N)` — stop when no improvement for N iterations.
- `TimeoutStopCondition(timeout_seconds=...)` — stop on wall-clock.
- `ScoreThresholdStopper(threshold=...)` — stop once a target score is hit.
- `SignalStopper` / `FileStopper` — graceful external stop signals.
- `CompositeStopper(...)` — combine several.

These are useful for converging fast on easy tasks without burning the full budget — but remember they only *lower* the effective iteration count. The budget you set still has to be large enough to clear the recommended floor in the worst case.

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

The recommended budget for that configuration is `> 15 × 6 = 90` (or precisely `6 + 15 * (3 + 6) = 141` if you compute exactly). Either would have prevented the failure.

## How to read the warnings

**At the start of the run (pre-flight):**

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

A pool size of 2 means baseline + 1 candidate — the optimizer didn't get to explore. The post-run check uses a more conservative threshold (3 accepted) than the pre-flight (~15 attempts), because not every proposal gets accepted into the pool. The pre-flight is what enforces the budget for ~15 attempts.

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
