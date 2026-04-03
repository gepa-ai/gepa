# Candidate Selection Strategies

Each iteration, GEPA picks one existing candidate to mutate. The **candidate selection strategy** controls which candidate is chosen. The right strategy depends on the shape of your objective landscape — single-metric, multi-metric, or heavily multi-objective.

---

## Built-in Strategies

### `"pareto"` (default)

Samples from the **non-dominated Pareto frontier**. Each candidate on the frontier is weighted by the number of frontier keys (validation examples and/or objectives) it appears in, so candidates that cover more of the frontier are sampled more often.

This is the best default for multi-objective or multi-example optimization — it naturally explores diverse trade-offs rather than collapsing to a single "best" candidate.

```python
# gepa.optimize
result = gepa.optimize(
    ...,
    candidate_selection_strategy="pareto",
)
```

### `"current_best"`

Always selects the candidate with the highest aggregate validation score. Good when you have a single metric and want to greedily refine the best-performing candidate.

```python
result = gepa.optimize(
    ...,
    candidate_selection_strategy="current_best",
)
```

### `"epsilon_greedy"`

With probability `epsilon` (default 0.1), selects a random candidate; otherwise picks the best by aggregate score. Provides a simple exploration/exploitation trade-off.

```python
result = gepa.optimize(
    ...,
    candidate_selection_strategy="epsilon_greedy",
)
```

### `"top_k_pareto"`

Restricts Pareto selection to the top K candidates by aggregate score (default K=5), then applies the same weighted-frequency sampling as `"pareto"`. Useful when your candidate pool is large and you want Pareto diversity but only among high-performing candidates.

```python
result = gepa.optimize(
    ...,
    candidate_selection_strategy="top_k_pareto",
)
```

---

## Configuring in `optimize_anything`

In `optimize_anything`, set the strategy on `EngineConfig`:

```python
from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig

config = GEPAConfig(
    engine=EngineConfig(
        candidate_selection_strategy="pareto",
        max_metric_calls=200,
    ),
)

result = optimize_anything(config=config, ...)
```

---

## How the Pareto Frontier Works

Understanding the Pareto frontier helps you pick the right strategy and `frontier_type`.

### Frontier keys and candidate frequency

GEPA maintains a mapping from **frontier keys** to the set of candidates that are "best" for that key. What counts as a frontier key depends on `frontier_type`:

| `frontier_type` | Frontier keys | Default in |
|---|---|---|
| `"instance"` | One key per validation example | `gepa.optimize` |
| `"objective"` | One key per objective in `side_info["scores"]` | — |
| `"hybrid"` | Both instance and objective keys | `optimize_anything` |
| `"cartesian"` | One key per (example, objective) pair | — |

When GEPA uses the `"pareto"` strategy, it counts how many frontier keys each non-dominated candidate appears in and samples proportionally. A candidate appearing in 5 keys is 5x more likely to be selected than one appearing in 1 key.

### Non-dominated candidates that aren't "best in class"

A common question: if a candidate is never the single best on any individual metric but performs well across all of them, will it still be sampled?

**Yes.** The Pareto frontier includes all **non-dominated** candidates — a candidate is non-dominated as long as no other single candidate beats it on every frontier key it appears in. A candidate that is "good but not best" on several metrics is non-dominated and will appear in the frontier, because the candidates that beat it on metric A don't beat it on metric B (and vice versa).

For example, consider three candidates scored on two objectives:

| Candidate | Accuracy | Speed |
|---|---|---|
| A | **0.95** | 0.30 |
| B | 0.50 | **0.90** |
| C | 0.80 | 0.70 |

Candidate C is never the best on either metric, but it is non-dominated — neither A nor B beats it on *both* metrics. All three candidates are on the Pareto frontier and eligible for selection.

!!! tip "Use `frontier_type='hybrid'` for multi-objective problems"
    With `"hybrid"` (the default in `optimize_anything`), GEPA tracks frontiers per validation example *and* per objective. This gives well-rounded candidates more frontier keys to appear in, increasing their sampling probability relative to candidates that are only extreme on one axis.

### Frontier type selection guide

- **Single metric, multiple examples**: `"instance"` — diversity comes from different validation examples.
- **Multiple metrics, few examples**: `"objective"` — diversity comes from the different objective scores.
- **Multiple metrics and examples** (most common): `"hybrid"` — combines both sources of diversity.
- **Fine-grained control**: `"cartesian"` — creates a key for every (example, objective) pair. Produces the richest frontier but may be noisy with many examples.

```python
# gepa.optimize
result = gepa.optimize(
    ...,
    candidate_selection_strategy="pareto",
    frontier_type="hybrid",
)

# optimize_anything
config = GEPAConfig(
    engine=EngineConfig(
        candidate_selection_strategy="pareto",
        frontier_type="hybrid",
    ),
)
```

---

## Custom Strategies

If the built-in strategies don't fit your needs, implement the `CandidateSelector` protocol and pass an instance directly:

```python
from gepa.proposer.reflective_mutation.base import CandidateSelector
from gepa.core.state import GEPAState


class MyCandidateSelector(CandidateSelector):
    def select_candidate_idx(self, state: GEPAState) -> int:
        # state.program_full_scores_val_set — list of aggregate scores per candidate
        # state.per_program_tracked_scores — list of tracked scores per candidate
        # state.get_pareto_front_mapping() — dict mapping frontier keys to sets of candidate indices
        # state.program_candidates — list of all candidates
        ...
        return chosen_index
```

Pass it wherever you would pass a string strategy name:

```python
# gepa.optimize
result = gepa.optimize(
    ...,
    candidate_selection_strategy=MyCandidateSelector(),
)

# optimize_anything
config = GEPAConfig(
    engine=EngineConfig(
        candidate_selection_strategy=MyCandidateSelector(),
    ),
)
```

---

## Which Strategy Should I Use?

| Scenario | Recommended strategy | Why |
|---|---|---|
| Multi-objective optimization | `"pareto"` | Explores the full Pareto frontier across metrics |
| Single metric, want fast convergence | `"current_best"` | Greedy refinement of the top performer |
| Single metric, worried about local optima | `"epsilon_greedy"` | Random exploration prevents getting stuck |
| Large candidate pool, multiple metrics | `"top_k_pareto"` | Pareto diversity among top performers only |
| Domain-specific selection logic | Custom `CandidateSelector` | Full control over selection |

!!! note "Multi-objective scoring"
    To enable multi-objective Pareto tracking, return a `"scores"` dict inside `side_info` from your evaluator:

    ```python
    def evaluator(candidate, example):
        ...
        side_info = {
            "scores": {
                "accuracy": accuracy_score,
                "efficiency": efficiency_score,
            },
        }
        return score, side_info
    ```

    All values in `"scores"` must follow **higher is better**. GEPA maintains the Pareto frontier across these objectives automatically.
