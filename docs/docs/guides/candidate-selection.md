# Candidate Selection Strategies

Each iteration, GEPA picks one existing candidate to mutate. The **candidate selection strategy** controls which candidate is chosen. The right strategy depends on the shape of your objective landscape — single-metric, multi-metric, or heavily multi-objective.

---

## Built-in Strategies

### `"pareto"` (default)

Samples from GEPA's **per-key Pareto frontier**. For each frontier key (a validation example, objective, or both — depending on `frontier_type`), GEPA tracks which candidates achieve the best score. Selection probability is proportional to how many keys a candidate is best on.

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

## How the Per-Key Pareto Frontier Works

Understanding how the frontier is built helps you decide when a built-in strategy is sufficient and when you need a custom one.

### Frontier keys and "best in class" tracking

GEPA maintains a mapping from **frontier keys** to the set of candidates that achieved the **highest score** for that key. A candidate only enters the frontier for a given key if it ties or beats the current best score on that key. What counts as a frontier key depends on `frontier_type`:

| `frontier_type` | Frontier keys | Default in |
|---|---|---|
| `"instance"` | One key per validation example | `gepa.optimize` |
| `"objective"` | One key per objective in `side_info["scores"]` | — |
| `"hybrid"` | Both instance and objective keys | `optimize_anything` |
| `"cartesian"` | One key per (example, objective) pair | — |

When GEPA uses the `"pareto"` strategy, it counts how many frontier keys each candidate appears in and samples proportionally. A candidate appearing in 5 keys is 5x more likely to be selected than one appearing in 1.

### When "good across the board" candidates are missed

Because the frontier tracks the **best per key**, a candidate that is competitive but never the single best on any key will not appear in the frontier. Consider three candidates scored on two objectives:

| Candidate | Accuracy | Speed |
|---|---|---|
| A | **0.95** | 0.30 |
| B | 0.50 | **0.90** |
| C | 0.80 | 0.70 |

With `frontier_type="objective"`, the frontier contains:

- accuracy key → {A}
- speed key → {B}

Candidate C is never best on either metric, so **it never enters the frontier** and is never eligible for selection — even though it is non-dominated in the true multi-objective sense (neither A nor B beats it on *both* metrics).

This is by design: the per-key frontier is lightweight and works well when you have many frontier keys (many validation examples or many objectives) that give well-rounded candidates opportunities to be best on at least some keys. But when you have few objectives and candidates that live in the non-dominated interior of the Pareto front, they can fall through the cracks.

### Frontier type selection guide

Choosing the right `frontier_type` determines how many keys exist, which affects how many candidates can enter the frontier:

- **Single metric, multiple examples**: `"instance"` — diversity comes from different validation examples.
- **Multiple metrics, few examples**: `"objective"` — diversity comes from the different objective scores. Note: only candidates that are the absolute best on at least one objective are sampled.
- **Multiple metrics and examples** (most common): `"hybrid"` — combines both sources of keys, giving more candidates a chance to enter the frontier.
- **Fine-grained control**: `"cartesian"` — creates a key for every (example, objective) pair. Most keys, so candidates have the most opportunities to be "best" on something.

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

If the built-in strategies don't fit your needs, implement the `CandidateSelector` protocol and pass an instance directly.

### The `CandidateSelector` protocol

```python
from gepa.proposer.reflective_mutation.base import CandidateSelector
from gepa.core.state import GEPAState


class MyCandidateSelector(CandidateSelector):
    def select_candidate_idx(self, state: GEPAState) -> int:
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

### Useful `GEPAState` fields

Inside `select_candidate_idx`, you have access to the full optimization state:

| Field | Type | Description |
|---|---|---|
| `state.program_candidates` | `list[dict[str, str]]` | All candidates (index = program index) |
| `state.program_full_scores_val_set` | `list[float]` | Aggregate validation score per candidate |
| `state.per_program_tracked_scores` | `list[float]` | Tracked scores used for ranking |
| `state.prog_candidate_objective_scores` | `list[dict[str, float]]` | Per-objective scores per candidate |
| `state.prog_candidate_val_subscores` | `list[dict[DataId, float]]` | Per-example scores per candidate |
| `state.get_pareto_front_mapping()` | `dict[FrontierKey, set[int]]` | Built-in per-key frontier mapping |

### Example: true multi-objective non-dominated selector

The built-in `"pareto"` strategy only samples candidates that are best-in-class on at least one frontier key. If you want to sample from all **non-dominated** candidates — including those that are competitive across multiple objectives without being the single best on any — you can implement a true Pareto dominance selector:

```python
import random
from gepa.proposer.reflective_mutation.base import CandidateSelector
from gepa.core.state import GEPAState


class NonDominatedSelector(CandidateSelector):
    """Samples uniformly from all non-dominated candidates across objectives."""

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random(0)

    def select_candidate_idx(self, state: GEPAState) -> int:
        scores = state.prog_candidate_objective_scores
        if not scores or not scores[0]:
            # No objective scores available; fall back to best aggregate
            return state.program_full_scores_val_set.index(
                max(state.program_full_scores_val_set)
            )

        objectives = list(scores[0].keys())
        n = len(scores)

        # Find all non-dominated candidates
        non_dominated = []
        for i in range(n):
            dominated = False
            for j in range(n):
                if i == j:
                    continue
                # j dominates i if j is >= on all objectives and > on at least one
                if all(
                    scores[j].get(obj, 0) >= scores[i].get(obj, 0) for obj in objectives
                ) and any(
                    scores[j].get(obj, 0) > scores[i].get(obj, 0) for obj in objectives
                ):
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(i)

        return self.rng.choice(non_dominated)
```

With the example from earlier (A: 0.95/0.30, B: 0.50/0.90, C: 0.80/0.70), this selector would include all three candidates since none is dominated by another.

```python
result = gepa.optimize(
    ...,
    candidate_selection_strategy=NonDominatedSelector(),
)

# or with optimize_anything
config = GEPAConfig(
    engine=EngineConfig(
        candidate_selection_strategy=NonDominatedSelector(),
    ),
)
```

!!! tip "Combining with per-example diversity"
    You can extend `NonDominatedSelector` to also consider per-example subscores from `state.prog_candidate_val_subscores`, or weight candidates by their crowding distance (how isolated they are in objective space) to favor under-explored regions of the Pareto front.

---

## Which Strategy Should I Use?

| Scenario | Recommended strategy | Why |
|---|---|---|
| General optimization | `"pareto"` | Default; explores per-key frontier across examples and objectives |
| Single metric, want fast convergence | `"current_best"` | Greedy refinement of the top performer |
| Single metric, worried about local optima | `"epsilon_greedy"` | Random exploration prevents getting stuck |
| Large candidate pool, multiple metrics | `"top_k_pareto"` | Per-key frontier diversity among top performers only |
| Multi-objective with few objectives | Custom `NonDominatedSelector` | Includes all non-dominated candidates, not just per-key bests |
| Domain-specific selection logic | Custom `CandidateSelector` | Full control over selection |

!!! note "Multi-objective scoring"
    To enable multi-objective tracking, return a `"scores"` dict inside `side_info` from your evaluator:

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

    All values in `"scores"` must follow **higher is better**. GEPA maintains the frontier across these objectives automatically.
