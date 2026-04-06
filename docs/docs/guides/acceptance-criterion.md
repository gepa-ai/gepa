# Acceptance Criterion

Each iteration, GEPA evaluates a proposed candidate on a minibatch and decides whether to accept it into the candidate pool. The **acceptance criterion** controls this decision. By default, GEPA requires the sum of minibatch scores to strictly improve. You can change this by passing a different built-in strategy or implementing your own.

---

## Built-in Criteria

### `"strict_improvement"` (default)

Accept only if the total minibatch score strictly improves:

```python
result = gepa.optimize(
    ...,
    acceptance_criterion="strict_improvement",
)
```

### `"improvement_or_equal"`

Also accept lateral moves (equal score). Useful for exploring different regions of the solution space when many candidates score the same:

```python
result = gepa.optimize(
    ...,
    acceptance_criterion="improvement_or_equal",
)
```

---

## Configuring in `optimize_anything`

```python
from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig

config = GEPAConfig(
    engine=EngineConfig(
        acceptance_criterion="improvement_or_equal",
    ),
)

result = optimize_anything(config=config, ...)
```

---

## Custom Criteria

Implement the `AcceptanceCriterion` protocol and pass an instance directly. The method receives:

- **`proposal`** — the full `CandidateProposal`, including `eval_before` and `eval_after` (`SubsampleEvaluation` objects with per-example `scores`, `outputs`, `objective_scores`, and `trajectories`).
- **`state`** — the full `GEPAState` (all candidates, validation scores, Pareto frontier, iteration count, etc.).

```python
from gepa.strategies.acceptance import AcceptanceCriterion
from gepa.proposer.base import CandidateProposal
from gepa.core.state import GEPAState
```

### Example: accept if any minibatch example improves

The default sums all scores and requires the total to go up. This means a large regression on one example can mask improvements on others. If you want to accept whenever *at least one* example improved (regardless of regressions elsewhere):

```python
class AnyExampleImproved:
    def should_accept(self, proposal: CandidateProposal, state: GEPAState) -> bool:
        old = proposal.subsample_scores_before or []
        new = proposal.subsample_scores_after or []
        return any(n > o for n, o in zip(new, old))
```

### Example: accept if any objective improves across the minibatch

When your evaluator returns multi-objective scores (via `side_info["scores"]`), you may want to accept a candidate that improves on *any single objective* aggregated across the minibatch — even if the blended score doesn't improve. This is useful for multi-objective optimization where you want the candidate pool to explore different trade-off directions:

```python
class AnyObjectiveImproved:
    """Accept if any objective's total across the minibatch increased."""

    def should_accept(self, proposal: CandidateProposal, state: GEPAState) -> bool:
        if proposal.eval_before is None or proposal.eval_after is None:
            return False
        old_obj = proposal.eval_before.objective_scores
        new_obj = proposal.eval_after.objective_scores
        if old_obj is None or new_obj is None:
            # Fall back to aggregate score comparison
            return sum(proposal.subsample_scores_after or []) > sum(
                proposal.subsample_scores_before or []
            )

        # Collect all objective names
        objectives = set()
        for s in old_obj:
            objectives.update(s.keys())

        # Accept if any single objective's sum improved
        for obj in objectives:
            old_total = sum(s.get(obj, 0.0) for s in old_obj)
            new_total = sum(s.get(obj, 0.0) for s in new_obj)
            if new_total > old_total:
                return True
        return False
```

```python
result = gepa.optimize(
    ...,
    acceptance_criterion=AnyObjectiveImproved(),
)
```

!!! note "Multi-objective scoring"
    To enable multi-objective tracking, return a `"scores"` dict inside `side_info` from your evaluator:

    ```python
    def evaluator(candidate, example):
        ...
        side_info = {
            "scores": {
                "accuracy": accuracy_score,
                "cost": cost_score,
            },
        }
        return score, side_info
    ```
