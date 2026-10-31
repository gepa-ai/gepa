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

### `NonRegressionAcceptance`

An aggregate improvement can hide a regression on an individual example. Use
`NonRegressionAcceptance` when a passing check must remain passing on each
example in the evaluated minibatch.

The criterion wraps another acceptance criterion. It first checks the protected
objectives, then applies the base criterion if those checks pass. Both `base`
and `protected` are required; GEPA's default remains `"strict_improvement"`.

```python
from gepa.strategies.acceptance import (
    NonRegressionAcceptance,
    StrictImprovementAcceptance,
)

criterion = NonRegressionAcceptance(
    base=StrictImprovementAcceptance(),
    protected=("schema_valid", "required_field"),
)
```

Pass `criterion` as `acceptance_criterion` to `gepa.optimize`, or assign it to
`EngineConfig(acceptance_criterion=criterion)` for `optimize_anything`.
To allow equal aggregate scores after the protected checks pass, use
`ImprovementOrEqualAcceptance()` as the base instead.

#### Report protected checks

The rule reads `eval_before.objective_scores` and
`eval_after.objective_scores`. For each protected objective, exactly `1.0`
means pass. Any other finite score means the check has not passed. Omit an
objective key on examples where the check does not apply.

For an adapter, return these dictionaries through
`EvaluationBatch.objective_scores`. Each dictionary must correspond to the same
example as the score at that position, in the input batch order. For example:

```python
from gepa.core.adapter import EvaluationBatch

parent_evaluation = EvaluationBatch(
    outputs=["valid structured answer", "incorrect answer", "incorrect answer"],
    scores=[1.0, 0.0, 0.0],
    objective_scores=[{"schema_valid": 1.0}, {}, {}],
)
candidate_evaluation = EvaluationBatch(
    outputs=["invalid structured answer", "correct answer", "correct answer"],
    scores=[0.0, 1.0, 1.0],
    objective_scores=[{"schema_valid": 0.0}, {}, {}],
)
```

The aggregate score rises from `1.0` to `2.0`, but `schema_valid` changes from
pass to fail on the first example. The wrapper rejects this candidate before
calling the base criterion. An objective that the parent already failed does
not prevent acceptance. Examples with no applicable protected objectives are
left to the base criterion.

For an `optimize_anything` evaluator, return the same objective dictionary in
`side_info["scores"]`, as described in [Multi-objective scoring](#custom-criteria)
below. Encode protected checks as pass or fail rather than passing a continuous
metric whose maximum has another meaning.

#### Rejection behavior and scope

The criterion rejects proposals when either evaluation or its objective scores
are missing, when the row counts are empty or inconsistent, or when protected
scores are nonnumeric or nonfinite. A missing candidate key also causes
rejection if the parent passed that check. The rejection message identifies the
check and minibatch index, or explains why the data could not be verified.
Ensure the names in `protected` match the keys your evaluator emits. An absent
parent key means the check does not apply on that row.

!!! note "Minibatch scope"
    The rule compares each proposed candidate with its parent on the evaluated
    minibatch and makes no additional model calls. It does not check unsampled
    examples, enforce a fixed seed baseline, or account for evaluation noise.
    GEPA's built-in selection strategies honor the criterion. Custom selection
    strategies must honor it too. Merge proposals use a separate acceptance
    path; keep `use_merge=False` (the default) when relying on this rule.

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
