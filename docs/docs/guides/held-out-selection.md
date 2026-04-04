# Held-Out Selection Policy

GEPA now supports a separate `held_out` split for final model selection.

This is useful because `valset` is doing double duty:

- it is the set used to maintain the Pareto frontier during optimization
- it is also the set you might otherwise use to pick the final candidate

That coupling can bias final selection toward candidates that happened to survive search pressure on the Pareto/validation set, even if they generalize worse on truly unseen data.

The safer setup is a four-way split:

- `trainset`: examples used for reflection and mutation
- `valset`: the validation set used during optimization for Pareto tracking and candidate selection
- `held_out`: a separate held-out validation split used only for final candidate selection
- `test`: a final offline benchmark you do not touch during development

## How It Works

When you pass `held_out=...` to `gepa.optimize()`:

1. GEPA still evaluates accepted candidates on the full `valset`.
2. The current valset leader is evaluated on `held_out` the first time it becomes the leader.
3. `HeldOutSetEvaluationPolicy` selects the final best candidate by average held-out score.
4. Candidates that never become the valset leader are never evaluated on `held_out`.

This keeps held-out usage cheap while ensuring the final returned candidate is chosen using a split that did not drive the search loop itself.

## Basic Usage

```python
import gepa

result = gepa.optimize(
    seed_candidate={"system_prompt": "You are a helpful assistant."},
    trainset=trainset,
    valset=valset,
    held_out=held_out_set,
    adapter=adapter,
    reflection_lm="openai/gpt-5",
    max_metric_calls=200,
)
```

If `held_out` is provided and you do not explicitly pass a held-out-aware policy, GEPA uses `HeldOutSetEvaluationPolicy`.

If you explicitly pass `FullEvaluationPolicy()` together with `held_out`, GEPA raises an error.

## Policy Semantics

`HeldOutSetEvaluationPolicy` separates search-time selection from final model selection.

This means you should think of:

- `valset` as the search-time selection set
- `held_out` as the final model-selection set

## Result Semantics

When `held_out` is configured, `GEPAResult` exposes both rankings:

```python
result.best_idx         # best candidate by held-out score
result.valset_best_idx  # best candidate by valset score
result.held_out_scores  # {candidate_idx: average_held_out_score}
```

This is the main behavior change to be aware of: `result.best_idx` no longer necessarily points to the candidate with the highest valset score.

## Budget Semantics

Held-out evaluations are tracked separately from the main optimization budget.

- `result.total_metric_calls` does not include held-out evaluations
- `result.num_held_out_evals` reports how many held-out examples were scored

This lets you keep the search budget stable while still tracking model-selection cost.

## Relationship To Existing Policies

- `FullEvaluationPolicy`: evaluate all of `valset` and select the final candidate by valset score
- `HeldOutSetEvaluationPolicy`: evaluate all of `valset`, evaluate promoted leaders on `held_out`, and select the final candidate by held-out score

If you pass a custom `val_evaluation_policy` together with `held_out`, it must be a `HeldOutSetEvaluationPolicy`-compatible policy. Otherwise GEPA raises an error.
