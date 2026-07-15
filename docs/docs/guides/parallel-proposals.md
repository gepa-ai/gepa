# Parallel Proposals: Sampling & Selection Strategies

GEPA's reflective mutation loop can generate and evaluate **multiple candidate
proposals per iteration**. This is controlled by two small, composable
strategy objects rather than a numeric knob:

- a **sampling strategy** decides *which (parent, minibatch) proposal tasks*
  are attempted this iteration, and
- a **selection strategy** decides *which of the resulting improvements* enter
  the candidate pool.

The defaults (`SingleMutationSampling` + `AllImprovements`) reproduce classic
GEPA behavior exactly: one parent, one mutation per iteration. Existing runs
do not change.

!!! note
    These strategies are the interface for multi-proposal iterations
    (design: [RFC #329](https://github.com/gepa-ai/gepa/issues/329)).

## Sampling strategies

```python
from gepa.strategies.proposal_sampling import (
    SingleMutationSampling,  # default: 1 parent, 1 mutation
    SameParentSampling,      # n mutations of the same parent
    IndependentSampling,     # n independent (parent, minibatch) tasks
    PxNSampling,             # p parents x n mutations each
)
```

| Strategy | Per iteration | Use when |
|---|---|---|
| `SingleMutationSampling()` | 1 parent, 1 mutation | Default; classic GEPA |
| `SameParentSampling(n=4)` | 1 parent, `n` mutations | Exploit a strong parent with diverse edits |
| `IndependentSampling(n=4)` | `n` parents, 1 mutation each | Explore the frontier broadly |
| `PxNSampling(p=2, n=2)` | `p` parents × `n` mutations | Balanced explore/exploit |

## Selection strategies

```python
from gepa.strategies.proposal_selection import (
    AllImprovements,   # default: every improving proposal joins the pool
    BestImprovement,   # only the single best improvement joins
    TopKImprovements,  # the k best improvements join
)
```

## Using them

Both `gepa.optimize()` and `optimize_anything` accept the strategies directly:

```python
import gepa
from gepa.strategies.proposal_sampling import PxNSampling
from gepa.strategies.proposal_selection import TopKImprovements

result = gepa.optimize(
    seed_candidate=seed,
    trainset=trainset,
    valset=valset,
    task_lm="openai/gpt-4.1-mini",
    reflection_lm="openai/gpt-5",
    max_metric_calls=600,
    sampling_strategy=PxNSampling(p=2, n=2),
    selection_strategy=TopKImprovements(k=2),
)
```

Every proposal is still evaluated on its minibatch. The selection strategy
is the authority over which proposals enter the pool: it receives **all**
evaluated proposals plus the configured `acceptance_criterion`. The built-in
strategies apply the criterion (so defaults only admit improvements); a custom
strategy may deliberately surface a proposal the criterion would reject — e.g.
an exploration pick. Anything not selected fires `on_candidate_rejected`.

## Where the parallelism actually runs: `batch_evaluate`

Multi-proposal iterations batch their evaluations at the **adapter edge**. The
engine and proposer stay sequential; adapters opt into true parallelism by
implementing one optional method:

```python
class MyAdapter:
    def evaluate(self, batch, candidate, capture_traces=False): ...

    # Optional. When absent, GEPA calls evaluate() once per item.
    def batch_evaluate(self, items: list[tuple[Candidate, list]]) -> list[EvaluationBatch]:
        ...  # e.g. fan out across threads, processes, or an eval service
```

Third-party adapters that only implement `evaluate()` keep working unmodified
(`default_batch_evaluate` is the sequential fallback). The reflection LM side
is batched analogously: the built-in stateless reflector batches all
per-component reflection calls via `litellm.batch_completion`.

### `batch_evaluator`: one external call for the whole batch

In `optimize_anything`, you can take over batching entirely: pass
`batch_evaluator=` and GEPA hands you **all pending `(candidate, example)`
pairs in a single call** — submit them as one provider batch job, one Slurm
array, one Ray job — and return one result per pair (`score` or
`(score, side_info)`).

```python
def my_batch_eval(pairs, opt_states):   # opt_states is optional — declare it to receive warm-start state
    jobs = [submit(candidate, example) for candidate, example in pairs]
    return [(job.score(), {"log": job.diagnostics()}) for job in jobs]

optimize_anything(evaluator=my_eval, batch_evaluator=my_batch_eval, ...)
```

The contract, precisely: str-candidate unwrapping, per-pair `opt_states`
injection (if your signature accepts it), exception policy
(`raise_on_exception`), **evaluation caching** (shared store and keys with the
sequential path — only cache misses reach your function, still as one call),
warm-start history, and output packaging all match the sequential path. What
does **not** apply — structurally, because there is no per-evaluation call to
scope them to — is `oa.log()` and stdout/stderr capture: return diagnostics
inside each pair's `side_info` instead.

`evaluator=` is optional when `batch_evaluator=` is set. With only a batch
function, *everything* routes through it: multi-pair work (minibatch stages,
valset/seed/merge evaluations) as grouped calls, and single-pair resolutions
(refiner steps) as **singleton batches** — make your function efficient for
`len(pairs) == 1` if you enable the refiner. When both are provided, grouped
work prefers `batch_evaluator` and singles use `evaluator` (whose per-call
`oa.log()`/stdio capture then applies to those calls).

## Budget accounting and events

Totals are identical to the sequential path, at finer event granularity: each
iteration emits separate `on_budget_updated` events for the parent-evaluation
and child-evaluation stages (plus one per accepted candidate's full-valset
evaluation), and `metric_calls_used` remains monotonic with self-consistent
deltas. If you aggregate budget events, sum `metric_calls_delta` rather than
counting events.

## Advanced: custom reflection strategies

The reflection call itself is a seam (`ReflectionLM`,
[RFC #329](https://github.com/gepa-ai/gepa/issues/329)): pass
`reflection_strategy=` to `gepa.optimize()` (or set
`ReflectionConfig.reflection_strategy` in `optimize_anything`) to replace the stateless
single-call reflector with a stateful or aggregating implementation (e.g.
session-based reflection, or map-reduce aggregation over parallel reflections
as in [ComBEE, PR #307](https://github.com/gepa-ai/gepa/pull/307)).
Implementations provide `reflect()`; `reflect_many()` is optional and enables
batched reflection.
