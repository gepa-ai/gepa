# Batch Sampling Strategies

Each iteration, GEPA samples a **minibatch** of training examples to evaluate the current candidate on. The **batch sampler** controls which examples are selected and in what order. This directly affects what feedback the reflection LM sees — and therefore what improvements it proposes.

GEPA ships with one built-in strategy (`EpochShuffledBatchSampler`) and a `BatchSampler` protocol for writing your own.

---

## How Batch Sampling Fits into the Loop

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Select candidate │ ──▶ │ Sample minibatch │ ──▶ │  Evaluate + Reflect
│ from Pareto front│     │ (batch sampler)  │     │  on minibatch    │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

The batch sampler is called once per iteration. It returns a list of training example IDs, which are then fetched and passed to the adapter for evaluation. The reflection LM sees the outputs and feedback from **only these examples**, so the minibatch composition determines what failure modes get surfaced each iteration.

---

## Built-in: `"epoch_shuffled"` (default)

`EpochShuffledBatchSampler` shuffles the training set and walks through it in consecutive chunks of `reflection_minibatch_size`. Once all examples have been seen (one epoch), it reshuffles and starts over.

### Key properties

- **Coverage**: Every training example is seen exactly once per epoch before any is repeated.
- **Deterministic**: Seeded by the global `seed` parameter — same seed produces the same sequence.
- **Padding**: If the training set size isn't divisible by `minibatch_size`, the least-frequently-seen examples are repeated to fill the last chunk.

### Configuration

**With `gepa.optimize()`:**

```python
import gepa

result = gepa.optimize(
    ...,
    batch_sampler="epoch_shuffled",       # default
    reflection_minibatch_size=5,          # examples per iteration (default: 3)
    seed=42,
)
```

**With `optimize_anything()`:**

```python
from gepa.optimize_anything import optimize_anything, GEPAConfig, ReflectionConfig

result = optimize_anything(
    ...,
    config=GEPAConfig(
        reflection=ReflectionConfig(
            reflection_minibatch_size=5,  # default: 1 (single-task) or 3 (multi-task)
        ),
    ),
)
```

!!! tip "Choosing `reflection_minibatch_size`"
    - **Smaller (1-3)**: Each iteration focuses on fewer examples, giving the reflection LM more detailed feedback per example. Better for tasks where individual failures are informative. More iterations needed to cover the full training set.
    - **Larger (5-20)**: The reflection LM sees a broader cross-section of failures each iteration. Better for tasks with many distinct failure modes. Fewer iterations needed per epoch, but the reflection prompt is longer.
    - The default (3) works well in most cases. Increase it if you notice the optimizer is slow to discover certain failure modes, or decrease it if the reflection LM is getting overwhelmed by too many examples.

---

## Choosing Minibatch Size

The `reflection_minibatch_size` parameter has a direct impact on optimization dynamics:

| Minibatch size | Examples per epoch | Iterations per epoch | Trade-off |
|---|---|---|---|
| 1 | N | N | Deep focus on individual examples; slow coverage |
| 3 (default) | N | N/3 | Balanced |
| 10 | N | N/10 | Broad view; shorter per-example feedback |
| 20 | N | N/20 | Fastest coverage; risk of reflection prompt being too long |

Where N is the training set size.

For **classification tasks** with many categories, a minibatch of 5-10 helps the reflection LM see errors across multiple categories in one iteration, enabling cross-category disambiguation rules.

For **code optimization** or **single-task** problems, a minibatch of 1-3 is usually sufficient since each example provides rich diagnostic feedback.

---

## GEPA-Curator: choosing evidence views

`TaskCuratorSampler` selects tasks using an LLM and retains successful executions for reflection on later failures. Pass it through the existing `batch_sampler` argument; existing adapters do not need new methods.

We recommend using GEPA-Curator with greedy GEPA. The [GEPA-Curator paper](https://arxiv.org/abs/2610.37651v1) evaluates this setting, where each accepted mutation becomes the parent for the next iteration. The example below follows that single lineage.

An optional `evidence_fn(candidate, evaluation)` controls what the curator sees. Like an adapter's `make_reflective_dataset`, it receives the evaluated candidate and an `EvaluationBatch`. Here the batch always contains **one task**, with its original output, score, trajectory, and objective scores. `num_metric_calls` is `None`, since a batch's cost cannot be apportioned to individual tasks.

Return `CuratorEvidence` with three independent views:

| Field | Used for | Default |
|---|---|---|
| `history` | Detailed recent executions | Supplied source evidence |
| `summary` | Older executions, alongside score and outcome metadata | No additional summary |
| `success` | Success selection and the retained reference injected into reflection | Supplied source evidence |

Source evidence is the trajectory, or its explicit `task_curator_specific_info` packet when present. Shortening `history` does not shorten the default `success`. Set `history=None` for metadata-only history, or `success=None` to decline to provide an anchor. A selected success is retained independently of the history window.

For example, suppose your evaluator supplies trajectories containing `steps` and `feedback`:

```python
from gepa.strategies.task_curator import CuratorEvidence, TaskCuratorSampler


def curator_evidence(candidate, evaluation):
    trace = evaluation.trajectories[0]
    return CuratorEvidence(
        history={"steps": trace["steps"][-3:], "feedback": trace["feedback"]},
        summary={"step_count": len(trace["steps"])},
        success=trace,  # Keep the complete supplied trajectory for the anchor.
    )


curator = TaskCuratorSampler(
    lm=curator_lm,
    minibatch_size=16,
    history_window=3,
    evidence_fn=curator_evidence,
)

class LatestAcceptedSelector:
    def select_candidate_idx(self, state):
        return len(state.program_candidates) - 1


result = gepa.optimize(
    ...,
    adapter=adapter,
    batch_sampler=curator,
    candidate_selection_strategy=LatestAcceptedSelector(),
)
```

The callback is ordinary Python; it adds no LLM calls unless you implement them yourself. You can also define it on your adapter and pass `evidence_fn=adapter.make_curator_evidence`. This is an optional convention, not an addition to the adapter protocol. It does not replace `make_reflective_dataset`, which still controls the current execution's reflection material.

The same sampler works with `ReflectionConfig(batch_sampler=curator)`. With the current `OptimizeAnythingConfig`, use `engine_config={"reflection": {"batch_sampler": curator}}`. An `optimize_anything` evaluator still returns `(score, info)`; the callback receives that `info` as its trajectory.

### Recording, compression, and custom metrics

- `history_window=3` keeps the most recent three execution units in detail; a before/after pair counts as one. Use `None` to keep all units in detail. Older executions retain framework metadata plus your optional `summary`.
- Each rollout keeps its numeric score and one of four outcomes. By default, `score == 1` gives `success` and every other finite score gives `fail`. An actor timeout returned with a failure score is a `fail`.
- `infrastructure_error` takes precedence when an evaluator catches a crash and explicitly marks the affected rollout. The built-in `optimize_anything` scalar and batch evaluator wrappers mark recovered exceptions. Custom adapters can set the boolean `infrastructure_error=True` on the trajectory or its explicit curator packet. An adapter returning only a failure score follows ordinary scoring. Reflection or validation errors later in the iteration leave completed rollout labels intact.
- For custom metrics, `is_success` and `is_failure` receive the original score and trajectory. If an explicit failure predicate and the success predicate both return false, the result is `unknown`. For example, a pass threshold of 80 can be set with `is_success=lambda score, trace: score >= 80`. Only `success` can supply an anchor, and `fail` can receive one in reflection.
- Older checkpoints may contain labels inferred from error messages. On resume, the curator keeps saved success and failure judgments, recognizes explicitly marked crashes, and falls back to binary scoring for the default metric. An old result with insufficient evidence for a custom metric is `unknown`.
- Before and after evaluations use the candidate that actually produced each result, including rejected mutations. Shared parent evaluations reuse the recorded views. The curator sees the same success content during selection that the reflector later receives as its anchor.
- Views are snapshotted when recorded. Resume restores those snapshots without rerunning the callback. Changing the callback affects new records; it cannot reconstruct evidence omitted from old checkpoints. History, anchors, and cooldowns remain scoped to candidate lineages.

### Tuning the curator prompt

The curator prompt is also a tunable part of the method. As good practice, adapt its task-selection and success-selection guidance to your environment, evaluator, and evidence format, then evaluate the changes on validation data. Check that its definitions of mastery, low-signal failures, and useful success references fit your tasks. The `evidence_fn` callback controls the evidence presented to the curator; tuning the instructions themselves is a separate change to the curator prompt.

### Data contract and compatibility

Do not mutate callback inputs. Return text or JSON-serializable data; unsupported objects and non-finite numbers are rejected. Ordinary JSON conversions apply, such as tuples becoming lists and numeric dictionary keys becoming strings. For opaque trajectories or images, explicitly provide the fields or text you want shown. A callback may build views from `evaluation.outputs` even if trajectories are absent.

Without `evidence_fn`, the curator retains the supplied source evidence without filtering its field names. Default serialization uses `str()` for opaque objects, which may not capture their execution details; provide a callback when a meaningful projection is needed.

Existing `evidence_extractor` and `success_extractor` callbacks remain supported and keep their previous JSON-safe conversion. Do not combine them with `evidence_fn`. With those legacy callbacks, a custom history extractor also supplies the default success view unless `success_extractor` is provided. Returning `None` from a success extractor declines to provide an anchor.

---

## Custom Batch Samplers

You can implement your own batch sampler by writing a class that satisfies the `BatchSampler` protocol:

```python
from gepa.strategies.batch_sampler import BatchSampler
from gepa.core.data_loader import DataLoader
from gepa.core.state import GEPAState


class MyBatchSampler:
    """Custom batch sampler that always picks the hardest examples."""

    def __init__(self, minibatch_size: int):
        self.minibatch_size = minibatch_size

    def next_minibatch_ids(
        self, loader: DataLoader, state: GEPAState
    ) -> list:
        # Access the Pareto front to find the hardest examples
        hardest = sorted(
            state.pareto_front_valset.items(),
            key=lambda x: x[1],  # sort by best score (ascending)
        )
        # Return the IDs with the lowest best scores
        ids = [val_id for val_id, _score in hardest[: self.minibatch_size]]

        # Fall back to all IDs if not enough in the Pareto front
        if len(ids) < self.minibatch_size:
            all_ids = list(loader.all_ids())
            ids = all_ids[: self.minibatch_size]

        return ids
```

### Using a custom sampler

**With `gepa.optimize()`:**

```python
result = gepa.optimize(
    ...,
    batch_sampler=MyBatchSampler(minibatch_size=5),
)
```

!!! note
    When passing a custom `BatchSampler` instance, do **not** set `reflection_minibatch_size` — that parameter only applies to the built-in `"epoch_shuffled"` sampler.

**With `optimize_anything()`:**

```python
from gepa.optimize_anything import optimize_anything, GEPAConfig, ReflectionConfig

result = optimize_anything(
    ...,
    config=GEPAConfig(
        reflection=ReflectionConfig(
            batch_sampler=MyBatchSampler(minibatch_size=5),
        ),
    ),
)
```

### The `BatchSampler` protocol

```python
class BatchSampler(Protocol[DataId, DataInst]):
    def next_minibatch_ids(
        self, loader: DataLoader[DataId, DataInst], state: GEPAState
    ) -> list[DataId]: ...
```

Your sampler receives:

- **`loader`**: The training set `DataLoader`, with `loader.all_ids()` returning all available IDs and `len(loader)` for the size.
- **`state`**: The full `GEPAState`, giving you access to `state.i` (current iteration), `state.pareto_front_valset` (per-example best scores), `state.program_candidates` (all candidates), and more.

It must return a list of data IDs. The engine will call `loader.fetch(ids)` to retrieve the actual examples.

---

## API Reference

- [`BatchSampler` protocol](../api/strategies/BatchSampler.md)
- [`EpochShuffledBatchSampler`](../api/strategies/EpochShuffledBatchSampler.md)
