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
