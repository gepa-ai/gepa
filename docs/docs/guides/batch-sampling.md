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

With the default single-proposal strategy, the batch sampler is called once per iteration. [Parallel proposal strategies](parallel-proposals.md) can call it once per proposal task within the same iteration. It returns training example IDs, which are then fetched and passed to the adapter for evaluation. The adapter builds the reflective dataset from these evaluations, so the minibatch composition determines what failure modes can be surfaced to the reflection LM.

---

## Built-in: `"epoch_shuffled"` (default)

`EpochShuffledBatchSampler` shuffles the training set and walks through it in consecutive chunks of `reflection_minibatch_size`. Once all examples have been seen (one epoch), it reshuffles and starts over.

### Key properties

- **Coverage**: For a fixed, nonempty training set and one sampling call per consecutive iteration, each epoch contains every training ID once, followed by any padding repeats. Padding can share the last chunk with previously unseen IDs.
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
from gepa.optimize_anything import OptimizeAnythingConfig, optimize_anything

result = optimize_anything(
    ...,
    config=OptimizeAnythingConfig(
        engine="gepa",
        engine_config={
            "reflection": {
                "reflection_minibatch_size": 5,  # default: 1 (single-task) or 3 (multi-task)
            },
        },
    ),
)
```

`reflection_minibatch_size` controls training examples per proposal task, not the validation set size or the number of proposals. Use a positive integer. With the built-in sampler, a size greater than the training set is padded with repeated IDs; it does not create more distinct feedback. In single-task mode there is only one training example, so increasing the size repeats that example.

---

## Choosing Minibatch Size

For a fixed training set of `N > 0` examples and batch size `b > 0`, the built-in sampler creates `ceil(N / b)` chunks per epoch, containing `b * ceil(N / b)` example slots. The difference from `N` is padding, not new examples. With one sampling call per consecutive iteration, this takes `ceil(N / b)` iterations. Do not use this as an iteration count for parallel proposals or a custom sampler.

| Setting | Potential benefit | What to check |
|---|---|---|
| Smaller batches | Focused feedback and cheaper parent/child minibatch evaluations | More sampling calls to cover training IDs; rare failure modes may not appear early |
| Larger batches | More examples available for reflection in one proposal | Longer traces, conflicting feedback, and fewer proposals within an evaluation budget |
| `b > N` | No additional distinct training examples | Repeated IDs and feedback; prefer `b <= N` when tuning |

The default is a starting point, not a guarantee of the best size. A shuffled batch is not stratified: increasing its size does not guarantee coverage of every class or failure mode. Inspect the actual feedback, especially for imbalanced datasets or long code-execution traces.

### Account for evaluation and reflection budgets

In the ordinary single-proposal path, GEPA evaluates the parent on `b` examples with traces, reflects, and evaluates a valid child on the same `b` examples. With one metric call per example, no cache savings, and no skipped proposal, this is `2 * b` training metric calls. A child that passes minibatch acceptance also needs validation: the default full evaluation policy evaluates the validation set, adding `V` calls for `V` validation examples without cache savings. The seed needs validation before the search starts as well.

This is a planning estimate, not an exact run cost or a hard budget reservation. Perfect-score skips, missing trajectories, invalid proposals, caching, adapter-reported `num_metric_calls`, merges, custom evaluation policies, and parallel proposals change the accounting. Check the reported evaluation totals rather than inferring cost from iteration count.

Larger batches can therefore reduce the number of proposed improvements under a fixed `max_metric_calls`. Reflection tokens and money are a separate budget: a batch of three long traces may cost more than ten short ones. Record both evaluator usage and [reflection cost/token usage](cost-tracking.md), when your LM exposes them. For `optimize_anything`, use its server-side `max_evals` budget and GEPA engine reflection settings; do not assume its evaluation accounting is identical to `gepa.optimize`.

### A controlled tuning procedure

1. Fix the seed candidate, train/validation split, evaluator, reflection LM and its settings, and all other optimizer settings. Keep a separate test set out of size selection.
2. Start with a small grid such as the distinct values of `min(1, N)`, `min(3, N)`, and `min(5, N)` for a nonempty training set. These are trial sizes, not measured recommendations. Include the default as a baseline where applicable.
3. Give each size the same evaluation budget and, where supported, the same reflection-cost cap. Use fresh samplers and separate run directories so a resumed state or reused cache does not contaminate the comparison. Repeat the same set of seeds across sizes; a shared seed alone does not ensure identical model responses.
4. Record validation score, actual evaluations, proposal count, wall time, reflection tokens/cost, and which failure modes appear in the reflective dataset. [Callbacks](callbacks.md) can expose sampled IDs, evaluation skips, and reflective datasets. Compare quality at comparable actual resource usage, not just equal iteration counts.
5. If failures remain unseen, try a larger batch or a sampler tailored to training labels. If feedback is too long, truncated, or contradictory, try a smaller batch or a reflection strategy designed for larger batches, such as [ComBEE](combee.md). Recheck validation quality after each change rather than assuming broader feedback improves it.
6. Select a size using validation results across seeds, then evaluate the selected candidate on the untouched test set. Do not report a quality or speed improvement without measurements from your task.

### Reproduce coverage without an LLM

This example calls the real `EpochShuffledBatchSampler` on seven examples. It uses only the iteration counter from optimizer state and models one sampling call per consecutive iteration, starting at zero. Save the block as a Python script and run it with `uv run python <script.py>` from a GEPA checkout. No API key or model call is needed.

```python
# Offline coverage example
import random
from math import ceil
from types import SimpleNamespace

from gepa.core.data_loader import ListDataLoader
from gepa.strategies.batch_sampler import EpochShuffledBatchSampler

loader = ListDataLoader(list(range(7)))
rows = []
for size in (1, 3, 5, 10):
    sampler = EpochShuffledBatchSampler(size, rng=random.Random(42))
    calls = ceil(len(loader) / size)
    batches = [
        sampler.next_minibatch_ids(loader, SimpleNamespace(i=i))
        for i in range(calls)
    ]
    ids = [data_id for batch in batches for data_id in batch]
    rows.append((size, calls, len(ids), len(set(ids)), len(ids) - len(set(ids))))

print("size calls slots unique repeats")
for row in rows:
    print(*row)
```

Expected output:

```text
size calls slots unique repeats
1 7 7 7 0
3 3 9 7 2
5 2 10 7 3
10 1 10 7 3
```

These counts demonstrate coverage and padding only. They are not an optimization-quality, token-cost, or speed benchmark, and the last row shows why a batch larger than the dataset adds no new examples.

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
from gepa.optimize_anything import OptimizeAnythingConfig, optimize_anything

result = optimize_anything(
    ...,
    config=OptimizeAnythingConfig(
        engine="gepa",
        engine_config={
            "reflection": {
                "batch_sampler": MyBatchSampler(minibatch_size=5),
            },
        },
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
