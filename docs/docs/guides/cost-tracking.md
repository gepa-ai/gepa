# Cost & Token Tracking

GEPA tracks cumulative cost and token usage on the reflection LM automatically. No configuration needed — every call through the LM accumulates metrics that you can read at any time.

---

## Reading cost and tokens

After optimization, read directly from the LM instance:

```python
import gepa
from gepa.lm import LM

reflection_lm = LM("openai/gpt-4.1")

result = gepa.optimize(
    seed_candidate={"instructions": "..."},
    trainset=trainset,
    reflection_lm=reflection_lm,
    max_metric_calls=500,
)

print(f"Total cost: ${reflection_lm.total_cost:.4f}")
print(f"Tokens in:  {reflection_lm.total_tokens_in:,}")
print(f"Tokens out: {reflection_lm.total_tokens_out:,}")
```

The same works with `optimize_anything`:

```python
import gepa.optimize_anything as oa
from gepa.lm import LM

lm = LM("openai/gpt-5.1")

result = oa.optimize_anything(
    seed_candidate="...",
    evaluator=my_evaluator,
    config=oa.GEPAConfig(
        reflection=oa.ReflectionConfig(reflection_lm=lm),
    ),
)

print(f"Total cost: ${lm.total_cost:.4f}")
```

---

## What gets tracked

| LM type | `total_cost` | `total_tokens_in` | `total_tokens_out` |
|---|---|---|---|
| String model name (e.g. `"openai/gpt-4.1"`) | Exact via `litellm.completion_cost` | Exact from API usage | Exact from API usage |
| `LM` instance passed directly | Exact | Exact | Exact |
| Plain callable (e.g. `lambda p: ...`) | `0.0` | Estimated (~4 chars/token) | Estimated (~4 chars/token) |

When you pass a string model name, GEPA creates an `LM` instance internally. When you pass a plain callable, GEPA wraps it in a `TrackingLM` that estimates tokens from string length. In both cases, the tracking attributes are available on the object GEPA actually uses.

!!! tip "Keep a reference to the LM"
    If you pass a string like `reflection_lm="openai/gpt-4.1"`, GEPA creates the `LM` internally and you won't have a reference to read cost from. To access cost after optimization, create the `LM` yourself and pass the instance.

---

## Stopping on cost budget

Use `max_reflection_cost` to stop optimization when the reflection LM's cumulative cost reaches a USD budget:

```python
result = gepa.optimize(
    ...,
    reflection_lm="openai/gpt-4.1",
    max_reflection_cost=5.00,  # stop after $5 of reflection calls
)
```

Or in `optimize_anything`:

```python
config = oa.GEPAConfig(
    engine=oa.EngineConfig(
        max_reflection_cost=10.00,
    ),
)
```

This reads `lm.total_cost` at each iteration. For plain callables (which report cost as `0.0`), the stopper never triggers — it only stops on costs the LM actually reports.

`max_reflection_cost` can be combined with `max_metric_calls` — optimization stops when either budget is reached first.
