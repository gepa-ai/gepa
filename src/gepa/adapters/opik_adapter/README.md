# OpikAdapter

Native GEPA adapter for [Comet Opik](https://www.comet.com/docs/opik/) datasets and metrics. Optimize a system prompt against an Opik `Dataset` using any Opik (or Opik-style) metric — the metric's `reason` becomes the natural-language feedback GEPA's reflection LM reads.

This is the GEPA-side counterpart to Opik's [`GepaOptimizer`](https://www.comet.com/docs/opik/development/optimization-runs/algorithms/gepa_optimizer). The optimizer-side wrapper lives in `opik-optimizer`; this adapter lets you skip that layer and run GEPA directly against an Opik dataset + metric.

## Install

```bash
pip install "gepa[opik]"
```

The `opik` extra installs `opik` (the lightweight SDK) and `litellm`. It does **not** install `opik-optimizer` — you don't need it.

## Quickstart

```python
import opik
from opik.evaluation.metrics import LevenshteinRatio

from gepa import optimize
from gepa.adapters.opik_adapter import OpikAdapter, opik_dataset_to_examples

# 1. An Opik dataset of items shaped like {"input": str, "label": str, ...}.
dataset = opik.Dataset(name="capitals")
dataset.insert([
    {"input": "Capital of France?",  "label": "Paris"},
    {"input": "Capital of Germany?", "label": "Berlin"},
    {"input": "Capital of Japan?",   "label": "Tokyo"},
])

# 2. An Opik-style metric. Anything with .value and .reason works.
def metric(item, llm_output):
    return LevenshteinRatio().score(reference=item["label"], output=llm_output)

# 3. Wire up the adapter and run GEPA.
adapter = OpikAdapter(
    model="openai/gpt-4o-mini",
    metric=metric,
    input_field="input",
)

examples = opik_dataset_to_examples(dataset)
result = optimize(
    seed_candidate={"system_prompt": "Answer concisely."},
    trainset=examples,
    valset=examples,
    adapter=adapter,
    reflection_lm="openai/gpt-5",
    max_metric_calls=50,
)
print(result.best_candidate["system_prompt"])
```

## Why a native adapter?

The Opik docs describe a `GepaOptimizer` class that wraps GEPA from the Opik side. That works but requires the `opik-optimizer` package and routes calls through its evaluator. If you're already using GEPA's `gepa.optimize` API, the `OpikAdapter` is a thinner integration:

- Lightweight dependency: only `opik` itself.
- Lazy imports: nothing fails at module load if Opik isn't configured; the SDK is only touched when you actually pass an `opik.Dataset` to `opik_dataset_to_examples`.
- Native GEPA shape: works with all `gepa.optimize` knobs (Pareto selection, merge proposer, custom proposers, multi-component candidates).
- Provider-agnostic via litellm — same `openai:gpt-...` / `anthropic:claude-...` strings as the rest of GEPA.

## Metric contract

The `metric` callable receives `(dataset_item, llm_output)` and may return:

1. An object with `.value: float` and optional `.reason: str` — the standard Opik `ScoreResult`. Any duck-typed object with this surface also works.
2. A `(score, feedback)` tuple — convenient for ad-hoc metrics.
3. A bare `float` — feedback defaults to an empty string.

The feedback (from `.reason` or the tuple's second element) is what GEPA's reflection LM uses to diagnose failures and propose improvements — so the richer the metric's `reason` field, the faster optimization converges.

## Custom chat backend

Pass a callable instead of a litellm string if you have your own chat model:

```python
def my_chat(messages):
    # `messages` is a list of {"role": ..., "content": ...}
    return my_client.complete(messages).text

adapter = OpikAdapter(model=my_chat, metric=metric, input_field="input")
```

## What gets logged to Opik?

Right now: nothing automatic. Use `opik.track` or `opik.opik_context` around your metric or candidate evaluation if you want per-rollout traces in your Opik project. This adapter focuses on the optimization loop; trace logging is left as a separate, optional concern so the adapter works without Opik credentials configured.
