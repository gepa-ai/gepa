# ComBEE: Scalable Parallel Prompt Learning

ComBEE ([paper](https://arxiv.org/abs/2505.03738)) is a Map-Shuffle-Reduce framework that prevents **context overload** when learning from large reflection batches. Enable it with `use_combee=True`.

---

## The Problem: Context Overload

GEPA's standard reflection step feeds all minibatch traces to the LM in a single call. This works well for small batches, but at large batch sizes (e.g. `reflection_minibatch_size=40`) the aggregator LLM becomes overwhelmed and discards specific, high-value insights in favour of generic patterns — degrading the quality of the proposed instruction update.

---

## How ComBEE Works

ComBEE replaces the single reflection call with a three-phase pipeline:

1. **Augmented Shuffle** — each reflection is duplicated `p=2` times and the full set is shuffled, giving every reflection multiple chances to be incorporated.
2. **Map (Level-1)** — the augmented set is split into `k = ⌊√n⌋` groups; one LM call per group produces `k` intermediate instruction proposals.
3. **Reduce (Level-2)** — a second LM call aggregates the `k` proposals into a single final context update.

With `n=40`, `k=⌊√40⌋=6`, so each group sees ~13 traces instead of 40 — well within the LM's effective reasoning range.

!!! note "Minimum batch size"
    GEPA's default `reflection_minibatch_size` is 3. With `n=3`, `k=⌊√3⌋=1` and ComBEE silently falls back to standard GEPA. **Set `reflection_minibatch_size` to at least 4** (and ideally 20+) for ComBEE to activate.

---

## Usage

### `gepa.optimize`

```python
import gepa

result = gepa.optimize(
    seed_candidate={"system_prompt": "You are a helpful assistant."},
    trainset=trainset,
    valset=valset,
    task_lm="openai/gpt-4.1-mini",
    reflection_lm="openai/gpt-5.1",
    reflection_minibatch_size=40,  # n — ComBEE forms k=6 groups
    use_combee=True,
)
```

### `optimize_anything`

```python
import gepa.optimize_anything as oa
from gepa.optimize_anything import optimize_anything, GEPAConfig, ReflectionConfig

result = optimize_anything(
    seed_candidate="<your initial artifact>",
    evaluator=my_evaluator,
    objective="Describe what you want to optimize for.",
    config=GEPAConfig(
        reflection=ReflectionConfig(
            reflection_minibatch_size=40,
            use_combee=True,
        )
    ),
)
```

---

## When to Use ComBEE

| Scenario | Recommendation |
|---|---|
| Small batches (`n < 4`) | Use standard GEPA; ComBEE has no effect |
| Medium batches (`4 ≤ n < 20`) | ComBEE activates but gains are modest |
| Large batches (`n ≥ 20`) | ComBEE recommended; prevents quality degradation |
| Parallel agent deployments | ComBEE is designed for this setting |

---

## Reference

> *Combee: Scaling Prompt Learning for Self-Improving Language Model Agents*  
> [https://arxiv.org/abs/2505.03738](https://arxiv.org/abs/2505.03738)
