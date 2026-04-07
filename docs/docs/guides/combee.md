# ComBEE: Scalable Parallel Prompt Learning

ComBEE ([arXiv:2604.04247](https://arxiv.org/abs/2604.04247)) is a framework for scaling GEPA's reflection step to large batch sizes without quality degradation. Enable it with `use_combee=True`.

---

## The Problem: Context Overload

GEPA's standard reflection step feeds all minibatch traces to the reflection LM in a single call. This works well at small batch sizes, but degrades sharply as `reflection_minibatch_size` grows:

| Batch size | Formula accuracy | FiNER accuracy |
|---|---|---|
| 1 | 87.0% | 76.0% |
| 10 | ~83% | ~74% |
| 100 | **72.5%** | **70.6%** |

The aggregator LLM is not truncating — it retains broad, generic patterns while **discarding the specific, high-value insights** that drive downstream accuracy. The ComBEE paper calls this *context overload*.

---

## How ComBEE Works: Map-Shuffle-Reduce

ComBEE replaces the single reflection call with a three-phase pipeline:

```
n traces → [Augmented Shuffle] → k groups → [Level-1: k LM calls] → k proposals → [Level-2: 1 LM call] → final instruction
```

### 1. Augmented Shuffle (§3.2)

Each reflection is duplicated `p=2` times and the full augmented set is shuffled before being distributed across groups. This gives every reflection multiple chances to be incorporated, improving robustness even at large batch sizes.

### 2. Level-1 — Map (§3.1)

The augmented set (`p·n` items) is split into `k = ⌊√n⌋` groups. One reflection LM call per group produces `k` intermediate instruction proposals. Each group sees `p·n/k ≈ p·√n` traces — a manageable context even when `n` is large.

With `n=40`: `k = ⌊√40⌋ = 6` groups, each seeing ~13 traces instead of 40.

### 3. Level-2 — Reduce (§3.1)

A second LM call aggregates the `k` intermediate proposals into a single final context update. This two-level design keeps each LM call well within its effective reasoning range, empirically shown to preserve significantly more information than naive aggregation.

The choice `k = ⌊√n⌋` balances both levels: Level-1 processes `√n` traces per group, Level-2 aggregates `√n` proposals — both at the same scale.

---

## Results

On AppWorld (ReAct agent), ComBEE at batch size 40 achieves the **highest average score** across all methods, with a **12× training speedup** over the sequential baseline at comparable cost — while naive batch-40 degrades to barely above the no-learning baseline.

On Terminal-Bench 2.0, ComBEE at batch size 30 recovers most of the sequential quality with a **17× speedup**. A key indicator: ComBEE's learned playbook retains 6,887 tokens vs. only 526 for naive batch-40, confirming that parallel scan aggregation preserves far more information.

---

## Usage

!!! warning "Set `reflection_minibatch_size` to at least 4"
    GEPA's default `reflection_minibatch_size` is 3. With `n=3`, `k=⌊√3⌋=1` — ComBEE silently falls back to a single LM call (standard GEPA). Set `reflection_minibatch_size` to **20 or more** for meaningful speedups.

### `gepa.optimize`

```python
import gepa

result = gepa.optimize(
    seed_candidate={"system_prompt": "You are a helpful assistant."},
    trainset=trainset,
    valset=valset,
    task_lm="openai/gpt-4.1-mini",
    reflection_lm="openai/gpt-5.1",
    reflection_minibatch_size=40,  # n — ComBEE forms k=6 groups automatically
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
| `reflection_minibatch_size` ≤ 3 (default) | No effect; ComBEE falls back to standard GEPA |
| Small batches (n < 9) | ComBEE activates with k=2 groups; modest benefit |
| Medium batches (n ≥ 9) | ComBEE recommended; clear quality improvement |
| Large batches (n ≥ 20) | ComBEE strongly recommended; prevents context overload |
| Parallel agent deployments with many traces | ComBEE is designed for exactly this setting |

---

## Reference

> *Combee: Scaling Prompt Learning for Self-Improving Language Model Agents*
> Hanchen Li, Runyuan He, Qizheng Zhang, et al. (2026)
> [https://arxiv.org/abs/2604.04247](https://arxiv.org/abs/2604.04247)
