# ComBEE: Scalable Parallel Prompt Learning

ComBEE ([arXiv:2604.04247](https://arxiv.org/abs/2604.04247)) scales GEPA's
reflection step to large minibatches without quality degradation. It ships as
a [`ReflectionLM`](https://github.com/gepa-ai/gepa/issues/329) implementation:
pass `reflection_strategy=ComBEEReflectionLM(...)` to `gepa.optimize`, or set
`ReflectionConfig(reflection_strategy=...)` in `optimize_anything`.

## How ComBEE works: Map-Shuffle-Reduce

ComBEE replaces the single reflection call with a three-phase pipeline, per
component:

```
n traces → [Augmented Shuffle] → k groups → [Level-1: k LM calls] → k proposals → [Level-2: 1 LM call] → final instruction
```

### 1. Augmented Shuffle (§3.2)

Each reflection record is duplicated `p` times (`duplication_factor`, default
2) and the augmented set is shuffled with a seeded RNG before being
distributed across groups. Every record gets multiple chances to be
incorporated, improving robustness at large batch sizes — and runs stay
reproducible.

### 2. Level-1 — Map (§3.1)

The augmented set (`p·n` items) is split into `k = ⌊√n⌋` groups. One
reflection-LM call per group produces `k` intermediate instruction proposals.
Each group sees `p·n/k ≈ p·√n` traces — a manageable context even when `n` is
large. With `n=40`: `k = 6` groups, each seeing ~13 traces instead of 40.

### 3. Level-2 — Reduce (§3.1)

A final LM call synthesizes the `k` intermediate proposals into one
instruction (`aggregation_prompt_template`, customizable). The choice
`k = ⌊√n⌋` balances both levels: Level-1 processes `√n` traces per group,
Level-2 aggregates `√n` proposals — both at the same scale.

## Usage

!!! warning "Raise `reflection_minibatch_size`"
    GEPA's default `reflection_minibatch_size` is 3. With `n < 4`,
    `k = ⌊√n⌋ = 1` and ComBEE falls back to a standard single reflection call
    (it logs when this happens). Set `reflection_minibatch_size` to **20 or
    more** for meaningful benefit.

### `gepa.optimize`

```python
import gepa
from gepa.proposer.reflective_mutation.combee import ComBEEReflectionLM

result = gepa.optimize(
    seed_candidate={"system_prompt": "You are a helpful assistant."},
    trainset=trainset,
    valset=valset,
    task_lm="openai/gpt-4.1-mini",
    reflection_strategy=ComBEEReflectionLM("openai/gpt-5.1"),
    reflection_minibatch_size=40,  # n — ComBEE forms k=6 groups automatically
    max_metric_calls=600,
)
```

### `optimize_anything`

```python
from gepa.optimize_anything import GEPAConfig, ReflectionConfig, optimize_anything
from gepa.proposer.reflective_mutation.combee import ComBEEReflectionLM

result = optimize_anything(
    seed_candidate=seed,
    evaluator=my_evaluator,
    dataset=dataset,
    config=GEPAConfig(
        reflection=ReflectionConfig(
            reflection_strategy=ComBEEReflectionLM("openai/gpt-5.1"),
            reflection_minibatch_size=40,
        ),
    ),
)
```

### Options

```python
ComBEEReflectionLM(
    lm,                                # model name string, or any LanguageModel callable
    reflection_prompt_template=None,   # Level-1 template: str, or dict per component
    aggregation_prompt_template=None,  # Level-2 template (must contain <curr_param> and <side_info>)
    duplication_factor=2,              # p — augmented-shuffle duplication (§3.2)
    rng=random.Random(0),              # seeded shuffle for reproducible runs
    logger=None,
)
```

## Cost and observability

ComBEE makes **`k + 1` reflection-LM calls per component per proposal** (vs 1
for the default reflector). The per-call intermediates are recorded in
proposal metadata under `combee:`-namespaced keys — `combee:<comp>:k`,
`combee:<comp>:level1_prompts` / `level1_outputs`,
`combee:<comp>:num_lm_calls`, `combee:<comp>:mode`, and
`combee:total_lm_calls` — visible to `on_proposal_end` consumers and
experiment trackers. `total_cost` and token totals are exposed by delegation
to the wrapped LM, so `max_reflection_cost` works as a stop condition.

## Credits

ComBEE support was originally contributed by
[@nuglifeleoji](https://github.com/nuglifeleoji) in
[#307](https://github.com/gepa-ai/gepa/pull/307) and re-hosted onto the
`ReflectionLM` protocol introduced in
[#369](https://github.com/gepa-ai/gepa/pull/369).
