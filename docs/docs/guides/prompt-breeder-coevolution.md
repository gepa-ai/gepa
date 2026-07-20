# PromptBreeder Reflection Co-Evolution

GEPA can evolve more than the visible task prompt. With PromptBreeder co-evolution enabled, every accepted GEPA candidate also carries a hidden reflection-process genome that describes how the reflection model should think about future mutations.

The two populations are coupled:

```text
GEPA task-prompt population
          ↕ fitness attribution
PromptBreeder reflection-genome population
```

The reflection model's weights are never trained. Instead, GEPA evolves the inherited prompt-level mutation policy supplied to that model.

## What Evolves

Each accepted task-prompt candidate is paired with a component-level reflection genome containing:

- mutation/reflection prompt
- thinking style
- genome identifier
- parent genome identifiers
- generation
- task lineage
- operator used to create the genome
- attempts, accepted children, and rejected children
- cumulative reward/fitness credit
- operator-level aggregate statistics

GEPA continues to own candidate acceptance, Pareto-front updates, minibatch and validation evaluation, evaluation budgets, and merge scheduling. PromptBreeder owns genome selection, zero-order mutation, hypermutation, lineage crossover, and mutator credit assignment.

## Fitness Attribution

PromptBreeder genomes do not become successful merely because an LLM produced text.

The fitness loop is:

1. GEPA selects a parent task prompt.
2. PromptBreeder selects the inherited reflection genome for each component.
3. A mutation operator proposes a trial reflection genome and then a trial task-prompt child.
4. GEPA evaluates the task-prompt child and decides whether to accept it.
5. Only accepted children promote their trial genomes into the breeder population.

That means:

- accepted child: the trial genome is assigned to the new GEPA candidate, receives creation reward, and becomes available for future breeding
- rejected child: the trial genome is discarded, but the parent mutator still receives rejection statistics and reward feedback
- merge child: the imported candidate inherits the strongest relevant parent genome per component

## Operators

### Zero-order mutation

The current genome directly rewrites the task prompt using execution traces, evaluator feedback, the inherited mutation prompt, and the current task prompt.

### Hypermutation

The reflection model first rewrites its own mutation prompt and thinking style. That evolved reflection genome is then used to generate the task-prompt child.

### Lineage mutation

PromptBreeder samples a successful compatible genome from the accepted population, crosses it with the selected parent's genome, and uses the descendant reflection process to create the next task-prompt child.

## API Usage

Enable the feature through either public entry point.

### `gepa.optimize(...)`

```python
import gepa

result = gepa.optimize(
    seed_candidate={"system_prompt": "Start conservative and mention ambiguity explicitly."},
    trainset=trainset,
    valset=valset,
    task_lm="openai/gpt-4.1-mini",
    reflection_lm="openai/gpt-4.1",
    prompt_breeder_config=gepa.PromptBreederConfig(
        operator_weights={
            "zero_order": 0.4,
            "hypermutation": 0.3,
            "lineage": 0.3,
        },
        lineage_size=5,
        seed=0,
    ),
    max_metric_calls=200,
)
```

### `optimize_anything(...)`

```python
from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything

result = optimize_anything(
    seed_candidate="Respond cautiously and ask for clarification on ambiguity.",
    evaluator=my_evaluator,
    config=GEPAConfig(
        engine=EngineConfig(max_metric_calls=200),
        reflection=ReflectionConfig(
            reflection_lm="openai/gpt-4.1",
            prompt_breeder_config=gepa.PromptBreederConfig(seed=0),
        ),
    ),
)
```

### Advanced manual construction

```python
strategy = gepa.make_prompt_breeder_strategy(
    "openai/gpt-4.1",
    config=gepa.PromptBreederConfig(seed=7),
)
```

## Configuration Rules

`prompt_breeder_config` cannot be combined with:

- `reflection_strategy`
- `custom_candidate_proposer`
- an adapter that already owns `propose_new_texts`

These conflicts raise immediately instead of silently disabling PromptBreeder.

## Checkpoint and Resume

PromptBreeder state is stored in `GEPAState.proposer_state` and survives checkpoint save/load. The checkpoint contains:

- accepted genomes
- candidate-index-to-genome assignments
- candidate-hash-to-genome assignments used during reflection
- operator statistics
- recent breeder history
- breeder RNG state for deterministic resume

Older GEPA checkpoints that do not contain proposer state still load correctly. In that case, PromptBreeder starts from backward-compatible defaults.

## Metadata and Analysis

PromptBreeder stores diagnostic details in `CandidateProposal.metadata` under the `promptbreeder:` namespace without disturbing GEPA's existing `prompt:` and `raw_lm_output:` keys.

Useful fields include:

- trial genome ids and parent genome ids
- selected operator
- mutation prompt and thinking style
- lineage ancestry
- hypermutation/crossover prompts and raw outputs
- accepted or rejected outcome
- reward credited after the engine decision

You can also inspect strategy-level diagnostics with:

```python
strategy.diagnostics()
```

This summarizes the elite genomes, operator statistics, assignment counts, and recent breeder history.

## Example

The ambiguity-resolution example lives at [`examples/prompt_breeder_coevolution/main.py`](/Users/bpeinado/Desktop/crizm/gebr/examples/prompt_breeder_coevolution/main.py).

## Known Limitations

- Reflection genomes are hidden optimizer state, not user-visible candidate components.
- PromptBreeder state is sequential by design; batched reflection jobs still consume a deterministic ordered genome stream.
- Merge inheritance currently picks the fittest available parent genome per component rather than synthesizing a fresh merge-specific genome.
- Fitness is still downstream-task based, so noisy minibatch or sparse validation feedback can slow breeder adaptation.
