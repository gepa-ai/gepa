# Sequential evidence for noisy evaluations

`gepa.optimize` can collect fresh parent/child repetitions before applying its
normal reflective acceptance criterion. This is useful when one noisy rollout
can change which candidate enters the pool. The feature is opt-in:

```python
from gepa import SequentialEvidenceConfig, optimize

result = optimize(
    seed_candidate={"instructions": "..."},
    trainset=train_examples,
    valset=validation_examples,
    adapter=adapter,
    reflection_lm=reflection_lm,
    max_metric_calls=2000,
    run_dir="run-with-evidence",
    sequential_evidence=SequentialEvidenceConfig(
        evaluator_fingerprint="metric-v3/model-v2/environment-v1/data-v4",
        score_min=0.0,
        score_max=1.0,
        max_comparison_calls=400,
        max_calls_per_example=1,
        min_rounds=2,
        effect_margin=0.02,
        alpha=0.05,
    ),
)
print(result.sequential_evidence)
```

The caller must choose the fingerprint to identify all evaluator, model,
environment and dataset versions that could change score distributions. The
controller also fingerprints the fixed ordered data and IDs. This first release
requires JSON-serializable examples and integer or string IDs. Opaque objects,
mutable datasets, merge proposals, custom candidate proposers, custom proposal
strategies and custom validation policies are unsupported. Use the exact native
`FullEvaluationPolicy` and native reflective sampling/selection components.
Adapter-owned `propose_new_texts` and the native stateless reflection path are
supported; scorer work belongs exclusively in evaluation methods.

## What the decision means

Each proposal freezes its ordered minibatch IDs before observations. Each distinct ID is
a stratum. Native epoch padding can repeat an ID: those positions share one
fresh observation per side and round, with their positional weights added.
All original positions remain in the report. Padding never creates extra
independent samples or silently removes its weight from the target mean.
A complete round freshly evaluates both candidates on every distinct stratum.
Uniform positional weights are the default; `stratum_weights=(...)` can fix a
different positive distribution in proposal position order. Weights must sum to
one and match the native positional minibatch size, checked before seed work.

Ordinary proposal scores remain available to the existing criterion and
reflection machinery. They do not become independent evidence. Repetitions
bypass the evaluation cache, retain individual score identities, and do not
invent trajectories or unpack an aggregate adapter score into multiple samples.
The scalar scores in `val_aggregate_scores` keep their existing native meanings.
They are **not** confidence-calibrated estimates or a statistically calibrated
Pareto frontier.

The default `assumption="bounded_independent"` requires a fixed per-side,
per-stratum mean and independent bounded repetitions, conditionally on the
proposal history. `"bounded_martingale"` permits dependence when every repetition
has that same fixed conditional mean. The controller checks bounds and
provenance; it cannot establish stationarity or independence from score values.
Drift, correlated hidden retries and selective missing outcomes can violate
these assumptions. `assumption="heuristic"` retains budgeted collection and
explicitly makes no confidence-supported acceptance claim.

For comparison index `c >= 1`, stratum `s` among `S` strata, side and sample count
`n >= 1`, the two-sided interval spends

```text
delta(c,s,n) = alpha / (2 S c(c+1) n(n+1))
radius       = (score_max-score_min) sqrt(log(2/delta) / (2 n))
```

Both reciprocal series telescope to one. A union bound over comparisons,
strata, sides and counts spends at most `alpha` for the whole run under the
stated assumptions. The implementation computes the allocation in log space,
intersects successive per-stratum intervals, and aggregates fixed weighted
child-minus-parent intervals. Moments use normalized units internally to avoid
overflow; reported bounds use raw metric units. The rule uses the established
[Hoeffding bound](https://doi.org/10.1080/01621459.1963.10500830) plus a union
bound, without claiming a new theorem or a paired-variance improvement.

After `min_rounds`, a lower difference bound strictly above `effect_margin`
supports improvement. An upper bound strictly below `-effect_margin` supports
regression. Ties and ambiguous comparisons collect further complete rounds
until a configured cap prevents another round. Conservative intervals can use
many repetitions or remain unresolved, especially for small effects.

A scorer exception, absent score, invalid score, or interrupted round makes the
comparison permanently `inconclusive`. Partial observations remain diagnostics;
they never enter the learned moments. The controller does not discard failed
rounds and retry until a successful-only comparison appears significant. A
numeric task-failure score is an ordinary outcome only when it is genuinely
part of the declared bounded metric; infrastructure failures must be signaled.

## Budget and adapter contract

One reservation owner guards native seed evaluation, proposal evaluations,
evidence repetitions and final validation. `max_metric_calls` is mandatory and
finite in evidence mode. Reflection LM costs use their existing separate limit.

Before each external call, the controller reserves
`len(batch) * max_calls_per_example`. The adapter must guarantee this maximum
includes all scorer attempts, including internal retries and failures. A
successful `num_metric_calls`, when supplied, must be an integer between
`len(batch)` and that reserved maximum. Without it, cost is `len(batch)`.
Unsupported or dishonest hidden evaluator work cannot be intercepted inside an
adapter; a declared maximum is a caller contract, not provider-side enforcement.
A reported violation fails closed and charges the reserved maximum.

Failed calls are conservatively charged at their reserved maximum. Native
post-dispatch `increment_evals` reconciles the single guard instead of charging
again, including the seed evaluated before native state exists. Batch preflight
prevents partial native proposal batches on budget shortage. Evaluation calls
are serial in this initial release, even when an adapter exposes
`batch_evaluate`; proposal generation retains its native behavior.

Supported improvement still requires the original acceptance criterion and
complete native validation. If complete validation cannot fit the remaining
budget, the decision becomes `validation_deferred`, optimization stops, and the
result contains the last fully validated pool. No partial validation scores
enter that pool. Resume with an explicitly increased `max_metric_calls` and the
same context to finish deferred validation. Stop callbacks remain authoritative.

Custom `selection_strategy` remains authoritative and may deliberately choose a
proposal that fails the composed criterion. Such a choice is reported as
`unresolved_exploration`, then `exploration` after complete validation. Setting
`unresolved="explore"` likewise permits budget-exhausted/heuristic candidates
through the original criterion; it does not relabel them statistically
supported. Stateful acceptance criteria execute at most once per proposal.
Selection callbacks and other user hooks must not perform hidden scorer work.

Adapters may optionally implement:

```python
def evaluate_replication(self, batch, candidate, *, trial_context, capture_traces):
    # trial_context contains comparison/round/trial IDs, side, example IDs,
    # and a shared paired_seed. Consume it only if the evaluator supports it.
    return self.evaluate(batch, candidate, capture_traces=capture_traces)
```

The default calls `evaluate` afresh for each side. A supplied shared seed is not
proof that the evaluator actually uses common randomness; the current bound
claims no paired-randomness efficiency benefit either way.

## Checkpoints and inspection

Keep `gepa_state.bin` and `sequential_evidence.json` together. The native
checkpoint contains controller snapshots, the sampler and shared RNG state.
The atomic checksummed sidecar records reservations before external dispatch,
including seed work before the native state exists. It retains individual
scores and status/provenance records, but no repetition trajectories. Storage
grows with attempted evidence and comparisons; large-budget snapshot overhead
should be measured for the workload.

On process failure, a pending external call has unknown completion. Resume marks
it abandoned, charges its reserved maximum, and makes its interrupted comparison
inconclusive. It does not promise exactly-once model calls. A failed seed before
native state creation may need a new full seed evaluation, within the remaining
cap. A selected candidate whose complete pool write already committed is
reconciled by content identity instead of inserted twice.

Resume verifies fingerprints, allocation order, moments and intervals rebuilt
from complete observations, execution costs, native sampler configuration and
RNG sharing. Changed context or a decreased budget is rejected. Use a new run
directory to reset evidence explicitly. Legacy checkpoints can start fresh
evidence without inventing historical repetitions; their old metric costs are
preserved. Legacy mode remains unchanged when this option is absent. An evidence
run cannot silently resume with the option disabled.

Inspect `result.sequential_evidence` for statuses, reasons, raw-unit bounds,
completed rounds, observation counts, support assumptions, metric calls and
validation indices. It round-trips through `GEPAResult.to_dict/from_dict`; legacy
result JSON omits the optional field. An optional
`on_sequential_evidence(SequentialEvidenceEvent)` callback observes comparison,
deferral and promotion transitions. Its decision payload is a copy.

Use one active optimizer per run directory. User-owned adapter/selection state
outside native checkpoint hooks remains the caller's responsibility. No
real-agent quality, cost advantage or SOTA claim follows from the mathematical
contract or from scripted integration tests.
