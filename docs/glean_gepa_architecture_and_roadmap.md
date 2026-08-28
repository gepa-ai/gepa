# Glean GEPA: architecture, recent work, and roadmap

**Status:** working design document

**Last updated:** 2026-08-28

## Executive summary

We no longer use GEPA's public `optimize()` API. Our Glean path configures `GEPAEngine` directly and supplies its own adapters, proposer, data schedule, and screening policy. GEPA still provides candidate state, validation, budgets, persistence, lineage, and frontier updates. 

The roadmap is to:

1. support teacher/student tool-alignment optimization, with correctness as a validation metric;
2. add latency, completeness, and formatting objectives to both single mode and teacher/student mode adapters;
3. expose custom objectives, weights, frontiers, eval sets, and extension hooks through a CLI;
4. expand candidates from one prompt module to multiple SCs; and
5. eventually optimize repository-backed harness changes.

## Design choices from the past month

### 1. Use a custom Glean GEPA engine path

`src/glean_gepa/api.py` constructs `GEPAEngine` directly. This lets us own parent selection, reflection, multi-child generation, and high-signal screening without forking GEPA's core engine.

### 2. Keep two adapters because their data types differ

Teacher/student and single-model optimization have the same high-level shape, but different data contracts.

We will limit our adapters to these two unless we see a use case for meaningfully different training datatype.

### 3. Cache work at several layers

We cache eval-run IDs, Judge and shell analysis, enriched trace evidence, generated children, and GEPA state. Child cache keys include the root candidate and training slice, so a resumed run does not restart reflection and evaluation from the seed.

### 4. Progressively reveal training data and freeze validation

This is iterative search, not gradient descent. Each generation sees a new training eval-set version instead of repeatedly reflecting over all training data. Validation versions and scoring rules are fixed for the whole run and never used for reflection.

The current policy does not accumulate earlier training slices for speed. We can revisit that tradeoff if more context proves useful.

### 5. Use a Glean-specific proposer

More later.

### 6. Treat reflection-input selection as a policy

The current single-model path selects deduplicated error examples. Other clients may need a different sampling, grouping, or prioritization rule. **Reflection selection and prompt construction therefore need separate extension points.**

### 7. Screen on motivating failures

Original GEPA screens a mutation on a random training minibatch. We screen a child on the high-signal failures that motivated it and only continue when it fixes at least half of them. This spends full validation on changes that address the diagnosed problem.

## Current architecture

```text
runner.py
  configuration and adapter choice
       |
       v
adapter
  candidate -> eval run -> scores and traces
       |
       v
EvolutionaryProposer
  select evidence -> reflect -> create and screen children
       |
       v
GEPAEngine (unchanged)
  validate -> persist state -> update frontier
```

The responsibilities are:

- **Runner:** loads the experiment and chooses an adapter.
- **Adapter:** translates between Glean eval data and GEPA scores and traces.
- **Proposer:** selects reflection evidence, creates children, and applies the high-signal gate.
- **Engine:** owns budgets, validation, lineage, persistence, and the frontier.

GEPA does not import `glean_gepa`.

### Search loop

Each generation:

1. Reveal one unseen training version.
2. Select a frontier candidate and its reflection examples.
3. Generate or retrieve cached children.
4. Screen each child on the motivating failures.
5. Validate passing children on the frozen validation set and update state and the frontier.

If every child fails screening, the frontier remains unchanged and the next generation uses the existing candidate pool.

## Relationship to original GEPA

### What we keep

- `GEPAEngine`, `GEPAState`, and `GEPAResult`;
- named text components and candidate lineage;
- adapter-owned execution and traces;
- validation, budgets, persistence, and frontier state;
- higher-is-better objective semantics; and
- strategy boundaries for evaluation and acceptance.

### Where we deviate

- A Glean data item represents a remote eval-set run.
- Our proposer owns parent choice, reflection, and multi-child generation (all customizable policies for future clients).
- Training data is progressively revealed while validation remains fixed.
- Reflection examples are selected by domain-specific policies.
- Screening uses motivating failures rather than a random minibatch.
- Every child that passes screening may proceed to validation.

GEPA's current objective frontier retains the best candidate for each objective; it is not a full non-dominated-vector frontier. Custom dominance or weighted selection will be an explicit policy.

## Roadmap

### Phase 0 (9/1): productionize the current flow in single model mode that minimizes shell errors (Cathy) 
- Currently it times out when eval runs are slow and the quality of iterations could be better.

### Phase 0 (9/4): support teacher/student mode tool alignment (Grace)

- Optimize tool alignment.
- Keep correctness as a held-out validation gate.
- Build reflection examples from paired traces.
- Define missing-trace and valid-alternative-tool behavior.

### Phase 2 (9/4): expand the frontier for both adapters

- Add objective definitions with direction, weight, and normalization.
- Add latency and completeness; add formatting through the same contract.
- Store both raw metrics and normalized frontier values with more complex frontier.
- Keep correctness as a validation metric rather than hiding it in a weighted score.

### Phase 3 (9/15): client CLI and contracts (will iteratively update as we get here). 

The CLI will provide a quick way for anyone to use our engine for their custom usecase as we've added integration with eval and bigquery clients. 

The CLI will expose:

```text
glean-gepa validate experiment.yaml
glean-gepa run experiment.yaml
glean-gepa resume RUN_DIR
glean-gepa inspect RUN_DIR
```

The experiment file holds serializable configuration:

```yaml
schema_version: 1
adapter: teacher_student
candidate:
  codec: glean_gepa.candidates:NamedTextModules
  modules:
    WRITING_CODE: seeds/writing_code.txt
data:
  train_evalsets: [train-v1, train-v2]
  validation_evalsets: [validation-v1]
objectives:
  - name: tool_alignment
    direction: maximize
    role: frontier
    weight: 1.0
  - name: correctness
    direction: maximize
    role: validation
    threshold: 0.95
frontier:
  policy: my_package.frontier:CustomFrontier
extensions:
  reflection:
    WRITING_CODE:
      selector: my_package.reflection:SelectErrors
      policy: my_package.reflection:WritingCodePolicy
  candidate_integrator: my_package.integration:PromptIntegrator
  scoring_policy: my_package.scoring:CustomScoring
```

The schema covers the adapter, candidate modules, custom train and validation eval sets, objectives, score weights, validation thresholds, frontier policy, reflection behavior, integration hook, and run settings.

Behavior that cannot be expressed as data uses importable hooks:

```python
class CandidateCodec(Protocol):
    def load(self, spec: Mapping[str, Any]) -> Candidate: ...
    def modules(self, candidate: Candidate) -> Mapping[str, str]: ...
    def replace(self, candidate: Candidate, module: str, value: str) -> Candidate: ...


class CandidateIntegrator(Protocol):
    def compile(self, candidate: Candidate, context: EvalContext) -> EvalRunSpec: ...


class ReflectionSelector(Protocol):
    def select(self, module: str, traces: list[Trajectory]) -> list[ReflectiveExample]: ...


class ReflectionPolicy(Protocol):
    def build_prompt(self, module: str, examples: list[ReflectiveExample]) -> str: ...


class ScoringPolicy(Protocol):
    def score(self, batch: EvaluationBatch) -> ScoreDecision: ...


class FrontierPolicy(Protocol):
    def update(self, state: FrontierState, candidate: ScoredCandidate) -> FrontierDecision: ...
```

The CLI validates the schema and resolves the hooks before remote work. Candidate compilation, scoring, and reflection do not live in argument parsing.

**Outcome:** a client can supply custom eval sets, weights, frontier behavior, reflection selection, prompts, and eval integration without changing the engine.

### Phase 4: multi-SC candidates - this should be a quick followup from phase 3. 

- Treat each Search Configuration as a named candidate module.
- Validate seeds against the declared module schema.
- Compile each variant through the configured integrator.
- Record the changed module and compiled eval override.

### Phase 5: repository-aware harness evolution

- Replace text-only candidates with a `CandidateArtifact` contract: stable ID, base revision, serialization, materialization, and provenance.
- Let the proposal system inspect trace feedback and the Scio repository, then create an isolated child patch.

**Outcome:** the optimizer can evolve the full harness, not only prompt text.
