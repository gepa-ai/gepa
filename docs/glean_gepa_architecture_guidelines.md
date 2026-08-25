# Glean GEPA architecture guidelines

This is a short handoff guide for the `glean_gepa` integration.

## Target architecture

```text
CLI / experiment setup (`runner.py`)
        |
        v
GEPA engine wiring (`api.py`)
        |
        +--> EvolutionaryProposer
        |      selects parents, requests reflection, creates and screens children
        |
        +--> TeacherStudentAdapter
        |      evaluates student vs. teacher through EvalCLI + Judge
        |      objectives: correctness, tool alignment, grounding
        |
        +--> SingleModelAdapter
               evaluates one student through EvalCLI + BigQuery
               objective: shell_success_rate
               optionally creates a fresh replay eval set per candidate

Shared infrastructure (not adapters)
        - prompt.py: compile a candidate into an override
        - ALRunner: start/reuse EvalCLI runs
        - EvalCLI / BigQuery clients
        - cache records and cache persistence
        - common candidate, output, trajectory, and reflective-example types
        - shared reflection-dataset helpers only where their behavior is truly identical
```

Both adapters satisfy the same GEPA-facing contract: candidates are `dict[str, str]`; evaluations return `GleanEvaluationBatch`; and reflection is based on captured trajectories. The GEPA engine and `EvolutionaryProposer` must not branch on evaluation mode.

## Responsibility boundaries

### `TeacherStudentAdapter`

- Requires `ALRunner`, `Judge`, teacher/student model settings, and teacher/student cache state.
- Runs or reuses teacher and student evaluations, triggers the judge, and turns judged traces into per-entry trajectories.
- Owns correctness, tool-alignment, grounding, token, loop, and tool-error objective construction.
- Does not import BigQuery or know how shell-error analyses are queried.

### `SingleModelAdapter`

- Requires `ALRunner`, `BigQueryClient`, student model settings, and shell-analysis cache state.
- Runs or reuses a student evaluation, fetches per-entry shell-error metrics, and creates trajectories from high-signal failing entries.
- Owns the `shell_success_rate` objective, fresh-eval-set creation policy, and shell-error diagnostics passed to reflection.
- Does not create teacher runs, trigger judges, or emit teacher comparison fields.

### Shared code

- `ALRunner` is the only place that knows how to invoke and wait for EvalCLI evaluation runs.
- `prompt.py` is the only place that knows the prompt-override encoding.
- `fresh_evalset.py` is a service helper used by `SingleModelAdapter`, not an adapter itself.
- Cache serialization should use separate namespaces/records for run IDs, judge state, and shell analysis. Do not make one adapter deserialize fields owned by the other.
- Keep `Candidate`, `ModuleSpec`, prompt-budget helpers, and generic reflective-example formatting in neutral modules, not in either adapter file.

## What is already in place

- `src/glean_gepa/api.py` wires a Glean-specific proposer into the low-level `GEPAEngine`.
- `src/glean_gepa/evolutionary_proposer.py` handles frontier-parent selection, reflection-driven mutations, prompt-budget filtering, and child screening.
- `TeacherStudentAdapter` and `SingleModelAdapter` provide the two evaluation paths: teacher/student judging and single-model shell reliability.
- `ALRunner`, `Judge`, `shell_tool_error_util.py`, `fresh_evalset.py`, prompt compilation, and versioned cache serialization already provide most of the supporting pieces.

Keep evaluation behavior stable while changing the surrounding code. Do not simultaneously change scoring, candidate selection, or remote-evaluation semantics.

## Implementation rules

1. Select the concrete adapter explicitly in `runner.py`; keep each adapter free of branches for the other evaluation path.
2. Use a small protocol/base type only for the methods the proposer actually calls: `evaluate`, `make_reflective_dataset`, and `propose_new_texts`. Prefer duplicated small methods over a large inheritance hierarchy.
3. Each adapter's constructor must require only its own dependencies. Invalid combinations should be impossible to construct.
4. Keep objective names and score direction stable: all GEPA scores are higher-is-better; shell error rate enters GEPA as `shell_success_rate`.
5. Preserve the distinction between a selection score and reflection diagnostics. Scores choose candidates; traces, error strings, and per-entry data explain what to edit.
6. Cache keys must include every result-changing input: eval-set identity/version, model, prompt hash, and run label. Keep fresh eval-set cache behavior explicit.
7. Every extraction step gets characterization tests before cleanup. Remote EvalCLI/BigQuery runs are smoke tests, not unit tests.

## Suggested ownership split

The junior collaborator can own the **SingleModelAdapter vertical slice**—about one-third of the overall work—because it has a clear product boundary and can be tested with fakes. You retain the more coupled optimization and judge-comparison behavior.

### Her ownership: shell reliability (~1/3)

- Move shell-analysis cache read/write and cache-migration tests into its ownership boundary.
- Own `shell_tool_error_util.py`: classification edge cases, per-entry aggregation, query-builder tests, and diagnostic summaries.
- Own fresh eval-set behavior: replayability filtering, metadata, idempotency/cleanup policy, and EvalCLI-client fakes.
- Add local fixtures and a documented small shell-reliability smoke command.

### Your ownership: search and judged quality

- Extract/own `TeacherStudentAdapter`, including judge orchestration and judged-trace scoring.
- Own `EvolutionaryProposer`, module-selection strategy, parent selection, child screening, and evaluation-budget accounting.
- Own the candidate/module contract, prompt compilation contract, and any changes to GEPA wiring.
- Decide cross-cutting experiment policy: model defaults, eval-set versions, objective weighting, and which results are comparable.

### Pair-review changes

- The small adapter protocol and shared typed output/trajectory models.
- Objective names or score-direction changes.
- Cache-key changes that affect experiment reuse.
- The first end-to-end smoke test for each adapter.

## First collaboration milestone

1. Add characterization tests for the two adapter paths.
2. Move common types/utilities out of adapter modules; do not change behavior.
3. Verify `SingleModelAdapter` with its unit suite and one smoke evaluation.
4. Verify `TeacherStudentAdapter` with the same level of coverage.
5. Keep adapter selection explicit in the CLI and GEPA wiring.

This sequencing keeps each PR small and makes behavior changes obvious instead of hiding them inside a large refactor.
