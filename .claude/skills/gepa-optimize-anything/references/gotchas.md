# Gotchas and pitfalls

Read before any real run.

## 1. Reward hacking — GEPA exploits a weak proxy ruthlessly
If your score is gameable, GEPA *will* game it. Real example (KernelBench): a correctness-only score
drove GEPA to a prompt that tells the model to **wrap the reference implementation** — always
"correct", zero real work. Result: pass@1 ~0.87 but the actual goal (fast kernels) collapsed to ~base.
Mitigation: a **gated objective** — score 0 unless valid+correct, then increasing in the thing you
care about (see `writing_evaluators.md`). Sanity-check the *winning* candidate against your true goal,
not just the score.

## 2. Selection bias / winner's curse (generalization mode)
In generalization mode the chosen candidate is the **max** over many candidates scored on `valset`.
With noisy (especially N=1) scores over few examples, that max is optimistically inflated — it can sit
well above true generalization (e.g. a selected best of 0.87 with runner-up 0.60 and median 0.57).
This bias lives in **selection on `valset`**, so reduce it there:
- use **enough `valset` examples**, and **N>1 samples per eval** for stochastic models, to shrink the
  variance that inflates the max;
- treat `best_score` (the selection-set number) as optimistic.

`test_set` does **not** reduce this bias — it is reporting-only. But scoring the final candidate on a
held-out `test_set` gives you an *honest* number to report — `metadata["test_score"]` (average; also
`["test_scores"]` per-example) vs `["baseline_test_score"]` for the seed — separate from the inflated
selection score. (Note the exact keys: `test_score`/`test_scores`, **not** `test_score(s)`.)

## 3. Stochastic N=1 default
One eval call per (candidate, example). For a temperature>0 model that means single-sample scores,
which feeds the selection bias in #2. Average N samples inside `evaluate` (`writing_evaluators.md`).

## 4. Unbounded runs are only a warning
If neither `max_evals` nor `max_token_cost` is set, the run is unbounded and you get a
`warnings.warn`, not an error. Always set a budget explicitly.

## 5. `engine_config` typos are warn-and-dropped
Unknown keys (including a wrong-engine key after swapping `engine=`) are logged via
`warnings.warn` and ignored — easy to miss in a long run. Double-check keys against the engine's
`_<ENGINE>_CONFIG_KEYS` / core dataclasses. Swapping `engine=` requires swapping the `engine_config`
block too.

## 6. Agentic engines fail late
`autoresearch` / `meta_harness` `subprocess.Popen(["claude", ...])`; a missing/unauthenticated CLI
surfaces only after a subprocess error mid-run. Run `scripts/preflight.py` first.

## 7. The run won't terminate / spins to timeout (caching)
Evaluations are cached by default, so `max_evals` counts only cache *misses*. After the search
converges, the proposer keeps emitting candidates that hit the cache — `max_evals` stops depleting and
the run loops until your process times out, still spending (uncached) proposer-LLM calls. Always give
it a real stop condition: `stop_at_score` for a bounded metric (e.g. `1.0`), and/or `max_token_cost`
plus a wall-clock `timeout` on the launched process. Don't rely on `max_evals` alone to stop a run.

## 8. Pick the right mode
The mode is implicit in which sets you pass (`api.md`): no `dataset`/`valset` → single-task;
`dataset` only → multi-task; `dataset`+`valset` → generalization. A common mistake is wanting a
candidate that generalizes but passing only `dataset` (multi-task) — then there's no held-out
selection and the candidate can overfit the training examples. Pass a `valset` for generalization.

## 9. Saturated signal — GEPA returns the seed unchanged (no failure to learn from)
GEPA improves by reflecting on examples the current candidate gets **wrong**. If the seed already
scores near-perfect on the train minibatch the proposer reflects over, every mutation looks "not
better" and is rejected by the acceptance gate — so GEPA runs its full budget, accepts **zero**
proposals, and hands back the seed. Symptom: many proposals made, ~0 accepted, `best_candidate` ==
seed. This is "baseline saturation," not a bug: there's no gradient for reflection to climb.
Mitigations: make sure `dataset` contains examples the seed actually **fails** (harder/larger train
set, or a task-LM weak enough to generate failures); if the seed is already at ceiling, GEPA has
little to do — `best_of_n` (sample full rewrites) or just keeping the seed may be the better call.
Always check **proposals accepted**, not just proposals made.

## Quick pre-flight checklist
- [ ] mode matches intent (single-task / multi-task / generalization) via `dataset`/`valset`
- [ ] `dataset` contains examples the seed gets WRONG (failure signal) — else GEPA can't improve (#9)
- [ ] `score` matches the real goal (gated, not gameable)
- [ ] `info` carries actionable feedback (errors/diffs/partial credit)
- [ ] N>1 if the model is stochastic
- [ ] proposer LM valid + 1-call tested; creds present (or a custom LM-protocol callable)
- [ ] `max_evals` sized for many proposals (≳ 15–20 × the size of the selection set: `len(valset)` in
      generalization, `len(dataset)` in multi-task, or just ≳ 15–20 in single-task), not just one — see SKILL.md
- [ ] `run_dir` + `output_dir` set (so artifacts persist)
- [ ] `test_set` passed if you need an unbiased number to report
- [ ] for agentic engines: `claude` CLI on PATH + authed (`scripts/preflight.py`)
