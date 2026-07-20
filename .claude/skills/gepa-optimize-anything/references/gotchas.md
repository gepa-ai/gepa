# Gotchas and pitfalls

Read before any real run.

## 1. Reward hacking — the optimizer exploits a weak proxy ruthlessly
Every backend maximizes exactly the score you compute; if it is gameable, it *will* be gamed. Real
example (KernelBench, gepa backend): a correctness-only score drove the optimizer to a prompt that
tells the model to **wrap the reference implementation** — always "correct", zero real work. Result: pass@1 ~0.87 but the actual goal (fast kernels) collapsed to ~base.
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

## 4. The default budget is probably wrong for your run
`max_evals` defaults to **100** — bounded, but rarely sized for your selection set (see SKILL.md's
sizing rule: ≳ 15–20 × the selection set). Set it explicitly. Only if you set **both**
`max_evals=None` and `max_token_cost=None` is the run unbounded, and even that is just a
`warnings.warn`, not an error.

## 5. `engine_config` is validated strictly — unknown keys raise `TypeError`
Every backend parses `engine_config` into a typed dataclass (`GEPAConfig` for gepa,
`BestOfNConfig` / `AutoResearchConfig` / `MetaHarnessConfig` for the others), so a typo'd or
wrong-backend key fails **immediately at construction** with a `TypeError` — it is *not* silently
dropped. The practical consequence: swapping `engine=` requires swapping the whole `engine_config`
block, and old-API keys (`claude_code_agent`, top-level `reflection_lm_kwargs`, `objective`,
`background` inside `engine_config`) now crash. See `api.md` for each backend's valid keys.

## 6. Agentic backends fail late
`autoresearch` / `meta_harness` `subprocess.Popen(["claude", ...])`; a missing/unauthenticated CLI
(or missing `jq`, which autoresearch's generated `eval.sh` uses) surfaces only after a subprocess
error mid-run. Run `scripts/preflight.py` first.

## 7. Give runs a real stop condition (`stop_at_score` / `max_token_cost`)
`max_evals` caps eval calls, but two situations still burn money or time past the point of useful
work:
- **The metric has a ceiling** and a candidate reaches it — without `stop_at_score` the run keeps
  proposing at the optimum until the budget is gone. For a bounded metric (pass rate, accuracy),
  *always* pass `stop_at_score`.
- **You opted into evaluation caching** (`engine_config={"engine": {"cache_evaluation": True}}` —
  it is **off by default**). Then `max_evals` counts only cache *misses*: a converged search keeps
  emitting cache-hitting candidates without consuming eval budget and can spin until your process
  times out, still spending proposer-LLM tokens. With caching on, `stop_at_score` and/or
  `max_token_cost` (plus a wall-clock `timeout` on the launched process) are mandatory.
Agentic backends also spend LLM tokens between evals — cap them with `max_token_cost`
(enforced as `--max-budget-usd`).

## 8. Pick the right mode
The mode is implicit in which sets you pass (`api.md`): no `dataset`/`valset` → single-task;
`dataset` only → multi-task; `dataset`+`valset` → generalization. A common mistake is wanting a
candidate that generalizes but passing only `dataset` (multi-task) — then there's no held-out
selection and the candidate can overfit the training examples. Pass a `valset` for generalization.
(Held-out `valset` selection is implemented by the gepa backend; the other backends fold `valset`
into the scoring pool — see `api.md`.)

## 9. Saturated signal — the gepa backend returns the seed unchanged (no failure to learn from)
This one is specific to the gepa backend, whose optimizer improves by reflecting on examples the
current candidate gets **wrong**. If the seed already scores near-perfect on the train minibatch the
proposer reflects over, every mutation looks "not better" and is rejected by the acceptance gate —
so the run spends its full budget, accepts **zero** proposals, and hands back the seed. Symptom:
many proposals made, ~0 accepted, `best_candidate` == seed. This is "baseline saturation," not a
bug: there's no gradient for reflection to climb.
Mitigations: make sure `dataset` contains examples the seed actually **fails** (harder/larger train
set, or a task-LM weak enough to generate failures); if the seed is already at ceiling, reflective
mutation has little to do — the `best_of_n` baseline (independent full rewrites, no acceptance
gate) or just keeping the seed may be the better call.
Always check **proposals accepted**, not just proposals made.

## 10. Uncaught evaluator exceptions abort the run
On the gepa backend, `EngineConfig.raise_on_exception` defaults to `True`: an exception escaping
your evaluator stops the whole optimization. Either catch failures yourself and return them as low scores with rich
`info["error_*"]` feedback (preferred — the proposer learns from them), or set
`engine_config={"engine": {"raise_on_exception": False}}` to have them auto-converted to score
`0.0` with `info["error"]`.

## Quick pre-flight checklist
- [ ] mode matches intent (single-task / multi-task / generalization) via `dataset`/`valset`
- [ ] `dataset` contains examples the seed gets WRONG — reflection has no failure signal to learn
      from otherwise, and the gepa backend hands back the seed unchanged (#9)
- [ ] `score` matches the real goal (gated, not gameable)
- [ ] `info` carries actionable feedback (errors/diffs/partial credit)
- [ ] N>1 if the model is stochastic
- [ ] proposer LM valid + 1-call tested; creds present (default `openai/gpt-5.1` needs
      `OPENAI_API_KEY` — or set your own via `reflection.reflection_lm`)
- [ ] `max_evals` set explicitly and sized for many proposals (≳ 15–20 × the size of the selection
      set: `len(valset)` in generalization, `len(dataset)` in multi-task, or just ≳ 15–20 in
      single-task) — see SKILL.md
- [ ] `stop_at_score` set when the metric has a ceiling; `max_token_cost` for agentic backends
- [ ] `engine_config` keys match the chosen backend (typos raise `TypeError`, #5)
- [ ] `run_dir` + `output_dir` set (so artifacts persist)
- [ ] `test_set` passed if you need an unbiased number to report
- [ ] for agentic backends: `claude` CLI on PATH + authed, `jq` installed (`scripts/preflight.py`)
