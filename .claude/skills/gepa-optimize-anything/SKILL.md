---
name: gepa-optimize-anything
description: >-
  Automatically improve any text artifact that can be scored — prompts, programs/code, configs,
  specs, regex/SQL/schemas, agent scaffolds, or encoded search solutions — with optimize_anything,
  an LLM-driven optimization API: a search backend repeatedly proposes better versions from
  execution feedback and an evaluator you provide assigns the score (an objective metric or an
  LLM-as-judge for subjective tasks). optimize_anything is one interface across several optimizer
  backends — GEPA (reflective evolutionary search, the default), AutoResearch, MetaHarness — plus a
  best-of-N baseline: write the evaluator once and switch backend with a single argument.
  Use whenever a task needs auto-optimizing, tuning, or searching over text with any quality signal
  (accuracy, pass rate, latency, cost, judge rating); when building an evaluator + proposer loop;
  when running GEPA; or when comparing optimizers. The candidate can be any string an evaluator can
  grade, scored on one problem or generalized across many.
---

# `optimize_anything`

**Naming, precisely.** `optimize_anything` is the tool: a general API for optimizing text
artifacts. **GEPA** is one specific optimizer behind it — reflective evolutionary search, the
default backend (`engine="gepa"`) — and, for legacy reasons, also the name of the Python package
that ships all of this. In this skill, "the gepa backend" always means the optimizer; statements
about "the optimizer" or "the backend" apply to whichever engine you chose.

`optimize_anything` does **black-box optimization**: you provide (1) a seed artifact, (2) an
**evaluator** that scores any artifact and returns feedback, and (3) a backend, which repeatedly
proposes improved artifacts and scores them through your evaluator. "Black-box" refers to the
**evaluator**, not the artifact: the backend never sees how the score is computed — no gradients,
no metric internals — only the scalar score and the feedback text you emit. The candidate itself
*is* visible: the proposer reads and rewrites it, applying the LLM's understanding of your artifact.
The framework just imposes no structure on it — any string an evaluator can score works. The
leverage is in your score and your feedback.

**You write the task and evaluator once, then choose the search algorithm with one `engine`
argument** — and the same code runs under any of them:
- **`gepa`** — the GEPA optimizer: reflective evolutionary search, in-process (an LLM reflects on
  feedback and mutates candidates; keeps a Pareto frontier). The default; strongest when feedback
  is rich.
- **`autoresearch`** — an agentic optimizer: one Claude Code subprocess iterates like a researcher in
  a work dir, scoring candidates through an HTTP eval server.
- **`meta_harness`** — an agentic proposer (Claude subprocess) that reads the frontier/history each
  iteration and writes new candidates for the engine to benchmark.

(There is also a `best_of_n` engine — sample N independent candidates, keep the best. It is
deliberately naive: use it as a **baseline** to compare an optimizer against, not as the optimizer.)

This makes it easy to start with one backend and benchmark others on the identical task/evaluator.
There are also **composition/pipeline helpers** that combine backends over the same task:
`optimize_sequential` (a pipeline — each stage's best seeds the next), `optimize_parallel`,
`optimize_best_of`, `optimize_vote` (re-score each branch's best for a fair pick), and an adaptive
scheduler that rotates backends on score plateaus — see `references/api.md`.

## What can be a candidate
A candidate is **any string your evaluator can score**. "Text in → a number out (higher is better),
plus optional feedback" is the entire contract, which covers a wide range of artifacts:
- **prompts** — system/user prompts, instruction templates, rubrics, few-shot exemplars
- **programs / code** — functions, whole files, CUDA kernels, scored by compiling + running + benchmarking
- **configs / specs / schemas / regex / SQL** — any text whose effect you can measure
- **agent scaffolds** — tool instructions, planner/critic prompts, orchestration text
- **pure search artifacts** — a mathematical construction, a packing layout, a plan, encoded as text

The score can be an **objective metric** (accuracy, pass rate, runtime, cost) **or an LLM-as-judge**
rating for subjective tasks (writing quality, helpfulness, style). At this API the candidate is a
single string (`seed_candidate: str | None`; `None` = seedless — the engine bootstraps from
`objective`/`background`). Multi-component dict candidates exist only in the lower-level
`gepa.gepa_launcher.optimize_anything` API, not here.

## Three optimization modes (choose by how you pass data)
The mode is determined by whether you provide `dataset` and `valset`:
1. **Single-task** (`dataset=None, valset=None`) — solve one hard problem; the candidate *is* the
   solution; the evaluator is called with no example. *E.g. one CUDA kernel, a circle-packing layout.*
2. **Multi-task** (`dataset=<list>, valset=None`) — solve a batch of related problems with one shared
   candidate, transferring insight across them; evaluator called per example. *E.g. a single prompt
   that works across many tasks.*
3. **Generalization** (`dataset=<list>, valset=<list>`) — build a candidate that transfers to
   **unseen** problems; optimize on `dataset`, select on `valset`. *E.g. a prompt tuned to generalize.*
   *Note:* GEPA is the algorithm designed around this mode, and `valset`-based held-out selection is
   currently implemented only by the gepa backend — the other backends fold `valset` into the
   training pool (they can still generalize; there's just no separate selection split).

`test_set` is **separate from the modes and reporting-only**: the seed and the final candidate are
scored on it after optimization for an unbiased number — it never enters the search, selection, or
budget (for the agentic backends it is sealed at the eval server's HTTP layer, so the agent cannot
even see it). See `references/api.md` for details and when to use each mode.

## Install
```bash
pip install "gepa[full]"   # [full] pulls cloudpickle — needed to pickle closure evaluators for
                           # parallel workers / opt-in evaluation caching; plain `pip install gepa`
                           # can fail there when your evaluator closes over data.
# Proposer LLM: the gepa backend's reflection LM defaults to "openai/gpt-5.1" (a LiteLLM id) — set
# that provider's key (OPENAI_API_KEY), or pass your own id (e.g. "anthropic/claude-sonnet-4-6" with
# ANTHROPIC_API_KEY, or a Bedrock ARN with AWS creds). You can also pass any callable implementing
# GEPA's LM protocol — a self-hosted / custom inference engine — instead of a model-id string.
# Agentic backends (autoresearch, meta_harness) additionally need the `claude` CLI on PATH (plus
# `jq` for the generated eval.sh).
```

## Mental model (4 pieces)
1. **Candidate** — the text you optimize (a `str`). The seed is the start; `None` = seedless.
2. **`evaluate(candidate, example) -> (score, info)`** — `score` is a float (higher is better);
   `info` is a free-form dict of **feedback the backend's proposer reads** to make better candidates
   (the gepa backend's reflection LM, or the agentic backends' agent; the `best_of_n` baseline
   ignores feedback). When evals batch better than they stream (e.g. a provider batch API), pass
   `batch_evaluator=` instead — all pending `(candidate, example)` pairs in one call; see
   `references/api.md`.
3. **Engine (backend)** — `"gepa"` (default), `"autoresearch"`, `"meta_harness"`, the `"best_of_n"`
   baseline, or a constructed `Engine` instance.
4. **Budget** — `max_evals` (server-side eval-call cap, **default 100**) and/or `max_token_cost`
   (USD cap on the backend's own proposer-LLM spend). Setting both to `None` makes the run unbounded
   (only a warning). **Size `max_evals` for many proposal rounds, not one** (see below) — this is the
   most common way agents misuse this API.

## Sizing the valset and the budget (read this — the #1 mistake)
`max_evals` is the *main* control over how long the optimizer runs. Leave it at the default (100)
with a large valset, or set it too low, and the run stops after a **single proposal**, then reports
a "best candidate" that looks fine but is barely optimized.

- **valset** — on the gepa backend the best candidate is selected by its scores on the valset, so
  make it a **representative subset of your data**. There is no fixed size: pick what represents the task (a
  handful for small/expensive tasks, more when data is cheap and plentiful — anywhere from a few to
  hundreds). Bigger valset = less noisy selection but more eval calls per candidate.
- **budget** — every proposed candidate is scored on the **whole selection set**, so size
  `max_evals` off that set, not off a single proposal. Which set that is depends on the mode:
  ```
  generalization (dataset + valset):  max_evals ≳ 15–20 × len(valset)    # scored & selected on valset
  multi-task     (dataset only):      max_evals ≳ 15–20 × len(dataset)   # scored & selected on dataset
  single-task    (no dataset/valset): max_evals ≳ 15–20                  # 1 eval per candidate, so this
                                                                         #   IS the number of proposals
  ```
  The constant is the same everywhere — **let the backend propose AND evaluate ~15–20 candidates**
  (more if you can afford it). Anything much less and the run tries only a couple of candidates —
  i.e. it's barely optimizing. (This arithmetic is exact for the gepa backend — and the best_of_n
  baseline — which score every candidate on the full selection set; the agentic backends decide
  themselves how to spend eval calls, so treat it as a floor.)

After the run, check how many proposals actually happened (on the gepa backend,
`result.metadata["gepa_result"]` holds every candidate; with `engine.write_agent_state=True` the
`run_dir/iterations/` tree shows each one). **If it stopped after one proposal, the budget was too
low** — raise it and rerun.

### Give every run a real stop condition
`max_evals` bounds eval calls, but add explicit stops so runs end at the right moment:
- **`stop_at_score`** — set it whenever your metric has a known ceiling (e.g. `1.0` for a pass rate /
  accuracy). The backend stops the moment a candidate reaches it instead of burning the rest of the
  budget at the optimum.
- **`max_token_cost`** — a hard USD cap on the backend's own proposer/agent LLM spend. Especially
  important for the agentic backends (`autoresearch`, `meta_harness`), whose Claude subprocesses
  spend tokens between eval calls.
- **a wall-clock `timeout`** on the process you launch (e.g. `timeout 1200 python run.py`) as a
  backstop.
- If you **opt in** to evaluation caching (`engine_config={"engine": {"cache_evaluation": True}}` on
  the gepa backend — it is **off by default**), be aware `max_evals` then counts only cache *misses*:
  a converged search can keep proposing cache-hitting candidates without consuming eval budget, so
  `stop_at_score`/`max_token_cost` become mandatory, not optional.

## Minimal working example
The example optimizes a system prompt for concreteness, but the **shape is identical** for any
candidate — swap `SEED` for a code file / config / etc. and have `evaluate` compile/run/measure it.
```python
from gepa.optimize_anything import optimize_anything, OptimizeAnythingConfig

SEED = "You are an expert. Solve the task. Output only the final answer."

def evaluate(candidate: str, example) -> tuple[float, dict]:
    output = run_my_model(system_prompt=candidate, user_prompt=example["prompt"])  # your call
    score = grade(output, example)                                                 # float, higher=better
    return score, {                       # everything here is shown to the proposer LLM
        "score": score,
        "output": output,
        "error": example.get("error"),    # concrete, actionable feedback drives good proposals
    }

result = optimize_anything(
    seed_candidate=SEED,
    evaluator=evaluate,
    dataset=trainset,            # optimize on these (multi-task/generalization mode)
    valset=valset,               # select the best candidate on these (generalization mode)
    test_set=testset,            # OPTIONAL, reporting-only: seed + final candidate scored here at the end
    objective="Produce a prompt that maximizes task accuracy.",
    background="Domain rules, constraints, output format the model must follow.",
    config=OptimizeAnythingConfig(
        engine="gepa",               # swap to "autoresearch" / "meta_harness" — same code
                                     #   ("best_of_n" runs the same way, as a comparison baseline)
        name="my_run",
        max_evals=300,               # ≳ 15-20 × len(valset): enough for ~15-20 proposals (see above)
        stop_at_score=1.0,           # stop at the optimum (set when your metric has a known ceiling)
        max_concurrency=16,
        run_dir="runs/my_run",       # engine workspace (gepa run dir / agent work dir)
        output_dir="outputs/my_run", # eval server: per-eval JSON, progress_log.jsonl, summary.json
        engine_config={              # gepa backend: a GEPAConfig-shaped dict, validated strictly —
            "reflection": {          #   an unknown key raises TypeError immediately (fail fast)
                # a LiteLLM id (set the provider key) OR any callable implementing the LM protocol
                "reflection_lm": "anthropic/claude-sonnet-4-6",
                "reflection_minibatch_size": 5,
            },
            "engine": {"max_workers": 32, "seed": 0},  # seed = reproducibility
        },
    ),
)
print(result.best_candidate, result.best_score)
# held-out (only present if you passed test_set): "test_score" = average, "test_scores" = per-example
print("held-out:", result.metadata.get("test_score"),
      "seed held-out:", result.metadata.get("baseline_test_score"))
```

## Standard workflow
1. **Pick the mode** (single-task / multi-task / generalization) by which of `dataset`/`valset` you pass.
2. **Define the score deliberately.** The optimizer optimizes exactly what you measure — gate the
   score on what you actually care about (see `references/gotchas.md`, reward hacking).
3. **Write a feedback-rich `evaluate`.** The `info` dict is the proposer's signal — return errors,
   diffs, partial credit, not just a number (`oa.log()` and `capture_stdio` can route diagnostics in
   automatically). See `references/writing_evaluators.md`.
4. **Pick a proposer LLM** — a LiteLLM id (set the provider key) or a custom LM-protocol callable.
   Validate it with a 1-call test before a long run.
5. **Set a budget** (`max_evals` sized per above, and/or `max_token_cost`) plus `stop_at_score` when
   the metric has a ceiling.
6. **Run `python scripts/preflight.py`** to fail fast on missing creds / CLI before a long run.
7. **Launch**, watch the first 1-2 evals (the eval→model→score chain), then let it run.
8. **Read `result.best_candidate` and `run_dir/`** (and `result.metadata["test_score"]` if you
   passed a `test_set`).

## Critical gotchas (read before a real run)
These silently degrade *results* — skim before launching:
- **Reward hacking.** Every backend optimizes exactly what you score; a weak proxy gets gamed (e.g. a
  "correct"-only score → a do-nothing wrapper). Gate the score on the real goal. → `references/gotchas.md`.
- **Selection bias.** In generalization mode the best candidate is the max over many scored on
  `valset` — an optimistic estimate. Use enough `valset` examples (and N>1 for stochastic models), and
  report on a `test_set` for an unbiased number. → `references/gotchas.md`.
- **Stochastic models default to N=1 per eval** → noisy selection. Average N samples *inside*
  `evaluate`. → `references/writing_evaluators.md`.
- **Saturated signal → the gepa backend returns the seed unchanged.** If the seed already aces the
  training examples, every proposal looks "not better" and is rejected (many proposals, ~0 accepted).
  Reflection needs examples the seed gets *wrong* to learn from — ensure `dataset` has real
  failures. → `references/gotchas.md`.
- **`engine_config` is validated strictly per backend** — an unknown key (including a leftover key
  from a different backend after swapping `engine=`) raises `TypeError` at construction. Swapping
  `engine=` means swapping the `engine_config` block. → `references/api.md`.
- **Agentic backends (`autoresearch`, `meta_harness`) shell out to the `claude` CLI** and fail late if
  it's missing → run `scripts/preflight.py`; details in `references/api.md`.

## Reference files (load as needed)
- `references/api.md` — `OptimizeAnythingConfig`, the backends and their typed `engine_config`
  options, the three modes, the LM protocol, budget/cost semantics, `Result` shape, and the
  composition/pipeline helpers.
- `references/writing_evaluators.md` — the `(score, info)` contract, `oa.log()`/`capture_stdio`,
  LLM-as-judge scoring, multi-objective via `info["scores"]`, N>1 averaging, feedback design.
- `references/tracking.md` — enabling wandb / mlflow experiment tracking and what gets logged.
- `references/gotchas.md` — reward hacking, selection bias, the three modes, backend prerequisites.
- `scripts/preflight.py` — validate creds / proposer LM / `claude` CLI before launching.
