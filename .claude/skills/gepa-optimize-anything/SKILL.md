---
name: gepa-optimize-anything
description: >-
  Automatically improve any text artifact that can be scored — prompts, programs/code, configs,
  specs, regex/SQL/schemas, agent scaffolds, or encoded search solutions — by reflective evolutionary
  search, where an LLM repeatedly proposes better versions from execution feedback and an evaluator
  you provide assigns the score (an objective metric or an LLM-as-judge for subjective tasks).
  optimize_anything gives one interface across several optimizers (GEPA, AutoResearch, MetaHarness,
  best-of-N): write the evaluator once and switch optimizer with a single argument. Use whenever a
  task needs auto-optimizing, tuning, or searching over text with any quality signal (accuracy, pass
  rate, latency, cost, judge rating); when building an evaluator + proposer loop; when running GEPA;
  or when comparing optimizers. The candidate can be any string an evaluator can grade, single or
  multi-component, scored on one problem or generalized across many.
---

# GEPA / `optimize_anything`

GEPA is a **general optimizer for text artifacts**. You provide (1) a seed artifact, (2) an
**evaluator** that scores any artifact and returns feedback, and (3) a **proposer LLM**; GEPA then
repeatedly has the LLM read the feedback and propose improved artifacts, keeping a Pareto frontier
across your examples. It is a **black-box optimizer**: GEPA never needs to understand your artifact or
your metric — it only sees the scalar score and the feedback text your evaluator emits. The leverage
is in your score and your feedback.

**`optimize_anything` is a common interface over several optimizers.** You write the task and
evaluator once, then choose the search algorithm with one `engine` argument — and the same code runs
under any of them:
- **`gepa`** — reflective evolutionary search (an LLM reflects on feedback and mutates candidates;
  keeps a Pareto frontier). The default; strongest when feedback is rich.
- **`best_of_n`** — sample N candidates from a model and keep the best. Simple, strong baseline.
- **`autoresearch`** — an agentic optimizer (a Claude Code subprocess) that iterates like a researcher.
- **`meta_harness`** — an agentic proposer that reads the frontier/history and writes new candidates.

This makes it easy to start with one optimizer and benchmark others on the identical task/evaluator.

## What can be a candidate
A candidate is **any string your evaluator can score**. "Text in → a number out (higher is better),
plus optional feedback" is the entire contract, which covers a wide range of artifacts:
- **prompts** — system/user prompts, instruction templates, rubrics, few-shot exemplars
- **programs / code** — functions, whole files, CUDA kernels, scored by compiling + running + benchmarking
- **configs / specs / schemas / regex / SQL** — any text whose effect you can measure
- **agent scaffolds** — tool instructions, planner/critic prompts, orchestration text
- **pure search artifacts** — a mathematical construction, a packing layout, a plan, encoded as text

The score can be an **objective metric** (accuracy, pass rate, runtime, cost) **or an LLM-as-judge**
rating for subjective tasks (writing quality, helpfulness, style). You can also optimize **multiple
named components at once** (e.g. `{"system": ..., "rubric": ..., "examples": ...}`) so improvements
coordinate across the pieces.

## Three optimization modes (choose by how you pass data)
The mode is determined by whether you provide `dataset` and `valset`:
1. **Single-task** (`dataset=None, valset=None`) — solve one hard problem; the candidate *is* the
   solution; the evaluator is called with no example. *E.g. one CUDA kernel, a circle-packing layout.*
2. **Multi-task** (`dataset=<list>, valset=None`) — solve a batch of related problems with one shared
   candidate, transferring insight across them; evaluator called per example. *E.g. a single prompt
   that works across many tasks.*
3. **Generalization** (`dataset=<list>, valset=<list>`) — build a candidate that transfers to
   **unseen** problems; optimize on `dataset`, select on `valset`. *E.g. a prompt tuned to generalize.*

`test_set` is **separate from the modes and reporting-only**: the seed and the final candidate are
scored on it after optimization for an unbiased number — it never influences search or selection.
See `references/api.md` for details and when to use each mode.

## Install
```bash
pip install "gepa[full]"   # [full] pulls cloudpickle — needed to pickle closure evaluators for
                           # cache_evaluation/parallel workers; plain `pip install gepa` warns and
                           # can fail to cache when your evaluator closes over data.
# Proposer LLM: by default an id resolved via LiteLLM, so set that provider's key (e.g.
# ANTHROPIC_API_KEY, or AWS creds / AWS_BEARER_TOKEN_BEDROCK for Bedrock). You can also pass any
# callable implementing GEPA's LM protocol — a self-hosted / custom inference engine — instead.
# Agentic engines (autoresearch, meta_harness) additionally need the `claude` CLI on PATH.
```

## Mental model (4 pieces)
1. **Candidate** — the text you optimize (`str`, or a dict of named components). The seed is the start.
2. **`evaluate(candidate, example) -> (score, info)`** — `score` is a float (higher is better);
   `info` is a free-form dict of **feedback the proposer reads** to make better candidates.
3. **Engine** — the optimizer: `"gepa"` (default), `"best_of_n"`, `"autoresearch"`, `"meta_harness"`.
4. **Budget** — `max_evals` (eval-call cap) and/or `max_token_cost` (proposer-LLM USD cap). At least
   one must be set or the run is unbounded (only a warning). **Size it for many proposal rounds, not
   one** (see below) — this is the most common way agents misuse GEPA.

## Sizing the valset and the budget (read this — the #1 mistake)
`max_evals` is the *only* control over how long GEPA optimizes. Set it too low and the run stops after
a **single proposal**, then reports a "best candidate" that looks fine but is barely optimized.

- **valset** — GEPA selects the best candidate by scoring on the valset, so make it a
  **representative subset of your data**. There is no fixed size: pick what represents the task (a
  handful for small/expensive tasks, more when data is cheap and plentiful — anywhere from a few to
  hundreds). Bigger valset = less noisy selection but more eval calls per candidate.
- **budget** — every proposed candidate is scored on the **whole set GEPA selects from**, so size
  `max_evals` off that set, not off a single proposal. Which set that is depends on the mode:
  ```
  generalization (dataset + valset):  max_evals ≳ 15–20 × len(valset)    # scored & selected on valset
  multi-task     (dataset only):      max_evals ≳ 15–20 × len(dataset)   # scored & selected on dataset
  single-task    (no dataset/valset): max_evals ≳ 15–20                  # 1 eval per candidate, so this
                                                                         #   IS the number of proposals
  ```
  The constant is the same everywhere — **let GEPA propose AND evaluate ~15–20 candidates** (more if
  you can afford it). Anything much less and GEPA tries only a couple of candidates — i.e. it's barely
  optimizing.

After the run, check how many proposals actually happened (`result.metadata` / the `run_dir`
iterations). **If it stopped after one proposal, the budget was too low** — raise it and rerun.

### Make sure the run terminates (caching decouples `max_evals` from wall-clock)
GEPA caches evaluations by default, so **`max_evals` counts only cache *misses***. Once the search
converges, the proposer keeps generating candidates that hit the cache — `max_evals` stops being
consumed and the run **spins until your process times out**, still spending (uncached) proposer-LLM
calls the whole time. Protect every run with explicit stop conditions:
- **`stop_at_score`** — set it whenever your metric has a known ceiling (e.g. `1.0` for a pass rate /
  accuracy). GEPA stops the moment a candidate reaches it, instead of proposing forever at the optimum.
- **`max_token_cost`** — a hard USD cap on proposer-LLM spend; stops the run even when evals are cached.
- **a wall-clock `timeout`** on the process you launch (e.g. `timeout 1200 python run.py`) as a backstop.

Rule: for a bounded metric, *always* pass `stop_at_score`; otherwise set `max_token_cost` and a
wall-clock timeout. Don't rely on `max_evals` alone to end the run.

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
        engine="gepa",               # swap to "best_of_n" / "autoresearch" / "meta_harness" — same code
        name="my_run",
        max_evals=300,               # ≳ 15-20 × len(valset): enough for ~15-20 proposals (see below)
        stop_at_score=1.0,           # stop at the optimum (set when your metric has a known ceiling)
        max_concurrency=16,
        run_dir="runs/my_run",       # engine artifacts: iterations/, pareto/
        output_dir="outputs/my_run", # per-eval JSON, progress_log.jsonl, summary.json
        engine_config={
            "reflection": {
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
2. **Define the score deliberately.** GEPA optimizes exactly what you measure — gate the score on what
   you actually care about (see `references/gotchas.md`, reward hacking).
3. **Write a feedback-rich `evaluate`.** The `info` dict is the proposer's signal — return errors,
   diffs, partial credit, not just a number. See `references/writing_evaluators.md`.
4. **Pick a proposer LLM** — a LiteLLM id (set the provider key) or a custom LM-protocol callable.
   Validate it with a 1-call test before a long run.
5. **Set a budget** (`max_evals` and/or `max_token_cost`).
6. **Run `python scripts/preflight.py`** to fail fast on missing creds / CLI before a long run.
7. **Launch**, watch the first 1-2 evals (the eval→model→score chain), then let it run.
8. **Read `result.best_candidate` and `run_dir/`** (and `result.metadata["test_score"]` if you
   passed a `test_set`).

## Critical gotchas (read before a real run)
These silently degrade *results* — skim before launching:
- **Reward hacking.** GEPA optimizes exactly what you score; a weak proxy gets gamed (e.g. a
  "correct"-only score → a do-nothing wrapper). Gate the score on the real goal. → `references/gotchas.md`.
- **Selection bias.** In generalization mode the best candidate is the max over many scored on
  `valset` — an optimistic estimate. Use enough `valset` examples (and N>1 for stochastic models), and
  report on a `test_set` for an unbiased number. → `references/gotchas.md`.
- **Stochastic models default to N=1 per eval** → noisy selection. Average N samples *inside*
  `evaluate`. → `references/writing_evaluators.md`.
- **Saturated signal → GEPA returns the seed unchanged.** If the seed already aces the training
  examples, every proposal looks "not better" and is rejected (many proposals, ~0 accepted). GEPA needs
  examples the seed gets *wrong* to learn from — ensure `dataset` has real failures. → `references/gotchas.md`.
- **Agentic engines (`autoresearch`, `meta_harness`) shell out to the `claude` CLI** and fail late if
  it's missing → run `scripts/preflight.py`; details in `references/api.md`.

## Reference files (load as needed)
- `references/api.md` — `OptimizeAnythingConfig` + `engine_config` schema, the engines & their
  prerequisites, the three modes, the LM protocol, budget/cost semantics, `Result` shape.
- `references/writing_evaluators.md` — the `(score, info)` contract, LLM-as-judge scoring,
  multi-objective via `info["scores"]`, N>1 averaging, feedback design.
- `references/tracking.md` — enabling wandb / mlflow experiment tracking and what gets logged.
- `references/gotchas.md` — reward hacking, selection bias, the three modes, engine prerequisites.
- `templates/optimize_prompt.py` — a runnable, copy-paste starting point.
- `scripts/preflight.py` — validate creds / proposer LM / `claude` CLI before launching.
