# Writing evaluators

The evaluator is where almost all of your effort and almost all of the quality come from. Whatever
backend you choose can only be as good as the score you compute and the feedback you return.

## The contract
```python
def evaluate(candidate: str, example) -> tuple[float, dict]:
    return score, info
```
- `score: float` — **higher is better**. This is what the optimizer maximizes and selects on.
- `info: dict` — **free-form feedback shown to the proposer**: the gepa backend's reflection LM
  (as "Actionable Side Information", ASI) or the agentic backends' agent (the `best_of_n` baseline
  ignores it). This is the single biggest lever on mutation quality.
- For single-task runs the signature is `evaluate(candidate)`; with `dataset`/`valset`/`test_set`
  it's `evaluate(candidate, example)`. (Returning a bare `float` instead of a tuple also works —
  the wrapper normalizes it — but then the proposer gets no feedback; always return the tuple.)

## Make `info` feedback-rich (not just a number)
The proposer (reflection LM or agent) writes the next candidate by reading `info`. Give it specifics:
```python
return score, {
    "score": score,
    "output": output,                 # what the candidate produced
    "expected": example.get("gold"),  # what was wanted (if available)
    "error_type": err_type,           # compile error / wrong answer / format violation / timeout
    "error_detail": traceback_or_diff,# the actual message — concrete > vague
    "passed_checks": [...],           # partial credit signals
    "failed_checks": [...],
}
```
Rule of thumb: if a smart human reading only `info` could tell you *how to fix the candidate*, the
proposer LLM can too. If `info` is just `{"score": 0.0}`, the search is blind.

### Two built-in channels for diagnostics
- **`oa.log()`** — `import gepa.gepa_launcher as oa; oa.log("landing distance:", d)` inside your
  evaluator. Same calling convention as `print()`; output is captured per-eval (thread-safe) and
  auto-included in the feedback under `info["log"]`. For child threads, propagate the context via
  `oa.get_log_context()` / `oa.set_log_context()`.
- **`capture_stdio`** — set `engine_config={"engine": {"capture_stdio": True}}` (gepa backend) and
  any `print()`/stdout/stderr during evaluation lands in the feedback under `"stdout"`/`"stderr"` —
  useful for wrapping existing scripts with no code changes. (Doesn't catch C-extension or
  subprocess output that bypasses Python's `sys.stdout`; route that through `oa.log()`.)

## LLM-as-judge scoring (subjective tasks)
The score doesn't have to come from an objective metric. For open-ended tasks (writing quality,
helpfulness, tone, adherence to a rubric) the evaluator can call an **LLM judge** and use its rating
as the score — and, importantly, return the judge's **written critique** as feedback:
```python
def evaluate(candidate, example):
    output = run_my_model(candidate, example)
    verdict = judge_lm(JUDGE_RUBRIC.format(task=example, answer=output))  # your judge call
    score = verdict["rating"] / 10.0                  # normalize to a float, higher = better
    return score, {
        "score": score,
        "output": output,
        "critique": verdict["critique"],              # the judge's reasoning → drives better proposals
        "rubric_breakdown": verdict.get("by_criterion"),
    }
```
Tips: pin the judge (fixed model + temperature, ideally a stronger model than the one being
optimized), give it a concrete rubric, and average a few judge calls if its ratings are noisy
(this is the N>1 idea below). The critique is often more valuable to the proposer than the number.

## Stochastic models: beat the silent N=1 default
The eval server calls your function **once** per (candidate, example) — there is no
`samples_per_eval` knob. For a temperature>0 model that makes every score a single-sample estimate,
and candidate **selection** then runs on noisy numbers (this is how you get winner's-curse inflation).
Fix it inside `evaluate` by averaging N samples:
```python
def evaluate(candidate, example, N=4):
    outs = [run_my_model(candidate, example) for _ in range(N)]
    scores = [grade(o, example) for o in outs]
    score = sum(scores) / len(scores)           # mean correctness ~ pass@1 estimate
    return score, {"score": score, "n": N,
                   "samples": [{"out": o, "s": s} for o, s in zip(outs, scores)]}
```
Trade-off: N× more eval calls (budget). Pick N to balance variance vs `max_evals`.

## Multi-objective optimization (e.g. correctness AND speed)
The gepa backend keeps an objective-level Pareto front (the other backends ignore
`info["scores"]`). To use it, **return per-objective metrics under `info["scores"]`** — the adapter
forwards them as `objective_scores`:
```python
return score, {
    "score": score,                        # the scalar the optimizer selects on
    "scores": {"correct": correct_rate,    # per-objective metrics for the Pareto frontier
               "speedup": speedup},        # all "higher is better"
    ...
}
```
`engine_config={"engine": {"frontier_type": "hybrid"}}` is the default (instance- and
objective-level fronts combined); `"objective"`, `"instance"`, and `"cartesian"` are the
alternatives. The scalar `score` still drives final selection; the per-objective scores shape the
frontier that candidates are drawn from.

## Choose the score to match the real goal (avoid reward hacking)
The optimizer maximizes exactly what you write. A correctness-only score is gameable — e.g. in code/kernel
tasks the optimizer learns to emit a trivial wrapper that is "correct" but does nothing useful.
Prefer a **gated** objective:
```python
def score_fn(result):
    if not (result["compiled"] and result["correct"]):
        return 0.0
    return f(result["speedup"])   # e.g. min(speedup / target, 1.0), monotonically increasing
```
Then the only way to raise the score is to be correct **and** better on what you care about. See
`gotchas.md` for the full reward-hacking story.

## Determinism & robustness
- Make `evaluate` side-effect-free and resumable; it may run concurrently (`max_concurrency`,
  `engine.max_workers`) and be retried.
- Set a seed in `engine_config={"engine": {"seed": 0}}` for reproducible search order.
- Log your own per-eval record (id, score, sub-metrics, candidate hash) — you'll want it for
  analysis; the eval server's `output_dir` per-eval JSON covers the basics, and `oa.log()` covers
  in-feedback diagnostics.
- Catch and *return* failures as low scores with `info["error_*"]`, rather than raising —
  `EngineConfig.raise_on_exception` defaults to `True`, so an uncaught exception aborts the run.
  (Setting it to `False` converts exceptions to score `0.0` with `info["error"]` instead.)
