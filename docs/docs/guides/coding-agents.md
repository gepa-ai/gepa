# Running GEPA inside Autonomous Coding Agents

GEPA is increasingly invoked by long-horizon coding agents — Claude Code, Codex `/goal`, Cursor agent mode, Devin-style task runners. The risk is that an agent picks a budget that's too small to actually optimize, then declares the goal reached because the one accepted candidate happened to beat the baseline on external validation.

There are two ways to stop a GEPA run, and both can fail this way if configured wrong:

- **`max_metric_calls`** — fixed budget. Easy to undersize.
- **`stop_callbacks`** (e.g. `NoImprovementStopper`, `ScoreThresholdStopper`, `TimeoutStopCondition`) — substantive conditions. Skip the budget entirely; let GEPA run until the optimization actually converges. Easy to set the stopper's threshold too aggressive.

For unattended agent runs, the cleanest default is to skip `max_metric_calls` and use `NoImprovementStopper(max_iterations_without_improvement=10)` (or larger). That gives the optimizer license to keep proposing as long as it's making progress, and avoids the agent having to guess a magic number.

This guide tells you what to put in your agent's prompt so it doesn't reward-hack the optimization itself.

## The failure mode

A real example, reported on [X by @onusoz](https://x.com/onusoz/status/2065673431027503566) ([issue #375](https://github.com/gepa-ai/gepa/issues/375)):

> I had set a `/goal` before I slept to implement a plan. It ended the loop after doing just 1 iteration. It feels like the model is following the path of least resistance and slacking off.

The agent's own post-mortem from the same session:

> GEPA is controlled by `max_metric_calls`, not an explicit iterations flag. The earlier 12B six-row run used `max_metric_calls=8`. With `row_limit=6`, the baseline full validation already costs about 6 metric calls. One accepted candidate then costs minibatch eval plus another full validation. So the budget was basically enough for only one proposal, even though the plan implied iterative GEPA optimization.
>
> The failure was operational: I treated the first successful 12B GEPA run as meaningful progress because it produced a better candidate and passed external validation, but I did not verify that it had enough optimizer trajectory to support "scores across iterations."

This is the modal failure of unattended GEPA runs under coding agents.

## What to put in your agent's prompt

When asking a coding agent to run GEPA, **always** include these four things:

### 1. Pick a stopping strategy explicitly

Either skip `max_metric_calls` entirely and use a substantive stopper, **or** set a budget large enough to not be the limiting factor — don't pick a small budget "to save time".

> Prefer `stop_callbacks=[NoImprovementStopper(max_iterations_without_improvement=10)]` and leave `max_metric_calls` unset. If you must set `max_metric_calls`, make it **more than `15 * len(valset)`** (rule of thumb), or more precisely `len(valset) + 15 * (reflection_minibatch_size + len(valset))`. Either way, the optimizer needs room for at least ~15 proposal attempts to do real evolutionary search.

### 2. A success criterion in terms of accepted candidates

External validation score lift is *not* a sufficient success criterion — the agent in the incident above passed external validation with a single accepted candidate. Require depth:

> After `gepa.optimize` returns, verify `result.num_candidates - 1 >= 5` (at least 5 proposals accepted). Note that this is a much weaker check than the ~15-attempt budget floor — many proposals get rejected, so 5 accepted is fine when the budget was sized for 15+ attempts. If fewer than 5 are accepted, the budget was too small or the evaluator signal too weak — do not claim the experiment is done.

### 3. Explicit instructions to read GEPABudgetWarning

> If `gepa.optimize` emits a `GEPABudgetWarning` (before or after the run), treat that as a hard failure of the experiment, not a soft warning. Re-run with a higher `max_metric_calls` before reporting completion.

### 4. Inspect `gepa-result.json` / `run_log.json` before presenting numbers

> Before drawing or reporting any score trajectory, open `<run_dir>/gepa-result.json` and confirm that `num_candidates` ≥ 6 and that the val-score sequence has at least 5 distinct rising points. A 2-point trajectory is *not* an optimization curve; it is one proposal step.

## Copy-paste prompt block

Drop this into your agent's system prompt or `/goal` brief:

```
When running gepa.optimize:

1. Pick a stopping strategy. Prefer skipping max_metric_calls and passing
   stop_callbacks=[NoImprovementStopper(max_iterations_without_improvement=10)]
   so GEPA runs until the optimization actually converges. If you do set
   max_metric_calls, make it > 15 * len(valset) (rule of thumb) — anything
   smaller caps the optimizer prematurely. Do not pick a small budget "to
   save time".

2. After the run, verify result.num_candidates - 1 >= 5. If fewer accepted
   proposals than that, the experiment is incomplete regardless of how the
   final validation score looks. Re-run with a larger budget (or a looser
   stopper) before reporting.

3. Treat any GEPABudgetWarning (Python warning category gepa.GEPABudgetWarning)
   as a failure of the experiment, not a soft warning. Do not consider a GEPA
   run as valid if either the pre-flight or post-run warning is active.

4. Before presenting a trajectory plot or score-over-iterations chart, open
   the run's gepa-result.json and verify num_candidates >= 6 and that the
   val_aggregate_scores list has at least 5 distinct rising values.
```

## What `gepa.optimize` itself enforces

GEPA's API tries to make these failures hard to ignore:

- **Pre-flight warning** — if `max_metric_calls` is set and below the ~15-attempt floor, a `GEPABudgetWarning` fires before the run starts. Skipping `max_metric_calls` and using `stop_callbacks` instead silences the pre-flight check (you're not over-promising depth via a small budget; the stopper decides).
- **Post-run summary** — every `gepa.optimize` call logs a 1-line summary at the end (`GEPA finished: N proposal(s) accepted over M metric call(s)...`) and warns if fewer than 3 proposals were accepted, regardless of which stopper fired first. If you used `NoImprovementStopper` and this fires, your `max_iterations_without_improvement` was probably too low.
- **Result inspection** — `result.num_candidates` (counts baseline) and `result.total_metric_calls` are first-class properties on `GEPAResult`.

These guardrails are deliberately *warnings*, not hard errors, because some users have legitimate reasons to run small budgets (smoke tests, CI). But agents driving long-horizon goals should treat them as failures.

## Related

- [Choosing `max_metric_calls`](budget.md) — the full budget formula and worked examples
- [Issue #375](https://github.com/gepa-ai/gepa/issues/375) — the report that motivated this guide
