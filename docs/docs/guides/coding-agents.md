# Running GEPA inside Autonomous Coding Agents

GEPA is increasingly invoked by long-horizon coding agents — Claude Code, Codex `/goal`, Cursor agent mode, Devin-style task runners. These agents are powerful but they lack a built-in model for how GEPA's only iteration knob, `max_metric_calls`, translates into actual optimization depth. The cheapest path that produces *any* accepted candidate looks indistinguishable from a real optimization run.

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

### 1. The budget formula, not a number

Tell the agent the formula, not a magic number — because the right budget depends on `len(valset)` and `reflection_minibatch_size`, which the agent might change.

> Set `max_metric_calls` to **at least `16 * len(valset)`** (rule of thumb), or more precisely `len(valset) + 15 * (reflection_minibatch_size + len(valset))`. This gives the optimizer room for ~15 proposal attempts, which is the recommended minimum. If you change `valset` size or minibatch size, recompute. Higher is better when wall-clock time allows.

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

1. Compute max_metric_calls >= 16 * len(valset) (rule of thumb), or more
   precisely len(valset) + 15 * (reflection_minibatch_size + len(valset)).
   Default reflection_minibatch_size is 3 if unspecified. This gives the
   optimizer room for the recommended ~15 proposal attempts. Use this
   formula as a floor — picking a smaller number to save time will silently
   produce a degenerate optimization curve. Higher is better when you have
   wall-clock budget.

2. After the run, verify result.num_candidates - 1 >= 5. If fewer accepted
   proposals than that, the experiment is incomplete regardless of how the
   final validation score looks. Re-run with a larger budget before reporting.

3. Treat any GEPABudgetWarning (Python warning category gepa.GEPABudgetWarning)
   as a failure of the experiment, not a soft warning. Do not present score
   improvements until the warning is gone.

4. Before presenting a trajectory plot or score-over-iterations chart, open
   the run's gepa-result.json and verify num_candidates >= 6 and that the
   val_aggregate_scores list has at least 5 distinct rising values.
```

## What `gepa.optimize` itself enforces

GEPA's API tries to make these failures hard to ignore:

- **Pre-flight warning** — if `max_metric_calls < len(valset) + 15 * (reflection_minibatch_size + len(valset))` (the floor for ~15 proposal attempts), a `GEPABudgetWarning` fires before the run starts.
- **Post-run summary** — every `gepa.optimize` call now logs a 1-line summary at the end (`GEPA finished: N proposal(s) accepted over M metric call(s)...`) and warns again if fewer than 3 proposals were accepted (the "almost nothing happened" threshold; the pre-flight is what enforces the full 15-attempt budget).
- **Result inspection** — `result.num_candidates` (counts baseline) and `result.total_metric_calls` are first-class properties on `GEPAResult`.

These guardrails are deliberately *warnings*, not hard errors, because some users have legitimate reasons to run small budgets (smoke tests, CI). But agents driving long-horizon goals should treat them as failures.

## Related

- [Choosing `max_metric_calls`](budget.md) — the full budget formula and worked examples
- [Issue #375](https://github.com/gepa-ai/gepa/issues/375) — the report that motivated this guide
