# gepa.async_engine

A stage-decoupled asynchronous execution path for GEPA's reflective optimization loop.
It is a separate code path: it imports GEPA's building blocks and does not modify
`gepa.core.engine`, the proposers, or any `optimize_anything` backend.

## Why

The synchronous loop runs one iteration at a time, so an iteration costs the sum of its
stage latencies: parent rollout, reflection, child screening, validation. Batched parallel
proposals widen each step but keep the barriers between stages, and a wide step still waits
for its slowest reflection call.

This engine removes the barriers. Each heavy stage has its own buffer and workers, so while
one work item is being reflected on, another is being screened and a third validated.
Steady-state throughput is set by the slowest stage rather than by the sum, and each stage's
worker count is an independent dial.

```
sample -> [rollout] -> select components -> [propose] -> [screen] -> gate -> [validate] -> commit
 head      workers           head            workers      workers    head     workers      head
```

One head thread owns the optimizer state and is the only writer of the candidate pool.
Worker threads run the bracketed stages and never see the state.

## Usage

```python
from gepa.async_engine import AsyncEngineConfig, StageConfig, optimize

result = optimize(
    seed_candidate={"system_prompt": "..."},
    trainset=train, valset=val,
    adapter=my_adapter,                 # evaluate() must be thread-safe
    reflection_lm="openai/gpt-5-mini",
    max_metric_calls=3000,
    async_config=AsyncEngineConfig(
        rollout=StageConfig(workers=2),
        propose=StageConfig(workers=6),
        screen=StageConfig(workers=2),
        validate=StageConfig(workers=1),
        staleness_policy="guarded",
        max_staleness=4,
    ),
)
print(result.best_candidate)
print(result.metadata["async"])         # per-stage timing, overlap, staleness histograms
```

`optimize` takes the same core arguments as `gepa.optimize` and returns the same `GEPAResult`.

## Staleness

Overlap means a child can be built from a parent that was chosen several commits ago. Every
work item records the pool version (number of commits) it was sampled at; its *gap* is the
current version minus that.

| Policy | Behavior |
|---|---|
| `full` | Never checks the gap. Highest throughput. |
| `guarded` | Drops an item whose gap exceeds `max_staleness`, before spending reflection, screening, or validation on it. |
| `reflective` | Sends a stale proposal to the reflection LM with the version it started from, the current best, and what was accepted since. The LM rewrites it on top of the current best or discards it. The rewritten child is re-gated against its new parent on the same minibatch. |

Acceptance compares a child with its own parent on the same minibatch, so a stale child that
passes the gate is still a valid candidate. The policy controls how much outdated work the pool
absorbs, not correctness.

## Other controls

- **Per-stage workers and batching.** `StageConfig(workers, batch_size, max_queue)`. A batch
  goes to `adapter.batch_evaluate` in one call. A full buffer applies backpressure upstream.
- **Shared resources.** Rollout, screen and validate draw on the `evaluator` resource; propose
  and patch on the `proposer`. `resource_limits={"evaluator": 4}` caps tasks in flight per
  resource across stages.
- **Adaptive workers.** `adaptive_workers=True` compares stage production rates every
  `adjust_interval_seconds`. A slow stage with work waiting gains a worker; a fast stage filling
  a full downstream buffer loses one. At most one change per stage per adjustment.
- **Budget.** `reserve_budget=True` reserves metric calls at dispatch, so in-flight work cannot
  overshoot `max_metric_calls`.
- **Early-exit validation.** `validate_prefix_fraction` scores a candidate on a prefix of the
  validation ids first and validates the rest only if it keeps up with the current best there.
  Ids that recent candidates all pass are moved behind the prefix. Off by default, because
  rejected candidates never enter the pool.
- **Draining.** `drain="finish"` completes in-flight items after the stop condition fires;
  `"cancel"` drops everything that has not reached validation.

## Guarantees and differences

- `AsyncEngineConfig.sequential()` keeps one item in the pipeline and reproduces the
  synchronous engine's candidates, lineage, scores and metric-call counts on a seeded run
  (`tests/test_async_engine.py::test_sequential_config_reproduces_synchronous_run`).
- Beyond that the trajectory differs by design: proposals are accepted one by one as they
  finish, in completion order.
- Not supported here: the merge proposer, batch selection strategies (best-of-batch, top-k),
  `write_agent_state`, and resuming with a new seed candidate.
- A stateful `ReflectionLM` is chained in completion order when `propose.workers > 1`.
- Callbacks fire on the head thread. `iteration` in events is the work order's number, and
  events from different orders interleave.

## Outputs

With `run_dir` set: `gepa_state.bin` (resumable), `async_events.jsonl` (every sample, dispatch,
completion, routing decision and commit, with timestamps and queue depths) and
`async_stats.json` (the same summary as `result.metadata["async"]`).

## Try it

`uv run python examples/async_engine/simulated_latency.py` compares the synchronous engine,
batched parallel proposals, and this engine at several reflection worker counts on a synthetic
task with long-tailed reflection latency. No API keys needed.
