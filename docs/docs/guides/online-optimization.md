# Online optimization

`optimize_online` updates a candidate as labeled examples become available. It
evaluates each window **before** using that window's feedback for optimization.
Use it when you want to measure the performance of an adapting system on an
ordered task stream, rather than the final candidate's performance on a fixed
validation set.

## Evaluate, then update

For each window, the controller:

1. Freezes the current candidate and evaluates it on the new examples.
2. Records those scores, the example IDs, and the candidate that produced them.
3. Calls the existing `optimize` function using only already observed examples.
4. Uses the selected candidate for the next window.

For example, with a window size of 20, examples 0–19 are evaluated using the
initial candidate. Only then can their feedback change the candidate used on
examples 20–39. Neither the proposer nor its candidate-selection step receives
examples 20–39 during the first update.

An `EvaluationPolicy` controls validation sampling and candidate ranking; it
does not control when training examples become visible. The separate online
entrypoint owns that ordering while reusing GEPA's offline optimizer unchanged.

## Use an existing adapter

Pass a `GEPAAdapter` for your task. The adapter evaluates candidates and builds
reflection feedback just as it does for `optimize`:

```python
from gepa import optimize_online

result = optimize_online(
    seed_candidate={"instructions": "Solve the task and explain your answer."},
    data=ordered_examples,
    adapter=adapter,
    window_size=20,
    history_size=100,
    max_metric_calls_per_window=500,
    optimize_kwargs={
        "reflection_lm": reflection_lm,
        "reflection_minibatch_size": 5,
        "cache_evaluation": True,
    },
    run_dir="runs/online-example",
)

print(result.online_score)
print(result.candidate)  # For subsequent unseen examples.
```

`ordered_examples`, `adapter`, and `reflection_lm` are your task data, adapter,
and reflection model. A custom candidate proposer can also be supplied through
`optimize_kwargs`, without a reflection model.

`history_size` bounds the training data available to each update, including the
current window. It defaults to `window_size`. A final partial window is processed
immediately and receives the same per-window optimization budget. The budget has
the same stopping semantics as `optimize(max_metric_calls=...)`; the pre-update
evaluation is additional. The result reports both kinds of metric calls.

## Interpret the scores

`result.online_score` is the sum of all pre-update scores divided by the number
of evaluated examples. A partial window therefore contributes in proportion to
its size. Empty data produces `None`, rather than a performance estimate.

Each `OnlineWindowResult` records the evaluated candidate, example IDs, scores,
the next candidate, and evaluation/optimization metric calls. Training scores
used to select the next candidate do not replace the online scores. In
particular, `online_score` is **not** an estimate of the final candidate's score:
different windows may have been evaluated with different candidates.

This protocol prevents the controller from optimizing on a window before
measuring it. It does not make correlated or selectively labeled data
representative of future deployment, guarantee that updates improve performance,
or prevent an adapter from independently learning during evaluation. Use an
adapter whose evaluation runs do not train on their own feedback.

## Continue after more data arrives

The input can be a list or a `DataLoader`. Each call snapshots the loader's
ordered IDs. With a `run_dir`, call again after appending new examples to process
only the new suffix. IDs must be unique and stable, and existing examples must
remain unchanged. For a list, positions are the IDs. Removing or reordering a
processed ID prefix is rejected; changing data behind an existing ID violates
the loader contract and cannot be detected from IDs alone.

`max_windows` limits updates in one invocation. A subsequent call with the same
checkpoint can continue. Training history is bounded, but per-example scores,
IDs, and update checkpoints are retained, so storage grows over the run.

## Recover an interrupted update

The controller persists a window's pre-update result before starting its offline
optimization. If that update fails, the next call reuses the recorded result and
resumes the update in its own GEPA run directory. An interrupted checkpoint
replacement leaves the previous checkpoint intact.

Use a dedicated directory with a single writer. Keep the initial candidate,
window size, history size, and budget unchanged when resuming; mismatches are
rejected. Keep the adapter and other optimizer settings consistent for an
interrupted update as well.

Checkpoints use pickle, as GEPA's optimizer state does, and must come from a
trusted source. They do not serialize arbitrary external adapter state. An
evaluation interrupted before its result is persisted may execute again; the
controller does not guarantee exactly-once external model or tool calls.
