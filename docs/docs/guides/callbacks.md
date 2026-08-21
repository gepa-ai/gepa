# Using Callbacks

GEPA provides a callback system for observing and instrumenting optimization runs.  Callbacks receive events at every stage of the loop — you can log custom metrics, stream results to external systems, add dynamic training data, inspect every LM call, or implement domain-specific monitoring.

!!! note "Mostly observational, with sanctioned mutation points"
    Callbacks receive the **live** optimization state, not a copy.  Most events are intended for observation only, but a few fields are explicitly designed to support mutation:

    - `event["trainset_loader"]` in `on_iteration_start` — call `.add_items()` to grow the training set mid-run (see [Dynamic training data](#5-dynamic-training-data-add-examples-mid-run) below).

    Mutating `event["state"]` or other fields directly outside these sanctioned patterns is unsupported and may cause undefined behaviour.
    Exceptions in callbacks are caught, logged as warnings, and never crash the optimization.

---

## Quick Start

```python
import gepa

class MyCallback:
    def on_optimization_start(self, event):
        print(f"Starting: {event['trainset_size']} train / {event['valset_size']} val examples")

    def on_iteration_end(self, event):
        status = "✓" if event["proposal_accepted"] else "✗"
        print(f"  {status} iteration {event['iteration']}")

    def on_optimization_end(self, event):
        print(f"Done — {event['total_iterations']} iterations, "
              f"{event['total_metric_calls']} metric calls")

result = gepa.optimize(
    seed_candidate={"system_prompt": "You are a helpful assistant."},
    trainset=trainset,
    valset=valset,
    adapter=adapter,
    reflection_lm="openai/gpt-5",
    max_metric_calls=200,
    callbacks=[MyCallback()],
)
```

---

## Complete Event Reference

GEPA fires 21 event types.  Each is a `TypedDict` — access fields with `event["field"]`.

### Optimization lifecycle

| Method | When | Key fields |
|--------|------|-----------|
| `on_optimization_start` | Before the first iteration | `seed_candidate`, `trainset_size`, `valset_size`, `config` |
| `on_optimization_end` | After the last iteration | `best_candidate_idx`, `total_iterations`, `total_metric_calls`, `final_state` |

### Iteration lifecycle

| Method | When | Key fields |
|--------|------|-----------|
| `on_iteration_start` | Start of each iteration | `iteration`, `state`, `trainset_loader` |
| `on_iteration_end` | End of each iteration | `iteration`, `state`, `proposal_accepted` |

### Candidate events

| Method | When | Key fields |
|--------|------|-----------|
| `on_candidate_selected` | A candidate is chosen for mutation | `iteration`, `candidate_idx`, `candidate`, `score` |
| `on_candidate_accepted` | A proposal passes the subsample acceptance test | `iteration`, `new_candidate_idx`, `new_score`, `parent_ids` |
| `on_candidate_rejected` | A proposal fails the subsample test | `iteration`, `old_score`, `new_score`, `reason` |
| `on_minibatch_sampled` | Training minibatch selected | `iteration`, `minibatch_ids`, `trainset_size` |

### Evaluation events

| Method | When | Key fields |
|--------|------|-----------|
| `on_evaluation_start` | Before adapter.evaluate() | `iteration`, `candidate_idx`, `batch_size`, `capture_traces`, `inputs`, `is_seed_candidate` |
| `on_evaluation_end` | After adapter.evaluate() | `iteration`, `candidate_idx`, `scores`, `outputs`, `trajectories`, `objective_scores`, `is_seed_candidate` |
| `on_evaluation_skipped` | Evaluation skipped (no trajectories / perfect score) | `iteration`, `candidate_idx`, `reason`, `scores` |
| `on_valset_evaluated` | After a candidate is scored on the full validation set | `iteration`, `candidate_idx`, `candidate`, `scores_by_val_id`, `average_score`, `is_best_program`, `outputs_by_val_id` |

### Reflection events

| Method | When | Key fields |
|--------|------|-----------|
| `on_reflective_dataset_built` | Reflective dataset assembled | `iteration`, `candidate_idx`, `components`, `dataset` |
| `on_proposal_start` | Before calling the reflection LM | `iteration`, `parent_candidate`, `components`, `reflective_dataset` |
| `on_proposal_end` | After the reflection LM responds | `iteration`, `new_instructions`, `prompts`, `raw_lm_outputs` |

### Merge events

| Method | When | Key fields |
|--------|------|-----------|
| `on_merge_attempted` | Merge proposer generates a candidate | `iteration`, `parent_ids`, `merged_candidate` |
| `on_merge_accepted` | Merged candidate passes acceptance test | `iteration`, `new_candidate_idx`, `parent_ids` |
| `on_merge_rejected` | Merged candidate fails acceptance test | `iteration`, `parent_ids`, `reason` |

### State / budget events

| Method | When | Key fields |
|--------|------|-----------|
| `on_pareto_front_updated` | Pareto front changes | `iteration`, `new_front`, `displaced_candidates` |
| `on_state_saved` | Checkpoint written to disk | `iteration`, `run_dir` |
| `on_budget_updated` | Evaluation budget changes | `iteration`, `metric_calls_used`, `metric_calls_delta`, `metric_calls_remaining` |
| `on_error` | Exception during iteration | `iteration`, `exception`, `will_continue` |

---

## Cookbook

### 1. Live progress display

```python
class ProgressCallback:
    def __init__(self):
        self.best_score = 0.0

    def on_valset_evaluated(self, event):
        if event["is_best_program"]:
            delta = event["average_score"] - self.best_score
            self.best_score = event["average_score"]
            print(f"[iter {event['iteration']}] New best: {self.best_score:.4f}  (+{delta:.4f})")
```

### 2. Inspect every LM call — prompt and raw response

`on_proposal_end` exposes what was sent to the reflection LM and its exact response before instruction extraction.

```python
class LMCallLogger:
    def on_proposal_end(self, event):
        for component, prompt in event["prompts"].items():
            print(f"\n=== Reflection LM call — component: {component} ===")
            if isinstance(prompt, str):
                print("PROMPT (last 500 chars):", prompt[-500:])
            print("RAW OUTPUT:", event["raw_lm_outputs"].get(component, "")[:500])
            print("EXTRACTED:", event["new_instructions"].get(component, "")[:200])
```

### 3. Identify which LM call produced each accepted candidate

```python
class ProposalTracker:
    """Links accepted candidate indices to their LM calls."""

    def __init__(self):
        self._pending: dict = {}   # iteration → {component: ...}
        self.accepted: list[dict] = []
        self.rejected: list[dict] = []

    def on_proposal_end(self, event):
        self._pending[event["iteration"]] = {
            "prompts": event["prompts"],
            "raw_lm_outputs": event["raw_lm_outputs"],
            "new_instructions": event["new_instructions"],
        }

    def on_candidate_accepted(self, event):
        data = self._pending.pop(event["iteration"], {})
        self.accepted.append({
            "candidate_idx": event["new_candidate_idx"],
            "iteration": event["iteration"],
            "subsample_score": event["new_score"],
            **data,
        })

    def on_candidate_rejected(self, event):
        data = self._pending.pop(event["iteration"], {})
        self.rejected.append({
            "iteration": event["iteration"],
            "reason": event["reason"],
            **data,
        })
```

### 4. Stream every accepted prompt to a file

```python
import json

class PromptArchive:
    def __init__(self, path: str = "prompts.jsonl"):
        self.path = path

    def on_valset_evaluated(self, event):
        # Only write when a new best is found
        if not event["is_best_program"]:
            return
        record = {
            "iteration": event["iteration"],
            "candidate_idx": event["candidate_idx"],
            "score": event["average_score"],
            "candidate": event["candidate"],
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
```

### 5. Dynamic training data — add examples mid-run

`on_iteration_start` provides `trainset_loader`, a mutable reference to the training set.  Add new examples and they will be included in subsequent minibatches.

```python
class ActiveLearner:
    """Add hard examples from the valset to the trainset when we find a new best."""

    def __init__(self, hard_threshold: float = 0.3):
        self.hard_threshold = hard_threshold
        self._last_best = None

    def on_valset_evaluated(self, event):
        if event["is_best_program"] and event["outputs_by_val_id"]:
            self._last_best = event

    def on_iteration_start(self, event):
        if self._last_best is None:
            return
        scores = self._last_best["scores_by_val_id"]
        hard_ids = [vid for vid, s in scores.items() if s < self.hard_threshold]
        if hard_ids:
            # Add the raw inputs for the hard examples to the training set
            hard_examples = [{"id": vid} for vid in hard_ids]
            event["trainset_loader"].add_items(hard_examples)
            print(f"Added {len(hard_ids)} hard examples to trainset")
        self._last_best = None
```

### 6. Budget-aware early stopping

```python
class CostGuard:
    """Stop when remaining budget drops below a threshold."""

    def __init__(self, stop_at_remaining: int = 20):
        self.stop_at = stop_at_remaining

    def on_budget_updated(self, event):
        remaining = event.get("metric_calls_remaining")
        if remaining is not None and remaining <= self.stop_at:
            print(f"Budget guard: only {remaining} calls left, signalling stop.")
            # Write the stop file if run_dir is set
            import os
            # (The FileStopper already handles this if run_dir is configured.)
```

### 7. Slack / webhook notifications

```python
import requests

class SlackNotifier:
    def __init__(self, webhook_url: str):
        self.webhook = webhook_url

    def _post(self, text: str) -> None:
        try:
            requests.post(self.webhook, json={"text": text}, timeout=5)
        except Exception:
            pass

    def on_valset_evaluated(self, event):
        if event["is_best_program"]:
            self._post(
                f":tada: *GEPA* new best at iteration {event['iteration']}: "
                f"`{event['average_score']:.4f}`"
            )

    def on_optimization_end(self, event):
        self._post(
            f":checkered_flag: *GEPA* finished — "
            f"{event['total_iterations']} iterations, "
            f"{event['total_metric_calls']} metric calls."
        )
```

### 8. Per-example output collection

```python
class OutputCollector:
    """Collect model outputs on the validation set for each new best candidate."""

    def __init__(self):
        self.snapshots: list[dict] = []

    def on_valset_evaluated(self, event):
        if event["is_best_program"] and event["outputs_by_val_id"]:
            self.snapshots.append({
                "iteration": event["iteration"],
                "candidate_idx": event["candidate_idx"],
                "score": event["average_score"],
                "outputs": dict(event["outputs_by_val_id"]),
            })
```

!!! note
    `outputs_by_val_id` is only populated when `track_best_outputs=True` is set in `EngineConfig`.

### 9. Error monitoring

```python
import traceback

class ErrorMonitor:
    def __init__(self):
        self.errors: list[dict] = []

    def on_error(self, event):
        self.errors.append({
            "iteration": event["iteration"],
            "error": str(event["exception"]),
            "will_continue": event["will_continue"],
            "traceback": traceback.format_exception(type(event["exception"]),
                                                     event["exception"],
                                                     event["exception"].__traceback__),
        })
        print(f"[ERROR] iteration {event['iteration']}: {event['exception']}")
```

---

## Combining Multiple Callbacks

Pass a list directly — GEPA calls each in order:

```python
result = gepa.optimize(
    ...
    callbacks=[
        ProgressCallback(),
        LMCallLogger(),
        SlackNotifier(WEBHOOK_URL),
    ],
)
```

Or use `CompositeCallback` to bundle them:

```python
from gepa.core.callbacks import CompositeCallback

bundle = CompositeCallback([ProgressCallback(), SlackNotifier(WEBHOOK_URL)])
bundle.add(ErrorMonitor())   # add at runtime

result = gepa.optimize(..., callbacks=[bundle])
```

---

## Accessing the Full Optimization State

Every `on_iteration_*` event includes the live `GEPAState`.  Useful fields:

```python
def on_iteration_end(self, event):
    state = event["state"]

    # All candidate texts
    state.program_candidates          # list[dict[str, str]]

    # Validation scores per candidate
    state.prog_candidate_val_subscores  # list[dict[val_id, float]]

    # Aggregate val scores (one float per candidate)
    state.program_full_scores_val_set   # property → list[float]

    # Current Pareto front (val_id → best score)
    state.pareto_front_valset           # dict[val_id, float]

    # Which candidate(s) are best per val example
    state.program_at_pareto_front_valset  # dict[val_id, set[int]]

    # Lineage
    state.parent_program_for_candidate  # list[list[int | None]]

    # Budget
    state.total_num_evals              # int
```

---

## Tips

- **Keep callbacks fast** — they run synchronously between iterations.  Defer heavy work (network calls, file I/O) to a background thread or queue.
- **Exceptions are swallowed** — a crashing callback never stops optimization.  Check your logs if a callback seems inactive.
- **No-op by default** — only implement the methods you need; missing methods are silently skipped.
- **`on_proposal_end` is your observability window** — it's the only event with the raw LM prompt and response.  Use it for prompt debugging, audit logs, and fine-tuning data collection.

---

## API Reference

- [`GEPACallback`](../api/callbacks/GEPACallback.md) — full callback protocol
- [`CompositeCallback`](../api/callbacks/CompositeCallback.md)
- Individual event types: [`OptimizationStartEvent`](../api/callbacks/OptimizationStartEvent.md), [`ValsetEvaluatedEvent`](../api/callbacks/ValsetEvaluatedEvent.md), and [all others](../api/callbacks/GEPACallback.md)
