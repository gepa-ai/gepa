# Using Callbacks

GEPA provides a powerful callback system for observing and instrumenting optimization runs. Callbacks allow you to monitor progress, log custom metrics, implement early stopping, or integrate with external systems.

## Overview

Callbacks are synchronous, observational objects that receive events during optimization. They cannot modify the optimization state but have full read access for maximum flexibility.

## Basic Usage

To use callbacks, create a class that implements one or more callback methods:

```python
import gepa

class MyCallback:
    def on_optimization_start(self, event):
        print(f"Starting optimization with {event['trainset_size']} training examples")
    
    def on_iteration_end(self, event):
        status = 'accepted' if event['proposal_accepted'] else 'rejected'
        print(f"Iteration {event['iteration']}: {status}")
    
    def on_optimization_end(self, event):
        print(f"Optimization complete! {event['total_iterations']} iterations")

# Use with optimize
result = gepa.optimize(
    seed_candidate={"instructions": "..."},
    trainset=data,
    callbacks=[MyCallback()],
    # ... other args
)
```

## Available Events

GEPA fires events at various points during optimization:

### Optimization Lifecycle

| Event | Description |
|-------|-------------|
| `on_optimization_start` | Called when optimization begins |
| `on_optimization_end` | Called when optimization completes |

### Iteration Lifecycle

| Event | Description |
|-------|-------------|
| `on_iteration_start` | Called at the start of each iteration |
| `on_iteration_end` | Called at the end of each iteration |

### Candidate Events

| Event | Description |
|-------|-------------|
| `on_candidate_selected` | When a candidate is selected for mutation |
| `on_candidate_accepted` | When a new candidate is accepted |
| `on_candidate_rejected` | When a candidate is rejected |

### Evaluation Events

| Event | Description |
|-------|-------------|
| `on_evaluation_start` | Before evaluating a candidate |
| `on_evaluation_end` | After evaluating a candidate |
| `on_valset_evaluated` | After validation set evaluation |

### Merge Events

| Event | Description |
|-------|-------------|
| `on_merge_attempted` | When a merge is attempted |
| `on_merge_accepted` | When a merge is accepted |
| `on_merge_rejected` | When a merge is rejected |

### State Events

| Event | Description |
|-------|-------------|
| `on_pareto_front_updated` | When the Pareto front changes |
| `on_state_saved` | After state is saved to disk |
| `on_budget_updated` | When evaluation budget changes |
| `on_error` | When an error occurs |

## Event Data

Each event is a TypedDict containing relevant information. For example:

### OptimizationStartEvent

```python
{
    "seed_candidate": dict[str, str],   # Initial candidate
    "trainset_size": int,                # Number of training examples
    "valset_size": int,                  # Number of validation examples
    "config": dict[str, Any],            # Configuration options
}
```

### IterationEndEvent

```python
{
    "iteration": int,           # Current iteration number
    "state": GEPAState,         # Full optimization state (read-only)
    "proposal_accepted": bool,  # Whether the proposal was accepted
}
```

### ValsetEvaluatedEvent

```python
{
    "iteration": int,
    "candidate_idx": int,
    "candidate": dict[str, str],
    "scores_by_val_id": dict[Any, float],
    "average_score": float,
    "num_examples_evaluated": int,
    "total_valset_size": int,
    "parent_ids": list[int],
    "is_best_program": bool,
    "outputs_by_val_id": dict[Any, Any] | None,
}
```

See the [API Reference](../api/callbacks/GEPACallback.md) for complete event specifications.

## Practical Examples

### Progress Tracking

```python
class ProgressCallback:
    def __init__(self):
        self.best_score = float('-inf')
        self.improvements = []
    
    def on_valset_evaluated(self, event):
        if event['is_best_program']:
            improvement = event['average_score'] - self.best_score
            self.best_score = event['average_score']
            self.improvements.append({
                'iteration': event['iteration'],
                'score': event['average_score'],
                'improvement': improvement
            })
            print(f"New best at iteration {event['iteration']}: {event['average_score']:.4f} (+{improvement:.4f})")
```

### Custom Logging

```python
import json
from pathlib import Path

class JSONLoggerCallback:
    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.events = []
    
    def on_iteration_end(self, event):
        self.events.append({
            'iteration': event['iteration'],
            'accepted': event['proposal_accepted'],
            'num_candidates': len(event['state'].program_candidates),
        })
    
    def on_optimization_end(self, event):
        with open(self.log_path, 'w') as f:
            json.dump(self.events, f, indent=2)
```

### Integration with External Systems

```python
class SlackNotifier:
    def __init__(self, webhook_url: str, notify_every: int = 10):
        self.webhook_url = webhook_url
        self.notify_every = notify_every
    
    def on_iteration_end(self, event):
        if event['iteration'] % self.notify_every == 0:
            self._send_slack_message(
                f"GEPA iteration {event['iteration']}: "
                f"{len(event['state'].program_candidates)} candidates"
            )
    
    def on_optimization_end(self, event):
        self._send_slack_message(
            f"GEPA optimization complete! "
            f"Total iterations: {event['total_iterations']}"
        )
    
    def _send_slack_message(self, message):
        import requests
        requests.post(self.webhook_url, json={"text": message})
```

### Checkpointing

```python
class CheckpointCallback:
    def __init__(self, checkpoint_dir: str, save_every: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every
    
    def on_iteration_end(self, event):
        if event['iteration'] % self.save_every == 0:
            state = event['state']
            checkpoint_path = self.checkpoint_dir / f"checkpoint_{event['iteration']}.json"
            
            # Save best candidates
            best_idx = state.program_full_scores_val_set.index(
                max(state.program_full_scores_val_set)
            )
            checkpoint_data = {
                'iteration': event['iteration'],
                'best_candidate': state.program_candidates[best_idx],
                'best_score': max(state.program_full_scores_val_set),
                'num_candidates': len(state.program_candidates),
            }
            
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
```

## CompositeCallback

Use `CompositeCallback` to combine multiple callbacks:

```python
from gepa.core.callbacks import CompositeCallback

callbacks = CompositeCallback([
    ProgressCallback(),
    JSONLoggerCallback("./logs/optimization.json"),
    CheckpointCallback("./checkpoints"),
])

result = gepa.optimize(
    # ... args ...
    callbacks=[callbacks],  # or just pass the list directly
)
```

## Accessing Full State

Callbacks receive the full `GEPAState` object, giving you access to:

- `state.program_candidates` - All discovered candidates
- `state.prog_candidate_val_subscores` - Validation scores per candidate
- `state.pareto_front_valset` - Current Pareto frontier
- `state.total_num_evals` - Total evaluation count
- And more...

```python
class StateInspector:
    def on_iteration_end(self, event):
        state = event['state']
        
        # Get Pareto front candidates
        pareto_candidates = set()
        for front in state.program_at_pareto_front_valset.values():
            pareto_candidates.update(front)
        
        print(f"Pareto front size: {len(pareto_candidates)}")
        print(f"Total candidates: {len(state.program_candidates)}")
```

## Best Practices

1. **Keep callbacks lightweight** - Callbacks run synchronously, so avoid expensive operations
2. **Handle exceptions gracefully** - Callback errors are logged but won't stop optimization
3. **Use the right granularity** - Choose events that match your monitoring needs
4. **Avoid modifying state** - Callbacks should be observational only

## Next Steps

- See the [API Reference](../api/callbacks/GEPACallback.md) for complete callback protocol
- Check out [Event Types](../api/callbacks/OptimizationStartEvent.md) for event details
- Learn about [Experiment Tracking](../api/logging/ExperimentTracker.md) for built-in logging
