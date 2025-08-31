# Multi-Dimensional Scoring Implementation in GEPA Core

## Overview

This document describes the implementation of multi-dimensional scoring support in the GEPA core system. The goal was to modify the GEPA core (not just adapters) to track multiple scores per candidate in the Pareto frontier, enabling multi-objective optimization.

## Key Changes Made

### 1. Enhanced EvaluationBatch (`src/gepa/core/adapter.py`)

- **Added `multi_scores` field**: `list[dict[str, float]] | None = None`
- **Purpose**: Stores per-example multi-dimensional scores (e.g., `{"parser_accuracy": 0.8, "task_resolved": 1.0}`)
- **Backward compatibility**: Optional field, existing single-dimensional scoring continues to work

### 2. Enhanced GEPAState (`src/gepa/core/state.py`)

#### New Fields
- `program_multi_scores_val_set: list[list[dict[str, float]]]` - Multi-dimensional scores for each program candidate
- `prog_candidate_multi_scores: list[list[dict[str, float]]]` - Multi-dimensional subscores per validation instance

#### Constructor Updates
- **Added parameter**: `base_valset_multi_scores: list[dict[str, float]] | None = None`
- **Auto-initialization**: Creates default single-dimensional scores if multi-scores not provided
- **Consistency checks**: Updated `is_consistent()` method to validate multi-dimensional score fields

#### State Update Method
- **Enhanced `update_state_with_new_program()`**: Now accepts `valset_multi_scores` parameter
- **Multi-dimensional Pareto logic**: Implements proper dominance checking for multi-dimensional scores
- **Fallback support**: Maintains backward compatibility with single-dimensional scoring

### 3. Enhanced GEPAResult (`src/gepa/core/result.py`)

- **Added `val_multi_scores` field**: `list[list[dict[str, float]]] | None = None`
- **Serialization support**: Multi-dimensional scores included in `to_dict()` method
- **State extraction**: `from_state()` method extracts multi-dimensional scores from GEPAState

### 4. Enhanced GEPAEngine (`src/gepa/core/engine.py`)

- **Multi-score extraction**: `_run_full_eval_and_add()` method now extracts multi-dimensional scores from evaluation results
- **Flexible extraction**: Handles different output formats (EvaluationBatch, list of outputs)
- **State propagation**: Passes multi-dimensional scores to state update methods

## Multi-Dimensional Pareto Frontier Logic

### Dominance Checking
The system now implements proper multi-dimensional dominance:

```python
# Check if new score dominates old score (all dimensions are >= and at least one is >)
dominates = all(new_multi_score.get(k, 0.0) >= old_multi_score.get(k, 0.0) 
                for k in set(new_multi_score.keys()) | set(old_multi_score.keys()))
strictly_dominates = dominates and any(new_multi_score.get(k, 0.0) > old_multi_score.get(k, 0.0) 
                                      for k in set(new_multi_score.keys()) | set(old_multi_score.keys()))
```

### Pareto Front Updates
- **Strict dominance**: New candidate replaces existing ones on Pareto front
- **Non-dominance**: New candidate added to Pareto front (multiple non-dominated solutions)
- **Backward compatibility**: Single-dimensional logic preserved for existing use cases

## Usage Examples

### 1. Adapter Implementation
```python
def evaluate(self, batch, candidate, capture_traces=False):
    # ... evaluation logic ...
    
    # Create multi-dimensional scores
    multi_scores = [
        {"parser_accuracy": 0.8, "task_resolved": 1.0},
        {"parser_accuracy": 0.6, "task_resolved": 0.0},
        {"parser_accuracy": 0.9, "task_resolved": 1.0}
    ]
    
    return EvaluationBatch(
        outputs=outputs,
        scores=aggregated_scores,  # Single scores for backward compatibility
        multi_scores=multi_scores,  # Multi-dimensional scores
        trajectories=trajectories
    )
```

### 2. State Initialization
```python
state = GEPAState(
    seed_candidate=seed_candidate,
    base_valset_eval_output=base_output,
    base_valset_multi_scores=multi_scores,  # Optional
    track_best_outputs=False
)
```

### 3. State Updates
```python
new_program_idx, linear_pareto_front_program_idx = state.update_state_with_new_program(
    parent_program_idx=[0],
    new_program=new_program,
    valset_score=0.75,
    valset_outputs=outputs,
    valset_subscores=[0.85, 0.5, 0.9],
    valset_multi_scores=new_multi_scores,  # New parameter
    run_dir=None,
    num_metric_calls_by_discovery_of_new_program=3
)
```

## Backward Compatibility

### Single-Dimensional Scoring
- **No changes required**: Existing adapters continue to work unchanged
- **Auto-conversion**: Single scores automatically converted to `{"score": value}` format
- **Performance**: No overhead for single-dimensional use cases

### Existing APIs
- **Method signatures**: All existing method signatures preserved
- **Default values**: New parameters have sensible defaults
- **Error handling**: Graceful fallback to single-dimensional logic

## Benefits

### 1. Multi-Objective Optimization
- **Rich scoring**: Track multiple metrics simultaneously (e.g., accuracy, efficiency, fairness)
- **Better decisions**: Pareto frontier reflects true multi-dimensional trade-offs
- **Flexible aggregation**: Users can implement custom aggregation strategies

### 2. Enhanced Analysis
- **Detailed insights**: Understand performance across multiple dimensions
- **Trade-off analysis**: Identify candidates that excel in specific areas
- **Better debugging**: Pinpoint which metrics are causing failures

### 3. Future Extensibility
- **Scalable**: Easy to add new metrics without core changes
- **Plugin architecture**: Adapters can define their own metric sets
- **Research support**: Enables advanced multi-objective research

## Testing

The implementation includes comprehensive testing:
- **Unit tests**: All new functionality tested
- **Integration tests**: End-to-end multi-dimensional scoring workflow
- **Backward compatibility**: Single-dimensional scoring verified
- **Edge cases**: Error handling and boundary conditions tested

## Migration Guide

### For Existing Users
1. **No changes required**: Existing code continues to work
2. **Optional enhancement**: Add multi-dimensional scores when ready
3. **Gradual adoption**: Can migrate one metric at a time

### For New Users
1. **Implement multi-dimensional scoring** in your adapter's `evaluate` method
2. **Return `EvaluationBatch`** with `multi_scores` field populated
3. **GEPA automatically handles** the rest of the multi-dimensional logic

## Conclusion

The multi-dimensional scoring implementation successfully addresses the PhD student's requirements:

✅ **Core GEPA changes**: Multi-dimensional scoring integrated into core system  
✅ **Pareto frontier tracking**: Multiple scores tracked simultaneously  
✅ **Backward compatibility**: Existing functionality preserved  
✅ **Extensible design**: Easy to add new metrics and scoring strategies  

The system now provides a robust foundation for multi-objective optimization while maintaining the simplicity and reliability of the original GEPA architecture.
