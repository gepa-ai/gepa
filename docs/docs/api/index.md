# API Reference

Welcome to the GEPA API Reference. This documentation is auto-generated from the source code docstrings.

## Core

The core module contains the main optimization function and fundamental classes.

- [`optimize`](core/optimize.md) - Main optimization function
- [`GEPAAdapter`](core/GEPAAdapter.md) - Adapter protocol for integrating GEPA with your system
- [`EvaluationBatch`](core/EvaluationBatch.md) - Container for evaluation results
- [`GEPAResult`](core/GEPAResult.md) - Result container from optimization
- [`GEPACallback`](core/GEPACallback.md) - Callback protocol for optimization events
- [`DataLoader`](core/DataLoader.md) - Data loading protocol

## Stop Conditions

Stop conditions control when optimization terminates.

- [`StopperProtocol`](stop_conditions/StopperProtocol.md) - Protocol for custom stop conditions
- [`MaxMetricCallsStopper`](stop_conditions/MaxMetricCallsStopper.md) - Stop after N metric evaluations
- [`TimeoutStopCondition`](stop_conditions/TimeoutStopCondition.md) - Stop after time limit
- [`NoImprovementStopper`](stop_conditions/NoImprovementStopper.md) - Stop if no improvement
- [`ScoreThresholdStopper`](stop_conditions/ScoreThresholdStopper.md) - Stop when score threshold reached
- [`FileStopper`](stop_conditions/FileStopper.md) - Stop when file exists
- [`SignalStopper`](stop_conditions/SignalStopper.md) - Stop on OS signal
- [`CompositeStopper`](stop_conditions/CompositeStopper.md) - Combine multiple stop conditions

## Adapters

Adapters integrate GEPA with different systems and frameworks.

- [`DefaultAdapter`](adapters/DefaultAdapter.md) - Default single-turn LLM adapter
- [`RAGAdapter`](adapters/RAGAdapter.md) - Generic RAG system adapter
- [`MCPAdapter`](adapters/MCPAdapter.md) - Model Context Protocol adapter

## Logging

Logging utilities for tracking optimization progress.

- [`LoggerProtocol`](logging/LoggerProtocol.md) - Protocol for custom loggers
- [`ExperimentTracker`](logging/ExperimentTracker.md) - MLflow/WandB integration

## Strategies

Strategies for various aspects of the optimization process.

- [`BatchSampler`](strategies/BatchSampler.md) - Training batch sampling
- [`CandidateSelector`](strategies/CandidateSelector.md) - Candidate selection for mutation
- [`EvaluationPolicy`](strategies/EvaluationPolicy.md) - Validation evaluation policy
