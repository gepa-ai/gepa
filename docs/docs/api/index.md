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
- [`DSPyAdapter`](adapters/DSPyAdapter.md) - DSPy program adapter
- [`DSPyFullProgramAdapter`](adapters/DSPyFullProgramAdapter.md) - DSPy full program evolution adapter
- [`RAGAdapter`](adapters/RAGAdapter.md) - Generic RAG system adapter
- [`MCPAdapter`](adapters/MCPAdapter.md) - Model Context Protocol adapter
- [`TerminalBenchAdapter`](adapters/TerminalBenchAdapter.md) - Terminal benchmark adapter

## Proposers

Proposers generate new candidate programs during optimization.

- [`CandidateProposal`](proposers/CandidateProposal.md) - Data class for candidate proposals
- [`ProposeNewCandidate`](proposers/ProposeNewCandidate.md) - Protocol for proposer strategies
- [`ReflectiveMutationProposer`](proposers/ReflectiveMutationProposer.md) - LLM-based reflective mutation proposer
- [`MergeProposer`](proposers/MergeProposer.md) - Merge-based candidate proposer
- [`Signature`](proposers/Signature.md) - Base class for LLM prompt signatures
- [`LanguageModel`](proposers/LanguageModel.md) - Protocol for language models

## Logging

Logging utilities for tracking optimization progress.

- [`LoggerProtocol`](logging/LoggerProtocol.md) - Protocol for custom loggers
- [`ExperimentTracker`](logging/ExperimentTracker.md) - MLflow/WandB integration

## Strategies

Strategies for various aspects of the optimization process.

### Batch Sampling

- [`BatchSampler`](strategies/BatchSampler.md) - Protocol for batch sampling
- [`EpochShuffledBatchSampler`](strategies/EpochShuffledBatchSampler.md) - Epoch-based shuffled batch sampler

### Candidate Selection

- [`CandidateSelector`](strategies/CandidateSelector.md) - Protocol for candidate selection
- [`ParetoCandidateSelector`](strategies/ParetoCandidateSelector.md) - Selects from Pareto front
- [`CurrentBestCandidateSelector`](strategies/CurrentBestCandidateSelector.md) - Selects current best candidate
- [`EpsilonGreedyCandidateSelector`](strategies/EpsilonGreedyCandidateSelector.md) - Epsilon-greedy selection

### Component Selection

- [`ComponentSelector`](strategies/ComponentSelector.md) - Protocol for component selection
- [`RoundRobinComponentSelector`](strategies/RoundRobinComponentSelector.md) - Round-robin component selection
- [`AllComponentSelector`](strategies/AllComponentSelector.md) - Selects all components

### Evaluation Policy

- [`EvaluationPolicy`](strategies/EvaluationPolicy.md) - Protocol for evaluation policies
- [`FullEvaluationPolicy`](strategies/FullEvaluationPolicy.md) - Evaluates all validation instances

### Instruction Proposal

- [`InstructionProposalSignature`](strategies/InstructionProposalSignature.md) - Signature for instruction proposal prompts
