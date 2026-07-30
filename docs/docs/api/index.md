# API Reference

Welcome to the GEPA API Reference. This documentation is auto-generated from the source code docstrings.

## optimize_anything

The primary public API for GEPA. Define a task, provide an evaluator, and choose an optimization engine.

- [`optimize_anything`](optimize_anything/optimize_anything.md)
- [`Task`](optimize_anything/Task.md)
- [`OptimizeAnythingConfig`](optimize_anything/OptimizeAnythingConfig.md)
- [`Engine`](optimize_anything/Engine.md)
- [`Result`](optimize_anything/Result.md)
- [`EvalServer`](optimize_anything/EvalServer.md)
- [`BudgetTracker`](optimize_anything/BudgetTracker.md)
- [`optimize_anything_with_server`](optimize_anything/optimize_anything_with_server.md)
- [`list_engines`](optimize_anything/list_engines.md)
- [`register_engine`](optimize_anything/register_engine.md)

## GEPA Engine

Configuration for the built-in GEPA engine (`engine="gepa"`). These classes are passed via `OptimizeAnythingConfig(engine_config={...})` to control GEPA-specific behavior.

- [`GEPAConfig`](optimize_anything/GEPAConfig.md)
- [`EngineConfig`](optimize_anything/EngineConfig.md)
- [`ReflectionConfig`](optimize_anything/ReflectionConfig.md)
- [`MergeConfig`](optimize_anything/MergeConfig.md)
- [`RefinerConfig`](optimize_anything/RefinerConfig.md)
- [`TrackingConfig`](optimize_anything/TrackingConfig.md)
- [`Evaluator`](optimize_anything/Evaluator.md)
- [`OptimizationState`](optimize_anything/OptimizationState.md)
- [`LogContext`](optimize_anything/LogContext.md)
- [`log`](optimize_anything/log.md)
- [`get_log_context`](optimize_anything/get_log_context.md)
- [`set_log_context`](optimize_anything/set_log_context.md)
- [`make_litellm_lm`](optimize_anything/make_litellm_lm.md)

## Core

The core module contains the main optimization function and fundamental classes.

- [`optimize`](core/optimize.md)
- [`GEPAAdapter`](core/GEPAAdapter.md)
- [`EvaluationBatch`](core/EvaluationBatch.md)
- [`GEPAResult`](core/GEPAResult.md)
- [`GEPACallback`](core/GEPACallback.md)
- [`DataLoader`](core/DataLoader.md)
- [`GEPAState`](core/GEPAState.md)
- [`EvaluationCache`](core/EvaluationCache.md)

## Callbacks

Callback system for observing and instrumenting GEPA optimization runs.

- [`GEPACallback`](callbacks/GEPACallback.md)
- [`CompositeCallback`](callbacks/CompositeCallback.md)
- [`OptimizationStartEvent`](callbacks/OptimizationStartEvent.md)
- [`OptimizationEndEvent`](callbacks/OptimizationEndEvent.md)
- [`IterationStartEvent`](callbacks/IterationStartEvent.md)
- [`IterationEndEvent`](callbacks/IterationEndEvent.md)
- [`CandidateSelectedEvent`](callbacks/CandidateSelectedEvent.md)
- [`CandidateAcceptedEvent`](callbacks/CandidateAcceptedEvent.md)
- [`CandidateRejectedEvent`](callbacks/CandidateRejectedEvent.md)
- [`EvaluationStartEvent`](callbacks/EvaluationStartEvent.md)
- [`EvaluationEndEvent`](callbacks/EvaluationEndEvent.md)
- [`ValsetEvaluatedEvent`](callbacks/ValsetEvaluatedEvent.md)
- [`ParetoFrontUpdatedEvent`](callbacks/ParetoFrontUpdatedEvent.md)
- [`MergeAttemptedEvent`](callbacks/MergeAttemptedEvent.md)
- [`MergeAcceptedEvent`](callbacks/MergeAcceptedEvent.md)
- [`MergeRejectedEvent`](callbacks/MergeRejectedEvent.md)
- [`BudgetUpdatedEvent`](callbacks/BudgetUpdatedEvent.md)
- [`ErrorEvent`](callbacks/ErrorEvent.md)
- [`StateSavedEvent`](callbacks/StateSavedEvent.md)

## Stop Conditions

Stop conditions control when optimization terminates.

- [`StopperProtocol`](stop_conditions/StopperProtocol.md)
- [`MaxMetricCallsStopper`](stop_conditions/MaxMetricCallsStopper.md)
- [`TimeoutStopCondition`](stop_conditions/TimeoutStopCondition.md)
- [`NoImprovementStopper`](stop_conditions/NoImprovementStopper.md)
- [`ScoreThresholdStopper`](stop_conditions/ScoreThresholdStopper.md)
- [`FileStopper`](stop_conditions/FileStopper.md)
- [`SignalStopper`](stop_conditions/SignalStopper.md)
- [`CompositeStopper`](stop_conditions/CompositeStopper.md)

## Adapters

Adapters integrate GEPA with different systems and frameworks.

- [`DefaultAdapter`](adapters/DefaultAdapter.md)
- [`DSPyAdapter`](adapters/DSPyAdapter.md)
- [`DSPyFullProgramAdapter`](adapters/DSPyFullProgramAdapter.md)
- [`RAGAdapter`](adapters/RAGAdapter.md)
- [`MCPAdapter`](adapters/MCPAdapter.md)
- [`TerminalBenchAdapter`](adapters/TerminalBenchAdapter.md)

## Proposers

Proposers generate new candidate programs during optimization.

- [`CandidateProposal`](proposers/CandidateProposal.md)
- [`ProposeNewCandidate`](proposers/ProposeNewCandidate.md)
- [`ReflectiveMutationProposer`](proposers/ReflectiveMutationProposer.md)
- [`MergeProposer`](proposers/MergeProposer.md)
- [`Signature`](proposers/Signature.md)
- [`LanguageModel`](proposers/LanguageModel.md)

## Logging

Logging utilities for tracking optimization progress.

- [`LoggerProtocol`](logging/LoggerProtocol.md)
- [`StdOutLogger`](logging/StdOutLogger.md)
- [`Logger`](logging/Logger.md)
- [`ExperimentTracker`](logging/ExperimentTracker.md)
- [`create_experiment_tracker`](logging/create_experiment_tracker.md)

## Strategies

Strategies for various aspects of the optimization process.

- [`BatchSampler`](strategies/BatchSampler.md)
- [`EpochShuffledBatchSampler`](strategies/EpochShuffledBatchSampler.md)
- [`CandidateSelector`](strategies/CandidateSelector.md)
- [`ParetoCandidateSelector`](strategies/ParetoCandidateSelector.md)
- [`CurrentBestCandidateSelector`](strategies/CurrentBestCandidateSelector.md)
- [`EpsilonGreedyCandidateSelector`](strategies/EpsilonGreedyCandidateSelector.md)
- [`ComponentSelector`](strategies/ComponentSelector.md)
- [`RoundRobinComponentSelector`](strategies/RoundRobinComponentSelector.md)
- [`AllComponentSelector`](strategies/AllComponentSelector.md)
- [`EvaluationPolicy`](strategies/EvaluationPolicy.md)
- [`FullEvaluationPolicy`](strategies/FullEvaluationPolicy.md)
- [`InstructionProposalSignature`](strategies/InstructionProposalSignature.md)
