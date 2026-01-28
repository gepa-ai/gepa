#!/usr/bin/env python3
"""
Generate API documentation for GEPA using mkdocstrings.

This script creates markdown files with mkdocstrings directives for
automatic API documentation generation.
"""

from pathlib import Path

# API documentation mapping
# Maps category -> list of (module_path, class_or_function_name, display_name)
API_MAPPING = {
    "core": [
        ("gepa.api", "optimize", "optimize"),
        ("gepa.core.adapter", "GEPAAdapter", "GEPAAdapter"),
        ("gepa.core.adapter", "EvaluationBatch", "EvaluationBatch"),
        ("gepa.core.result", "GEPAResult", "GEPAResult"),
        ("gepa.core.callbacks", "GEPACallback", "GEPACallback"),
        ("gepa.core.data_loader", "DataLoader", "DataLoader"),
    ],
    "stop_conditions": [
        ("gepa.utils.stop_condition", "StopperProtocol", "StopperProtocol"),
        ("gepa.utils.stop_condition", "MaxMetricCallsStopper", "MaxMetricCallsStopper"),
        ("gepa.utils.stop_condition", "TimeoutStopCondition", "TimeoutStopCondition"),
        ("gepa.utils.stop_condition", "NoImprovementStopper", "NoImprovementStopper"),
        ("gepa.utils.stop_condition", "ScoreThresholdStopper", "ScoreThresholdStopper"),
        ("gepa.utils.stop_condition", "FileStopper", "FileStopper"),
        ("gepa.utils.stop_condition", "SignalStopper", "SignalStopper"),
        ("gepa.utils.stop_condition", "CompositeStopper", "CompositeStopper"),
    ],
    "adapters": [
        ("gepa.adapters.default_adapter.default_adapter", "DefaultAdapter", "DefaultAdapter"),
        ("gepa.adapters.dspy_adapter.dspy_adapter", "DspyAdapter", "DSPyAdapter"),
        ("gepa.adapters.dspy_full_program_adapter.full_program_adapter", "DspyAdapter", "DSPyFullProgramAdapter"),
        ("gepa.adapters.generic_rag_adapter.generic_rag_adapter", "GenericRAGAdapter", "RAGAdapter"),
        ("gepa.adapters.mcp_adapter.mcp_adapter", "MCPAdapter", "MCPAdapter"),
        ("gepa.adapters.terminal_bench_adapter.terminal_bench_adapter", "TerminusAdapter", "TerminalBenchAdapter"),
        ("gepa.adapters.anymaths_adapter.anymaths_adapter", "AnyMathsAdapter", "AnyMathsAdapter"),
    ],
    "proposers": [
        ("gepa.proposer.base", "CandidateProposal", "CandidateProposal"),
        ("gepa.proposer.base", "ProposeNewCandidate", "ProposeNewCandidate"),
        (
            "gepa.proposer.reflective_mutation.reflective_mutation",
            "ReflectiveMutationProposer",
            "ReflectiveMutationProposer",
        ),
        ("gepa.proposer.merge", "MergeProposer", "MergeProposer"),
        ("gepa.proposer.reflective_mutation.base", "Signature", "Signature"),
        ("gepa.proposer.reflective_mutation.base", "LanguageModel", "LanguageModel"),
    ],
    "logging": [
        ("gepa.logging.logger", "LoggerProtocol", "LoggerProtocol"),
        ("gepa.logging.experiment_tracker", "ExperimentTracker", "ExperimentTracker"),
    ],
    "strategies": [
        ("gepa.strategies.batch_sampler", "BatchSampler", "BatchSampler"),
        ("gepa.strategies.batch_sampler", "EpochShuffledBatchSampler", "EpochShuffledBatchSampler"),
        ("gepa.proposer.reflective_mutation.base", "CandidateSelector", "CandidateSelector"),
        ("gepa.strategies.candidate_selector", "ParetoCandidateSelector", "ParetoCandidateSelector"),
        ("gepa.strategies.candidate_selector", "CurrentBestCandidateSelector", "CurrentBestCandidateSelector"),
        ("gepa.strategies.candidate_selector", "EpsilonGreedyCandidateSelector", "EpsilonGreedyCandidateSelector"),
        ("gepa.proposer.reflective_mutation.base", "ReflectionComponentSelector", "ComponentSelector"),
        ("gepa.strategies.component_selector", "RoundRobinReflectionComponentSelector", "RoundRobinComponentSelector"),
        ("gepa.strategies.component_selector", "AllReflectionComponentSelector", "AllComponentSelector"),
        ("gepa.strategies.eval_policy", "EvaluationPolicy", "EvaluationPolicy"),
        ("gepa.strategies.eval_policy", "FullEvaluationPolicy", "FullEvaluationPolicy"),
        ("gepa.strategies.instruction_proposal", "InstructionProposalSignature", "InstructionProposalSignature"),
    ],
}


def generate_api_doc(module_path: str, name: str, display_name: str) -> str:
    """Generate markdown content for a single API item."""
    return f"""# {display_name}

::: {module_path}.{name}
    handler: python
    options:
        show_source: true
        show_root_heading: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false
        inherited_members: true
        members_order: source
        show_signature_annotations: true
"""


def generate_category_index(category: str, items: list) -> str:
    """Generate index page for a category."""
    title = category.replace("_", " ").title()
    content = f"""# {title}

This section contains API documentation for GEPA {title.lower()}.

"""
    for module_path, name, display_name in items:
        content += f"- [{display_name}]({display_name}.md)\n"

    return content


def main():
    api_dir = Path("docs/api")
    api_dir.mkdir(parents=True, exist_ok=True)

    # Generate API index
    index_content = """# API Reference

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
"""
    (api_dir / "index.md").write_text(index_content)

    # Generate individual API docs
    for category, items in API_MAPPING.items():
        category_dir = api_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        for module_path, name, display_name in items:
            doc_content = generate_api_doc(module_path, name, display_name)
            (category_dir / f"{display_name}.md").write_text(doc_content)

    print("API documentation generated successfully!")


if __name__ == "__main__":
    main()
