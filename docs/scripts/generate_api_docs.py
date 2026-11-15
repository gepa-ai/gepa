#!/usr/bin/env python3
"""
Generate API documentation for GEPA using mkdocstrings.

This script creates markdown files with mkdocstrings directives for
automatic API documentation generation.

Features:
- Auto-generates API documentation from API_MAPPING
- Auto-generates mkdocs nav structure
- Validates consistency between API_MAPPING and generated files
"""

from pathlib import Path

import yaml

# API documentation mapping
# Maps category -> list of (module_path, class_or_function_name, display_name)
API_MAPPING = {
    "optimize_anything": [
        ("gepa.optimize_anything", "optimize_anything", "optimize_anything"),
        ("gepa.optimize_anything", "GEPAConfig", "GEPAConfig"),
        ("gepa.optimize_anything", "EngineConfig", "EngineConfig"),
        ("gepa.optimize_anything", "ReflectionConfig", "ReflectionConfig"),
        ("gepa.optimize_anything", "MergeConfig", "MergeConfig"),
        ("gepa.optimize_anything", "RefinerConfig", "RefinerConfig"),
        ("gepa.optimize_anything", "TrackingConfig", "TrackingConfig"),
        ("gepa.optimize_anything", "Evaluator", "Evaluator"),
        ("gepa.optimize_anything", "OptimizationState", "OptimizationState"),
        ("gepa.optimize_anything", "LogContext", "LogContext"),
        ("gepa.optimize_anything", "log", "log"),
        ("gepa.optimize_anything", "get_log_context", "get_log_context"),
        ("gepa.optimize_anything", "set_log_context", "set_log_context"),
        ("gepa.optimize_anything", "make_litellm_lm", "make_litellm_lm"),
    ],
    "core": [
        ("gepa.api", "optimize", "optimize"),
        ("gepa.core.adapter", "GEPAAdapter", "GEPAAdapter"),
        ("gepa.core.adapter", "EvaluationBatch", "EvaluationBatch"),
        ("gepa.core.result", "GEPAResult", "GEPAResult"),
        ("gepa.core.callbacks", "GEPACallback", "GEPACallback"),
        ("gepa.core.data_loader", "DataLoader", "DataLoader"),
        ("gepa.core.state", "GEPAState", "GEPAState"),
        ("gepa.core.state", "EvaluationCache", "EvaluationCache"),
    ],
    "callbacks": [
        ("gepa.core.callbacks", "GEPACallback", "GEPACallback"),
        ("gepa.core.callbacks", "CompositeCallback", "CompositeCallback"),
        ("gepa.core.callbacks", "OptimizationStartEvent", "OptimizationStartEvent"),
        ("gepa.core.callbacks", "OptimizationEndEvent", "OptimizationEndEvent"),
        ("gepa.core.callbacks", "IterationStartEvent", "IterationStartEvent"),
        ("gepa.core.callbacks", "IterationEndEvent", "IterationEndEvent"),
        ("gepa.core.callbacks", "CandidateSelectedEvent", "CandidateSelectedEvent"),
        ("gepa.core.callbacks", "CandidateAcceptedEvent", "CandidateAcceptedEvent"),
        ("gepa.core.callbacks", "CandidateRejectedEvent", "CandidateRejectedEvent"),
        ("gepa.core.callbacks", "EvaluationStartEvent", "EvaluationStartEvent"),
        ("gepa.core.callbacks", "EvaluationEndEvent", "EvaluationEndEvent"),
        ("gepa.core.callbacks", "ValsetEvaluatedEvent", "ValsetEvaluatedEvent"),
        ("gepa.core.callbacks", "ParetoFrontUpdatedEvent", "ParetoFrontUpdatedEvent"),
        ("gepa.core.callbacks", "MergeAttemptedEvent", "MergeAttemptedEvent"),
        ("gepa.core.callbacks", "MergeAcceptedEvent", "MergeAcceptedEvent"),
        ("gepa.core.callbacks", "MergeRejectedEvent", "MergeRejectedEvent"),
        ("gepa.core.callbacks", "BudgetUpdatedEvent", "BudgetUpdatedEvent"),
        ("gepa.core.callbacks", "ErrorEvent", "ErrorEvent"),
        ("gepa.core.callbacks", "StateSavedEvent", "StateSavedEvent"),
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
        ("gepa.logging.logger", "StdOutLogger", "StdOutLogger"),
        ("gepa.logging.logger", "Logger", "Logger"),
        ("gepa.logging.experiment_tracker", "ExperimentTracker", "ExperimentTracker"),
        ("gepa.logging.experiment_tracker", "create_experiment_tracker", "create_experiment_tracker"),
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

# Category display names and descriptions for index page
CATEGORY_INFO = {
    "optimize_anything": {
        "title": "optimize_anything",
        "description": "The primary public API for GEPA. Optimize any text artifact with LLM-guided evolution — bring a seed candidate and an evaluator, and GEPA handles the rest.",
    },
    "core": {
        "title": "Core",
        "description": "The core module contains the main optimization function and fundamental classes.",
    },
    "callbacks": {
        "title": "Callbacks",
        "description": "Callback system for observing and instrumenting GEPA optimization runs.",
    },
    "stop_conditions": {
        "title": "Stop Conditions",
        "description": "Stop conditions control when optimization terminates.",
    },
    "adapters": {
        "title": "Adapters",
        "description": "Adapters integrate GEPA with different systems and frameworks.",
    },
    "proposers": {
        "title": "Proposers",
        "description": "Proposers generate new candidate programs during optimization.",
    },
    "logging": {
        "title": "Logging",
        "description": "Logging utilities for tracking optimization progress.",
    },
    "strategies": {
        "title": "Strategies",
        "description": "Strategies for various aspects of the optimization process.",
    },
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
    info = CATEGORY_INFO.get(category, {"title": category.replace("_", " ").title(), "description": ""})
    content = f"""# {info["title"]}

{info["description"]}

"""
    for module_path, name, display_name in items:
        content += f"- [{display_name}]({display_name}.md)\n"

    return content


def generate_index_content() -> str:
    """Auto-generate the API index content from API_MAPPING."""
    content = """# API Reference

Welcome to the GEPA API Reference. This documentation is auto-generated from the source code docstrings.

"""
    for category, items in API_MAPPING.items():
        info = CATEGORY_INFO.get(category, {"title": category.replace("_", " ").title(), "description": ""})
        content += f"## {info['title']}\n\n"
        content += f"{info['description']}\n\n"
        for module_path, name, display_name in items:
            content += f"- [`{display_name}`]({category}/{display_name}.md)\n"
        content += "\n"

    return content


def generate_nav_structure() -> list:
    """Generate the nav structure for mkdocs.yml API Reference section."""
    nav = [{"API Overview": "api/index.md"}]

    for category, items in API_MAPPING.items():
        info = CATEGORY_INFO.get(category, {"title": category.replace("_", " ").title()})
        category_nav = {}
        category_entries = []

        for module_path, name, display_name in items:
            category_entries.append({display_name: f"api/{category}/{display_name}.md"})

        category_nav[info["title"]] = category_entries
        nav.append(category_nav)

    return nav


def validate_api_mapping(skip_adapters: bool = False):
    """Validate that all items in API_MAPPING can be imported.

    Args:
        skip_adapters: If True, skip import errors for gepa.adapters.* modules.
                       Adapters often depend on external packages (dspy, mcp, etc.)
                       that may not be installed in all environments.
    """
    errors = []
    skipped = []
    import importlib

    for category, items in API_MAPPING.items():
        for module_path, name, display_name in items:
            try:
                module = importlib.import_module(module_path)
                if not hasattr(module, name):
                    errors.append(f"Module {module_path} does not have attribute {name}")
            except ImportError as e:
                # Skip adapter import errors if requested (they have external deps)
                if skip_adapters and module_path.startswith("gepa.adapters."):
                    skipped.append(f"Skipped {module_path}: {e}")
                else:
                    errors.append(f"Cannot import {module_path}: {e}")

    return errors, skipped


def print_nav_yaml():
    """Print the nav structure as YAML for manual copy into mkdocs.yml."""
    nav = generate_nav_structure()
    print("\n# Auto-generated API Reference nav structure:")
    print("# Copy this into mkdocs.yml under the 'API Reference:' section\n")
    print("        - API Reference:")
    for item in nav:
        print(yaml.dump([item], default_flow_style=False, indent=12).rstrip())


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate GEPA API documentation")
    parser.add_argument("--validate", action="store_true", help="Validate API_MAPPING imports")
    parser.add_argument(
        "--skip-adapters",
        action="store_true",
        help="Skip import errors for gepa.adapters.* (they have external dependencies)",
    )
    parser.add_argument("--print-nav", action="store_true", help="Print nav structure for mkdocs.yml")
    args = parser.parse_args()

    if args.validate:
        print("Validating API_MAPPING...")
        errors, skipped = validate_api_mapping(skip_adapters=args.skip_adapters)
        if skipped:
            print(f"Skipped {len(skipped)} adapter imports (external dependencies):")
            for s in skipped:
                print(f"  - {s}")
        if errors:
            print("Validation errors:")
            for e in errors:
                print(f"  - {e}")
            return 1
        print("All API_MAPPING entries validated successfully!")
        return 0

    if args.print_nav:
        print_nav_yaml()
        return 0

    api_dir = Path("docs/api")
    api_dir.mkdir(parents=True, exist_ok=True)

    # Generate API index (auto-generated from API_MAPPING)
    index_content = generate_index_content()
    (api_dir / "index.md").write_text(index_content)

    # Generate individual API docs
    for category, items in API_MAPPING.items():
        category_dir = api_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        for module_path, name, display_name in items:
            doc_content = generate_api_doc(module_path, name, display_name)
            (category_dir / f"{display_name}.md").write_text(doc_content)

    print("API documentation generated successfully!")
    print(
        f"Generated docs for {sum(len(items) for items in API_MAPPING.values())} API items across {len(API_MAPPING)} categories"
    )

    # Print summary of what was generated
    for category, items in API_MAPPING.items():
        info = CATEGORY_INFO.get(category, {"title": category})
        print(f"  - {info['title']}: {len(items)} items")


if __name__ == "__main__":
    main()
