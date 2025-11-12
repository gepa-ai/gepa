#!/usr/bin/env python3
"""
Example usage of MiniSWEAgentAdapter with GEPA.

This script demonstrates how to optimize mini-swe-agent configurations
on SWE-bench tasks using GEPA.

Usage:
    python example_usage.py --model anthropic/claude-sonnet-4 --subset verified --n-instances 5

Requirements:
    - mini-swe-agent (included in repo)
    - Docker (for running SWE-bench environments)
    - Optional: swebench (for validation)
"""

import argparse
from pathlib import Path

from gepa import GEPAEngine
from gepa.adapters.minisweagent_adapter import MiniSWEAgentAdapter, load_swebench_instances
from gepa.core.data_loader import DataLoaderList


def get_default_seed_candidate() -> dict[str, str]:
    """
    Get a default seed candidate with minimal but functional agent configuration.
    
    The candidate contains the text components that GEPA will optimize.
    These are merged with the base agent config (model, environment settings, etc.)
    to build the complete agent configuration.
    """
    return {
        "system_template": """You are a helpful assistant that can interact with a computer shell to solve programming tasks.
Your response must contain exactly ONE bash code block with ONE command.

Include a THOUGHT section before your command explaining your reasoning.

Format:
THOUGHT: Your reasoning here

```bash
your_command_here
```""",
        "instance_template": """<pr_description>
{{task}}
</pr_description>

<instructions>
You are a software engineer working on fixing an issue in a codebase.
Your task is to make changes to fix the issue described above.

Work interactively:
1. Explore the codebase
2. Understand the issue
3. Make necessary changes
4. Verify your fix

When complete, submit with:
```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached
```
</instructions>""",
    }


def main():
    parser = argparse.ArgumentParser(description="Optimize mini-swe-agent with GEPA")
    parser.add_argument(
        "--model",
        type=str,
        default="anthropic/claude-sonnet-4",
        help="Model name to use",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="verified",
        choices=["verified", "lite", "full"],
        help="SWE-bench subset",
    )
    parser.add_argument(
        "--n-instances",
        type=int,
        default=5,
        help="Number of instances to use (for quick testing)",
    )
    parser.add_argument(
        "--run-validation",
        action="store_true",
        help="Run SWE-bench validation (slow but accurate)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=5,
        help="Number of GEPA optimization iterations",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./minisweagent_gepa_output",
        help="Output directory for results",
    )
    parser.add_argument(
        "--environment",
        type=str,
        default="docker",
        choices=["docker", "singularity", "local"],
        help="Environment type",
    )
    
    args = parser.parse_args()
    
    # Map subset to dataset name
    dataset_map = {
        "verified": "princeton-nlp/SWE-bench_Verified",
        "lite": "princeton-nlp/SWE-bench_Lite",
        "full": "princeton-nlp/SWE-Bench",
    }
    dataset = dataset_map[args.subset]
    
    print(f"Loading {args.n_instances} instances from {dataset}...")
    
    # Load instances
    # Split into train (60%) and val (40%)
    n_train = int(args.n_instances * 0.6)
    n_val = args.n_instances - n_train
    
    train_instances = load_swebench_instances(
        dataset=dataset,
        split="test",  # Using test split for the example
        slice_spec=f"0:{n_train}",
    )
    
    val_instances = load_swebench_instances(
        dataset=dataset,
        split="test",
        slice_spec=f"{n_train}:{args.n_instances}",
    )
    
    print(f"Loaded {len(train_instances)} training instances and {len(val_instances)} validation instances")
    
    # Create adapter
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating adapter with model: {args.model}")
    print(f"Environment: {args.environment}")
    print(f"Validation: {'enabled' if args.run_validation else 'disabled'}")
    
    adapter = MiniSWEAgentAdapter(
        model_name=args.model,
        environment_class=args.environment,
        run_validation=args.run_validation,
        validation_max_workers=2,
        timeout=300,  # 5 minutes per instance
        temp_dir=output_dir / "temp",
    )
    
    # Define seed candidate
    seed_candidate = get_default_seed_candidate()
    
    print("\n" + "=" * 80)
    print("SEED CANDIDATE")
    print("=" * 80)
    for component, text in seed_candidate.items():
        print(f"\n{component}:")
        print(text[:200] + "..." if len(text) > 200 else text)
    print("=" * 80 + "\n")
    
    # Setup data loaders
    train_loader = DataLoaderList(train_instances)
    val_loader = DataLoaderList(val_instances)
    
    # Create GEPA engine
    engine = GEPAEngine(
        adapter=adapter,
        train_loader=train_loader,
        val_loader=val_loader,
        seed_candidate=seed_candidate,
        components_to_update=["system_template", "instance_template"],
        log_dir=output_dir / "logs",
    )
    
    print("Starting GEPA optimization...")
    print(f"Iterations: {args.num_iterations}")
    print(f"Components to optimize: system_template, instance_template")
    
    # Run optimization
    result = engine.optimize(
        num_iterations=args.num_iterations,
        train_batch_size=1,  # Process one at a time for SWE-bench
        proposal_batch_size=min(2, len(train_instances)),
    )
    
    # Get best candidate
    best_candidate = result.best_program()
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"Best validation score: {result.best_score():.3f}")
    print(f"Total iterations: {len(result.history)}")
    
    print("\n" + "=" * 80)
    print("BEST CANDIDATE")
    print("=" * 80)
    for component, text in best_candidate.items():
        print(f"\n{component}:")
        print(text)
    print("=" * 80 + "\n")
    
    # Save results
    results_file = output_dir / "best_candidate.txt"
    with open(results_file, "w") as f:
        f.write("BEST CANDIDATE\n")
        f.write("=" * 80 + "\n\n")
        for component, text in best_candidate.items():
            f.write(f"{component}:\n")
            f.write(text)
            f.write("\n\n" + "=" * 80 + "\n\n")
    
    print(f"Results saved to: {output_dir}")
    print(f"Best candidate saved to: {results_file}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Dataset: {dataset}")
    print(f"Train instances: {len(train_instances)}")
    print(f"Val instances: {len(val_instances)}")
    print(f"Iterations: {args.num_iterations}")
    print(f"Best score: {result.best_score():.3f}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

