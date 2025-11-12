#!/usr/bin/env python3
"""
GEPA training script for optimizing mini-swe-agent on SWE-bench.

This script:
1. Loads the default swebench.yaml config
2. Extracts instance_template as seed candidate (system_template kept constant)
3. Optimizes instance_template using GEPA
4. Evaluates on test set before and after optimization

Usage:
    python train_minisweagent.py --model anthropic/claude-sonnet-4 --n-instances 20
"""

import argparse
import json
import sys
from pathlib import Path

import litellm
import yaml

from gepa import optimize
from gepa.adapters.minisweagent_adapter import MiniSWEAgentAdapter, load_swebench_instances

# Path to the default swebench config from mini-swe-agent
MINI_SWE_AGENT_DIR = Path(__file__).parent.parent.parent.parent.parent / "mini-swe-agent"
DEFAULT_CONFIG_PATH = MINI_SWE_AGENT_DIR / "src" / "minisweagent" / "config" / "extra" / "swebench.yaml"


class Tee:
    """
    A file-like object that duplicates writes to multiple streams.
    Used to simultaneously write to console and log file.
    """
    
    def __init__(self, *streams):
        self.streams = streams
    
    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
    
    def flush(self):
        for stream in self.streams:
            stream.flush()
    
    def isatty(self):
        # Return True if any stream is a tty (for colorized output)
        return any(hasattr(s, 'isatty') and s.isatty() for s in self.streams)


def setup_logging_tee(output_dir: Path):
    """
    Set up tee functionality to duplicate stdout/stderr to out.log.
    
    Args:
        output_dir: Directory where out.log will be created
    
    Returns:
        The log file handle (kept open for the duration of the program)
    """
    log_file = output_dir / "out.log"
    
    # Open log file in write mode with line buffering
    log_handle = open(log_file, "w", buffering=1)
    
    # Save original streams (in case we need them later)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Replace stdout and stderr with Tee objects
    sys.stdout = Tee(original_stdout, log_handle)
    sys.stderr = Tee(original_stderr, log_handle)
    
    print(f"Logging to: {log_file}")
    print(f"All output will be saved to this file.\n")
    
    return log_handle


def load_config(config_path: Path) -> dict:
    """Load agent configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def extract_seed_candidate(config: dict) -> dict[str, str]:
    """
    Extract the seed candidate from the config file.
    
    The seed candidate contains the text components that GEPA will optimize.
    We only optimize instance_template, keeping system_template constant.
    """
    agent_config = config.get("agent", {})
    
    seed_candidate = {
        "instance_template": agent_config.get("instance_template", ""),
    }
    
    return seed_candidate


def main():
    parser = argparse.ArgumentParser(
        description="Optimize mini-swe-agent prompts using GEPA"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-5-mini",
        help="Model to use for agent execution",
    )
    parser.add_argument(
        "--reflection-model",
        type=str,
        default="openai/gpt-5",
        help="Model to use for reflection/optimization",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="princeton-nlp/SWE-bench_Verified",
        choices=["princeton-nlp/SWE-bench_Verified", "princeton-nlp/SWE-bench_Lite"],
        help="SWE-bench dataset to use",
    )
    parser.add_argument(
        "--n-instances",
        type=int,
        default=30,
        help="Total number of instances to use (will be split into train/val/test)",
    )
    parser.add_argument(
        "--environment",
        type=str,
        default="docker",
        choices=["docker", "local"],
        help="Environment type",
    )
    parser.add_argument(
        "--run-validation",
        action="store_true",
        help="Run SWE-bench validation (requires swebench package, slow but accurate)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to base agent config file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("gepa_minisweagent"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=400,
        help="Maximum number of metric calls for optimization",
    )
    parser.add_argument(
        "--reflection-batch-size",
        type=int,
        default=2,
        help="Batch size for reflection",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging tee to capture all output to out.log
    log_handle = setup_logging_tee(args.output_dir)
    
    print("=" * 80)
    print("GEPA Mini-SWE-Agent Optimization")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Reflection Model: {args.reflection_model}")
    print(f"Dataset: {args.dataset}")
    print(f"Total Instances: {args.n_instances}")
    print(f"Environment: {args.environment}")
    print(f"Validation: {'enabled' if args.run_validation else 'disabled'}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 80 + "\n")
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Extract constant system template (not optimized)
    system_template = config.get("agent", {}).get("system_template", "")
    
    # Extract seed candidate (only instance_template will be optimized)
    seed_candidate = extract_seed_candidate(config)
    
    print("\n" + "=" * 80)
    print("SEED CANDIDATE")
    print("=" * 80)
    print("\nSystem Template (constant, not optimized):")
    print(system_template[:500] + "..." if len(system_template) > 500 else system_template)
    print("\nInstance Template (to be optimized):")
    print(seed_candidate["instance_template"][:500] + "..." if len(seed_candidate["instance_template"]) > 500 else seed_candidate["instance_template"])
    print("=" * 80 + "\n")
    
    # Load SWE-bench instances
    print(f"Loading {args.n_instances} instances from {args.dataset}...")
    all_instances = load_swebench_instances(
        dataset=args.dataset,
        split="test",
        slice_spec=f"0:{args.n_instances}",
    )
    
    # Split into train (20%), val (20%), test (60%)
    n_train = int(args.n_instances * 0.2)
    n_val = int(args.n_instances * 0.2)
    
    trainset = all_instances[:n_train]
    valset = all_instances[n_train:n_train + n_val]
    testset = all_instances[n_train + n_val:]
    
    print(f"Train: {len(trainset)} instances")
    print(f"Val: {len(valset)} instances")
    print(f"Test: {len(testset)} instances")
    print()
    
    # Create adapter
    print("Creating adapter...")
    adapter = MiniSWEAgentAdapter(
        model_name=args.model,
        agent_config_path=args.config,
        environment_class=args.environment,
        run_validation=args.run_validation,
        validation_max_workers=10,
        timeout=300,  # 5 minutes per instance
        temp_dir=args.output_dir / "temp",
    )
    
    # Setup reflection LM
    print(f"Setting up reflection model: {args.reflection_model}")
    reflection_lm = (
        lambda prompt: litellm.completion(
            model=args.reflection_model,
            messages=[{"role": "user", "content": prompt}],
        )
        .choices[0]
        .message.content
    )
    
    # Evaluate on test set with empty prompts (baseline)
    print("\n" + "=" * 80)
    print("BASELINE EVALUATION (empty prompts)")
    print("=" * 80)
    empty_candidate = {
        "system_template": system_template,  # Keep system template constant
        "instance_template": "{{task}}",
    }
    
    print("Evaluating on test set with minimal instance_template...")
    testset_results_no_prompt = adapter.evaluate(
        testset,
        empty_candidate,
        capture_traces=True,
    )
    
    baseline_score = sum(testset_results_no_prompt.scores)
    print(f"Baseline score (empty prompts): {baseline_score}/{len(testset)} = {baseline_score/len(testset):.2%}")
    
    # Save baseline results
    baseline_file = args.output_dir / "testset_results_no_prompt.json"
    with open(baseline_file, "w") as f:
        json.dump(
            {
                "score": baseline_score,
                "total": len(testset),
                "percentage": baseline_score / len(testset),
                "scores": testset_results_no_prompt.scores,
                "trajectories": [
                    {
                        "instance_id": traj["data"]["instance_id"],
                        "exit_status": traj["exit_status"],
                        "n_calls": traj["n_calls"],
                        "cost": traj["cost"],
                    }
                    for traj in testset_results_no_prompt.trajectories
                ],
            },
            f,
            indent=2,
        )
    print(f"Saved baseline results to: {baseline_file}")
    
    # Evaluate on test set before optimization
    print("\n" + "=" * 80)
    print("PRE-OPTIMIZATION EVALUATION")
    print("=" * 80)
    print("Evaluating on test set with seed candidate...")
    full_seed_candidate = {
        "system_template": system_template,
        **seed_candidate,
    }
    testset_results_before_opt = adapter.evaluate(
        testset,
        full_seed_candidate,
        capture_traces=True,
    )
    
    pre_opt_score = sum(testset_results_before_opt.scores)
    print(f"Pre-optimization score: {pre_opt_score}/{len(testset)} = {pre_opt_score/len(testset):.2%}")
    
    # Save pre-optimization results
    pre_opt_file = args.output_dir / "testset_results_before_opt.json"
    with open(pre_opt_file, "w") as f:
        json.dump(
            {
                "score": pre_opt_score,
                "total": len(testset),
                "percentage": pre_opt_score / len(testset),
                "scores": testset_results_before_opt.scores,
                "trajectories": [
                    {
                        "instance_id": traj["data"]["instance_id"],
                        "exit_status": traj["exit_status"],
                        "n_calls": traj["n_calls"],
                        "cost": traj["cost"],
                    }
                    for traj in testset_results_before_opt.trajectories
                ],
            },
            f,
            indent=2,
        )
    print(f"Saved pre-optimization results to: {pre_opt_file}")
    
    # Run optimization
    print("\n" + "=" * 80)
    print("OPTIMIZATION")
    print("=" * 80)
    print(f"Max metric calls: {args.max_metric_calls}")
    print(f"Reflection batch size: {args.reflection_batch_size}")
    print(f"Components to optimize: instance_template (system_template kept constant)")
    print()
    
    optimized_results = optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        use_wandb=False,  # Set to True if you have wandb configured
        max_metric_calls=args.max_metric_calls,
        reflection_minibatch_size=args.reflection_batch_size,
        perfect_score=1.0,  # Each instance can score at most 1.0
        skip_perfect_score=False,
        run_dir=str(args.output_dir),
    )
    
    # Get best candidate
    best_candidate = optimized_results.best_candidate
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"Best validation score: {optimized_results.best_score:.3f}")
    print()
    
    # Save optimized candidate
    optimized_candidate_file = args.output_dir / "best_candidate.json"
    with open(optimized_candidate_file, "w") as f:
        json.dump(best_candidate, f, indent=2)
    print(f"Saved best candidate to: {optimized_candidate_file}")
    
    # Save optimized candidate as text
    optimized_candidate_text_file = args.output_dir / "best_candidate.txt"
    with open(optimized_candidate_text_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("OPTIMIZED CANDIDATE\n")
        f.write("=" * 80 + "\n\n")
        for component, text in best_candidate.items():
            f.write(f"{component}:\n")
            f.write("-" * 80 + "\n")
            f.write(text)
            f.write("\n\n" + "=" * 80 + "\n\n")
    print(f"Saved best candidate (text) to: {optimized_candidate_text_file}")
    
    # Evaluate on test set after optimization
    print("\n" + "=" * 80)
    print("POST-OPTIMIZATION EVALUATION")
    print("=" * 80)
    print("Evaluating on test set with optimized candidate...")
    full_best_candidate = {
        "system_template": system_template,
        **best_candidate,
    }
    testset_results_after_opt = adapter.evaluate(
        testset,
        full_best_candidate,
        capture_traces=True,
    )
    
    post_opt_score = sum(testset_results_after_opt.scores)
    print(f"Post-optimization score: {post_opt_score}/{len(testset)} = {post_opt_score/len(testset):.2%}")
    
    # Save post-optimization results
    post_opt_file = args.output_dir / "testset_results_after_opt.json"
    with open(post_opt_file, "w") as f:
        json.dump(
            {
                "score": post_opt_score,
                "total": len(testset),
                "percentage": post_opt_score / len(testset),
                "scores": testset_results_after_opt.scores,
                "trajectories": [
                    {
                        "instance_id": traj["data"]["instance_id"],
                        "exit_status": traj["exit_status"],
                        "n_calls": traj["n_calls"],
                        "cost": traj["cost"],
                    }
                    for traj in testset_results_after_opt.trajectories
                ],
            },
            f,
            indent=2,
        )
    print(f"Saved post-optimization results to: {post_opt_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Total instances: {args.n_instances}")
    print(f"  - Train: {len(trainset)}")
    print(f"  - Val: {len(valset)}")
    print(f"  - Test: {len(testset)}")
    print()
    print(f"Test Set Results:")
    print(f"  - Baseline (empty):  {baseline_score}/{len(testset)} = {baseline_score/len(testset):.2%}")
    print(f"  - Before opt:        {pre_opt_score}/{len(testset)} = {pre_opt_score/len(testset):.2%}")
    print(f"  - After opt:         {post_opt_score}/{len(testset)} = {post_opt_score/len(testset):.2%}")
    print()
    print(f"Improvement:")
    print(f"  - vs Baseline: {post_opt_score - baseline_score:+.1f} ({(post_opt_score - baseline_score)/len(testset):+.1%})")
    print(f"  - vs Pre-opt:  {post_opt_score - pre_opt_score:+.1f} ({(post_opt_score - pre_opt_score)/len(testset):+.1%})")
    print()
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)
    
    # Save summary
    summary_file = args.output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "model": args.model,
                "reflection_model": args.reflection_model,
                "n_instances": args.n_instances,
                "n_train": len(trainset),
                "n_val": len(valset),
                "n_test": len(testset),
                "test_results": {
                    "baseline": {
                        "score": baseline_score,
                        "total": len(testset),
                        "percentage": baseline_score / len(testset),
                    },
                    "before_optimization": {
                        "score": pre_opt_score,
                        "total": len(testset),
                        "percentage": pre_opt_score / len(testset),
                    },
                    "after_optimization": {
                        "score": post_opt_score,
                        "total": len(testset),
                        "percentage": post_opt_score / len(testset),
                    },
                },
                "improvement": {
                    "vs_baseline": post_opt_score - baseline_score,
                    "vs_baseline_percentage": (post_opt_score - baseline_score) / len(testset),
                    "vs_pre_opt": post_opt_score - pre_opt_score,
                    "vs_pre_opt_percentage": (post_opt_score - pre_opt_score) / len(testset),
                },
            },
            f,
            indent=2,
        )
    print(f"\nSaved summary to: {summary_file}")


if __name__ == "__main__":
    main()


