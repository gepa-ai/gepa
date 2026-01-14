"""Configuration and setup utilities for circle packing optimization."""

import argparse
import os
import time


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OptimizeAnything Blackbox Function Optimization"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="openai/gpt-5-nano",
        help="LLM model to use (default: openai/gpt-5-nano)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional suffix for log and wandb run names",
    )
    parser.add_argument(
        "--optimization-level",
        type=str,
        default="light",
        choices=["light", "medium", "heavy"],
        help="GEPA optimization level: light (fast), medium, heavy (thorough) (default: light)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Log directory (if not provided, will be auto-generated)",
    )
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=None,
        help="Maximum number of metric calls (overrides optimization level)",
    )
    parser.add_argument(
        "--num-proposals",
        type=int,
        default=10,
        help="Max number of proposals to generate through the search",
    )
    parser.add_argument(
        "--evaluation-budget",
        type=int,
        default=100,
        help="Evaluation budget per candidate (max number of evaluation calls per code candidate)",
    )
    parser.add_argument(
        "--problem-name",
        type=str,
        default=None,
        help="Name of the problem to optimize (deprecated, use --problem-index instead).",
    )
    parser.add_argument(
        "--problem-index",
        type=int,
        default=None,
        help="Index of the problem in experiments/polynomial/problems.py (0-55).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds for code execution (default: 300 = 5 minutes)",
    )
    return parser.parse_args()


def get_log_directory(args):
    """Get or create log directory."""
    if args.log_dir:
        log_dir = args.log_dir
    else:
        timestamp = time.strftime("%y%m%d_%H:%M:%S")
        # Sanitize model name (remove "openai/" prefix, replace "/" with "_")
        model_name = args.llm_model.replace("openai/", "").replace("/", "_")

        if args.problem_name:
            # Structure: experiments/blog_logs/bbox_opt/gepa/{problem_name}/{model_name}/{seed}/{timestamp}/
            log_dir = f"experiments/blog_logs/bbox_opt/gepa/{args.problem_name}/{model_name}/{args.seed}/{timestamp}"
        else:
            # Fallback for when problem_name is not provided
            run_id = f"{timestamp}"
            if args.run_name:
                run_id += f"_{args.run_name}"
            log_dir = f"experiments/blog_logs/logs/gepa/{run_id}"

    os.makedirs(log_dir, exist_ok=True)
    return log_dir
