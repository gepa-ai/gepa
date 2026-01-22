"""Configuration and setup utilities for gepa_blog blackbox optimization."""

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
        default="openai/gpt-5",
        help="LLM model to use (default: openai/gpt-5)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="gepa_blog",
        help="Optional suffix for log and wandb run names",
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
        "--evaluation-budget",
        type=int,
        default=100,
        help="Evaluation budget per candidate (max number of evaluation calls per code candidate)",
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
        model_name = args.llm_model.replace("openai/", "").replace("/", "_")
        log_dir = f"outputs/artifacts/polynomial/{args.run_name}/problem_{args.problem_index}/{model_name}/{args.seed}/{timestamp}"

    os.makedirs(log_dir, exist_ok=True)
    return log_dir
