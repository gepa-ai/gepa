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
    return parser.parse_args()


def get_log_directory(args):
    """Get or create log directory."""
    if args.log_dir:
        log_dir = args.log_dir
    else:
        timestamp = time.strftime("%y%m%d_%H:%M:%S")
        run_id = f"{timestamp}"
        if args.run_name:
            run_id += f"_{args.run_name}"
        log_dir = f"logs/polynomial/{run_id}"

    os.makedirs(log_dir, exist_ok=True)
    return log_dir
