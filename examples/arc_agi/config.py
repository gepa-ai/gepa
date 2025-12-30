"""
Configuration and argument parsing for ARC-AGI GEPA optimization.
"""

import argparse
from pathlib import Path
from typing import Dict, Any


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GEPA ARC-AGI Solver Optimization")

    # GEPA parameters
    parser.add_argument(
        "--max_metric_calls",
        type=int,
        default=100,
        help="Maximum number of GEPA iterations (default: 100)",
    )

    parser.add_argument(
        "--reflection_minibatch_size",
        type=int,
        default=3,
        help="Number of evaluations before reflection (default: 3)",
    )

    # LLM parameters
    parser.add_argument(
        "--llm_model",
        type=str,
        default="openai/gpt-4.1-nano",
        help="LLM model for reflection (default: openai/gpt-5)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in outputs/artifacts/arc_agi/",
    )

    return parser.parse_args()


def get_experiment_config(args) -> Dict[str, Any]:
    """Convert args to experiment config dict."""
    return {
        "max_metric_calls": args.max_metric_calls,
        "reflection_minibatch_size": args.reflection_minibatch_size,
        "llm_model": args.llm_model,
        "seed": args.seed,
    }


def get_latest_log_directory() -> Path:
    """Find the latest log directory in outputs/artifacts/arc_agi/."""
    base_dir = Path("outputs/artifacts/arc_agi")

    if not base_dir.exists():
        raise ValueError(f"Base directory {base_dir} does not exist")

    # Get all subdirectories
    log_dirs = [d for d in base_dir.iterdir() if d.is_dir()]

    if not log_dirs:
        raise ValueError(f"No log directories found in {base_dir}")

    # Sort by modification time (most recent first)
    log_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    return log_dirs[0]


def get_log_directory(resume: bool = False) -> Path:
    """Get or create log directory with timestamp."""
    if resume:
        return get_latest_log_directory()

    import time

    timestamp = time.strftime("%y%m%d_%H:%M:%S")
    log_dir = f"outputs/artifacts/arc_agi/{timestamp}"

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    return log_path
