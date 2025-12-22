"""
Configuration and argument parsing for ARC-AGI GEPA optimization.
"""

import argparse
import os


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GEPA ARC-AGI Solver Optimization")

    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=100,
        help="Maximum number of metric calls (evaluations)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


def get_log_directory():
    """Get fixed log directory."""
    log_dir = "results/arc_agi"
    os.makedirs(log_dir, exist_ok=True)
    return log_dir
