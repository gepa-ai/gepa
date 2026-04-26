"""Configuration for circle packing optimization."""

import argparse
import os
import time


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Circle packing optimization"
    )
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=100,
        help="Maximum number of metric calls (default: 100)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Log directory (if not provided, will be auto-generated)",
    )
    return parser.parse_args()


def get_log_directory(args):
    """Get or create log directory."""
    if args.log_dir:
        log_dir = args.log_dir
    else:
        timestamp = time.strftime("%y%m%d_%H:%M:%S")
        log_dir = f"logs/circle_packing/{timestamp}_n26"

    os.makedirs(log_dir, exist_ok=True)
    return log_dir
