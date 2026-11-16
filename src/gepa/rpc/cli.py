"""Entry point for the `gepa-rpc` console script."""

from __future__ import annotations

import argparse
import logging
import sys

from gepa.rpc.adapter import DEFAULT_CALLBACK_TIMEOUT_SECONDS
from gepa.rpc.server import DEFAULT_MAX_WORKERS, serve
from gepa.rpc.servicer import DEFAULT_MAX_CONCURRENT_RUNS, DEFAULT_RUNS_DIR


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gepa-rpc", description="Launch the GEPA gRPC server.")
    parser.add_argument("--port", type=int, default=50051,
                        help="TCP port to listen on (1-65535).")
    parser.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR, help="Directory for per-run checkpoints.")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument(
        "--max-concurrent-runs", type=int, default=DEFAULT_MAX_CONCURRENT_RUNS,
        help="Cap on simultaneous RunOptimization/RunOptimizationOmni streams. "
             "--max-workers must leave enough headroom above this for "
             "GetStatus/Cancel to stay responsive while every run slot is busy.",
    )
    parser.add_argument(
        "--callback-timeout", type=float, default=DEFAULT_CALLBACK_TIMEOUT_SECONDS,
        help="Seconds to wait for a client to answer an evaluate or "
             "reflective-dataset request before failing the run.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    if not (1 <= args.port <= 65535):
        parser.error(f"--port must be between 1 and 65535, got {args.port}")

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    serve(
        port=args.port,
        runs_dir=args.runs_dir,
        max_workers=args.max_workers,
        max_concurrent_runs=args.max_concurrent_runs,
        callback_timeout=args.callback_timeout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
