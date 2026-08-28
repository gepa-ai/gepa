"""Entry point for the `gepa-rpc` console script."""

from __future__ import annotations

import argparse
import logging
import sys

from gepa.rpc.server import serve
from gepa.rpc.servicer import DEFAULT_RUNS_DIR


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gepa-rpc", description="Launch the GEPA gRPC server.")
    parser.add_argument("--port", type=int, default=50051,
                        help="TCP port to listen on (1-65535).")
    parser.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR, help="Directory for per-run checkpoints.")
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    if not (1 <= args.port <= 65535):
        parser.error(f"--port must be between 1 and 65535, got {args.port}")

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    serve(port=args.port, runs_dir=args.runs_dir, max_workers=args.max_workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
