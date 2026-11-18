"""Entry point for the `gepa-rpc` console script."""

from __future__ import annotations

import argparse
import logging
import sys

from gepa.rpc.adapter import DEFAULT_CALLBACK_TIMEOUT_SECONDS
from gepa.rpc.server import DEFAULT_HOST, DEFAULT_MAX_WORKERS, serve
from gepa.rpc.servicer import (
    DEFAULT_HANDSHAKE_TIMEOUT_SECONDS,
    DEFAULT_MAX_CONCURRENT_RUNS,
    DEFAULT_REFLECTION_LM,
    DEFAULT_RUNS_DIR,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gepa-rpc", description="Launch the GEPA gRPC server.")
    parser.add_argument("--port", type=int, default=50051,
                        help="TCP port to listen on (1-65535).")
    parser.add_argument(
        "--host", default=DEFAULT_HOST,
        help="Interface to bind. Defaults to loopback only; pass 0.0.0.0 (or a "
             "specific interface) to accept connections from beyond this host -- "
             "consider --tls-cert/--tls-key and/or --auth-token when doing so.",
    )
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
    parser.add_argument(
        "--handshake-timeout", type=float, default=DEFAULT_HANDSHAKE_TIMEOUT_SECONDS,
        help="Seconds a client may hold an open stream before sending its "
             "start_request. Bounds the run slot a silent connection can "
             "occupy.",
    )
    parser.add_argument("--tls-cert", default=None, help="Path to a PEM certificate chain. Requires --tls-key.")
    parser.add_argument("--tls-key", default=None, help="Path to a PEM private key. Requires --tls-cert.")
    parser.add_argument(
        "--auth-token", default=None,
        help="Shared secret clients must send as the 'authorization' metadata value "
             "on every RPC. Not a substitute for TLS: without --tls-cert/--tls-key "
             "the token itself travels in plaintext.",
    )
    parser.add_argument(
        "--allowed-engines", default=None,
        help="Comma-separated allowlist of Optimize Anything engines clients may "
             "select (e.g. 'gepa,best_of_n'). Default: any. Restrict this on any "
             "server reachable by untrusted clients -- autoresearch and "
             "meta_harness spawn a Claude Code agent subprocess.",
    )
    parser.add_argument(
        "--allowed-reflection-lms", default=None,
        help="Comma-separated allowlist of reflection_lm values clients may select. "
             "Default: any. Restrict this on any server reachable by untrusted "
             "clients -- an arbitrary reflection_lm spends the operator's LLM "
             "budget on an arbitrary model.",
    )
    parser.add_argument(
        "--default-reflection-lm", default=DEFAULT_REFLECTION_LM,
        help="reflection_lm used when a client doesn't specify one.",
    )
    parser.add_argument(
        "--allow-insecure-remote", action="store_true",
        help="Acknowledge listening on a non-loopback address with no "
             "--auth-token and no TLS. Required in that configuration, since "
             "reaching this port means being able to spend this server's LLM "
             "budget and run an agent subprocess on this host. Only for "
             "genuinely trusted networks.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    if not (1 <= args.port <= 65535):
        parser.error(f"--port must be between 1 and 65535, got {args.port}")
    if bool(args.tls_cert) != bool(args.tls_key):
        parser.error("--tls-cert and --tls-key must be provided together")

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    serve(
        port=args.port,
        host=args.host,
        runs_dir=args.runs_dir,
        max_workers=args.max_workers,
        max_concurrent_runs=args.max_concurrent_runs,
        callback_timeout=args.callback_timeout,
        handshake_timeout=args.handshake_timeout,
        tls_cert_path=args.tls_cert,
        tls_key_path=args.tls_key,
        auth_token=args.auth_token,
        allowed_engines=args.allowed_engines.split(",") if args.allowed_engines else None,
        allowed_reflection_lms=(
            args.allowed_reflection_lms.split(",") if args.allowed_reflection_lms else None
        ),
        default_reflection_lm=args.default_reflection_lm,
        allow_insecure_remote=args.allow_insecure_remote,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
