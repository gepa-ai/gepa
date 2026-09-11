"""Thin wrapper around grpc.server for embedding GEPAService in a process."""

from __future__ import annotations

import logging
import signal
from concurrent import futures

import grpc

from gepa.rpc.generated import gepa_pb2_grpc as pb_grpc
from gepa.rpc.servicer import DEFAULT_RUNS_DIR, GEPAServicer

logger = logging.getLogger(__name__)

_GRACEFUL_SHUTDOWN_SECONDS = 30


_KEEPALIVE_OPTIONS = [
    # Send a ping every 60 s of idle to detect silently dropped TCP connections.
    ("grpc.keepalive_time_ms", 60_000),
    # Close the connection if the peer doesn't respond within 10 s.
    ("grpc.keepalive_timeout_ms", 10_000),
    # Allow keepalive pings even when there are no active RPCs.
    ("grpc.keepalive_permit_without_calls", True),
]


def build_server(
    port: int,
    runs_dir: str = DEFAULT_RUNS_DIR,
    max_workers: int = 16,
) -> grpc.Server:
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=_KEEPALIVE_OPTIONS,
    )
    pb_grpc.add_GEPAServiceServicer_to_server(GEPAServicer(runs_dir=runs_dir), server)
    bound = server.add_insecure_port(f"[::]:{port}")
    if bound == 0:
        raise RuntimeError(f"failed to bind gRPC server to port {port} (port may already be in use)")
    return server


def serve(port: int, runs_dir: str = DEFAULT_RUNS_DIR, max_workers: int = 16) -> None:
    server = build_server(port=port, runs_dir=runs_dir, max_workers=max_workers)

    def _shutdown(signum, frame):
        logger.info("received signal %d, stopping server (grace=%ds)", signum, _GRACEFUL_SHUTDOWN_SECONDS)
        server.stop(grace=_GRACEFUL_SHUTDOWN_SECONDS)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    server.start()
    logger.info("gepa-rpc listening on :%d (runs_dir=%s)", port, runs_dir)
    server.wait_for_termination()
