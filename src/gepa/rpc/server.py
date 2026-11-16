"""Thin wrapper around grpc.server for embedding GEPAService in a process."""

from __future__ import annotations

import logging
import signal
from concurrent import futures

import grpc

from gepa.rpc.adapter import DEFAULT_CALLBACK_TIMEOUT_SECONDS
from gepa.rpc.generated import gepa_pb2_grpc as pb_grpc
from gepa.rpc.servicer import DEFAULT_MAX_CONCURRENT_RUNS, DEFAULT_RUNS_DIR, GEPAServicer

logger = logging.getLogger(__name__)

_GRACEFUL_SHUTDOWN_SECONDS = 30

# gRPC pool threads reserved *above* max_concurrent_runs. Each in-flight
# RunOptimization/RunOptimizationOmni stream pins one pool thread for its
# whole life, so max_workers must leave headroom for GetStatus/Cancel -- and
# for the cheap rejection path of the next run attempt -- to always find a
# free thread even when every permitted run slot is busy.
CONTROL_PLANE_RESERVE = 4
DEFAULT_MAX_WORKERS = DEFAULT_MAX_CONCURRENT_RUNS + CONTROL_PLANE_RESERVE + 20


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
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_concurrent_runs: int = DEFAULT_MAX_CONCURRENT_RUNS,
    callback_timeout: float = DEFAULT_CALLBACK_TIMEOUT_SECONDS,
) -> grpc.Server:
    if max_workers < max_concurrent_runs + CONTROL_PLANE_RESERVE:
        raise ValueError(
            f"max_workers ({max_workers}) must be at least max_concurrent_runs "
            f"({max_concurrent_runs}) + {CONTROL_PLANE_RESERVE} (reserved so "
            "GetStatus/Cancel always have a free pool thread even when every "
            "run slot is saturated)"
        )
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=_KEEPALIVE_OPTIONS,
    )
    pb_grpc.add_GEPAServiceServicer_to_server(
        GEPAServicer(
            runs_dir=runs_dir,
            max_concurrent_runs=max_concurrent_runs,
            callback_timeout=callback_timeout,
        ),
        server,
    )
    bound = server.add_insecure_port(f"[::]:{port}")
    if bound == 0:
        raise RuntimeError(f"failed to bind gRPC server to port {port} (port may already be in use)")
    return server


def serve(
    port: int,
    runs_dir: str = DEFAULT_RUNS_DIR,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_concurrent_runs: int = DEFAULT_MAX_CONCURRENT_RUNS,
    callback_timeout: float = DEFAULT_CALLBACK_TIMEOUT_SECONDS,
) -> None:
    server = build_server(
        port=port,
        runs_dir=runs_dir,
        max_workers=max_workers,
        max_concurrent_runs=max_concurrent_runs,
        callback_timeout=callback_timeout,
    )

    def _shutdown(signum, frame):
        logger.info("received signal %d, stopping server (grace=%ds)", signum, _GRACEFUL_SHUTDOWN_SECONDS)
        server.stop(grace=_GRACEFUL_SHUTDOWN_SECONDS)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    server.start()
    logger.info("gepa-rpc listening on :%d (runs_dir=%s)", port, runs_dir)
    server.wait_for_termination()
