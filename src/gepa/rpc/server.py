"""Thin wrapper around grpc.server for embedding GEPAService in a process."""

from __future__ import annotations

import logging
import signal
from collections.abc import Collection
from concurrent import futures

import grpc

from gepa.rpc.adapter import DEFAULT_CALLBACK_TIMEOUT_SECONDS
from gepa.rpc.generated import gepa_pb2_grpc as pb_grpc
from gepa.rpc.servicer import DEFAULT_MAX_CONCURRENT_RUNS, DEFAULT_RUNS_DIR, GEPAServicer

logger = logging.getLogger(__name__)

_GRACEFUL_SHUTDOWN_SECONDS = 30

# Bind to loopback by default. A server reachable only from the same host
# can't be driven by an arbitrary network client to spend the operator's LLM
# budget or spawn autoresearch/meta_harness agent subprocesses. Pass
# --host 0.0.0.0 (or a specific interface) to opt into listening beyond this
# host, ideally alongside --tls-cert/--tls-key and/or --auth-token.
DEFAULT_HOST = "127.0.0.1"

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


class _AuthTokenInterceptor(grpc.ServerInterceptor):
    """Rejects any RPC whose `authorization` metadata doesn't match the
    configured shared secret.

    A minimal shared-token check, not a replacement for TLS: it stops an
    unauthenticated caller from invoking anything, but the token itself
    travels in plaintext unless --tls-cert/--tls-key is also set. Intended
    to close "anyone who can reach the port can start runs that spend the
    operator's LLM budget", not to be a full auth system.
    """

    def __init__(self, token: str):
        self._token = token

    def intercept_service(self, continuation, handler_call_details):
        metadata = dict(handler_call_details.invocation_metadata or ())
        if metadata.get("authorization") == self._token:
            return continuation(handler_call_details)

        # Resolve the real handler only to learn its streaming shape -- this
        # does NOT invoke it -- so the rejection handler we substitute
        # matches the RPC's actual unary/streaming cardinality. Returning the
        # wrong shape gives the client a confusing deserialization error
        # instead of a clean UNAUTHENTICATED.
        handler = continuation(handler_call_details)
        if handler is None:
            return None

        def abort(_request_or_iterator, context):
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "missing or invalid auth token")

        if handler.request_streaming and handler.response_streaming:
            return grpc.stream_stream_rpc_method_handler(abort)
        if handler.request_streaming:
            return grpc.stream_unary_rpc_method_handler(abort)
        if handler.response_streaming:
            return grpc.unary_stream_rpc_method_handler(abort)
        return grpc.unary_unary_rpc_method_handler(abort)


def build_server(
    port: int,
    host: str = DEFAULT_HOST,
    runs_dir: str = DEFAULT_RUNS_DIR,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_concurrent_runs: int = DEFAULT_MAX_CONCURRENT_RUNS,
    callback_timeout: float = DEFAULT_CALLBACK_TIMEOUT_SECONDS,
    tls_cert_path: str | None = None,
    tls_key_path: str | None = None,
    auth_token: str | None = None,
    allowed_engines: Collection[str] | None = None,
    allowed_reflection_lms: Collection[str] | None = None,
    default_reflection_lm: str | None = None,
) -> grpc.Server:
    if bool(tls_cert_path) != bool(tls_key_path):
        raise ValueError("tls_cert_path and tls_key_path must be provided together")
    if max_workers < max_concurrent_runs + CONTROL_PLANE_RESERVE:
        raise ValueError(
            f"max_workers ({max_workers}) must be at least max_concurrent_runs "
            f"({max_concurrent_runs}) + {CONTROL_PLANE_RESERVE} (reserved so "
            "GetStatus/Cancel always have a free pool thread even when every "
            "run slot is saturated)"
        )
    interceptors: list[grpc.ServerInterceptor] = (
        [_AuthTokenInterceptor(auth_token)] if auth_token else []
    )
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=_KEEPALIVE_OPTIONS,
        interceptors=interceptors,
    )
    pb_grpc.add_GEPAServiceServicer_to_server(
        GEPAServicer(
            runs_dir=runs_dir,
            max_concurrent_runs=max_concurrent_runs,
            callback_timeout=callback_timeout,
            allowed_engines=allowed_engines,
            allowed_reflection_lms=allowed_reflection_lms,
            default_reflection_lm=default_reflection_lm,
        ),
        server,
    )

    address = f"{host}:{port}"
    if tls_cert_path and tls_key_path:
        with open(tls_key_path, "rb") as f:
            private_key = f.read()
        with open(tls_cert_path, "rb") as f:
            certificate_chain = f.read()
        credentials = grpc.ssl_server_credentials([(private_key, certificate_chain)])
        bound = server.add_secure_port(address, credentials)
    else:
        bound = server.add_insecure_port(address)
    if bound == 0:
        raise RuntimeError(f"failed to bind gRPC server to {address} (port may already be in use)")
    return server


def serve(
    port: int,
    host: str = DEFAULT_HOST,
    runs_dir: str = DEFAULT_RUNS_DIR,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_concurrent_runs: int = DEFAULT_MAX_CONCURRENT_RUNS,
    callback_timeout: float = DEFAULT_CALLBACK_TIMEOUT_SECONDS,
    tls_cert_path: str | None = None,
    tls_key_path: str | None = None,
    auth_token: str | None = None,
    allowed_engines: Collection[str] | None = None,
    allowed_reflection_lms: Collection[str] | None = None,
    default_reflection_lm: str | None = None,
) -> None:
    server = build_server(
        port=port,
        host=host,
        runs_dir=runs_dir,
        max_workers=max_workers,
        max_concurrent_runs=max_concurrent_runs,
        callback_timeout=callback_timeout,
        tls_cert_path=tls_cert_path,
        tls_key_path=tls_key_path,
        auth_token=auth_token,
        allowed_engines=allowed_engines,
        allowed_reflection_lms=allowed_reflection_lms,
        default_reflection_lm=default_reflection_lm,
    )

    def _shutdown(signum, frame):
        logger.info("received signal %d, stopping server (grace=%ds)", signum, _GRACEFUL_SHUTDOWN_SECONDS)
        server.stop(grace=_GRACEFUL_SHUTDOWN_SECONDS)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    server.start()
    logger.info("gepa-rpc listening on %s:%d (runs_dir=%s)", host, port, runs_dir)
    server.wait_for_termination()
