"""Utilities for configuring LiteLLM HTTP clients with connection pooling."""

from __future__ import annotations

import asyncio
import atexit
from typing import Any

_LOGGING_PATCHED = False
_ATEXIT_REGISTERED = False
_CLIENT_ATTR = "_turbo_gepa_max_connections"


def configure_litellm_client(max_concurrency: int) -> None:
    """Configure shared LiteLLM HTTP clients for sync and async call paths."""

    if max_concurrency <= 0:
        max_concurrency = 1

    try:
        import litellm  # type: ignore
        import httpx  # type: ignore
    except ImportError:
        return

    _disable_litellm_logging(litellm)

    keepalive = max(1, min(20, max_concurrency // 4))
    timeout = httpx.Timeout(180.0, connect=30.0)

    _ensure_async_client(litellm, httpx, max_concurrency, keepalive, timeout)
    _ensure_sync_client(litellm, httpx, max_concurrency, keepalive, timeout)
    _register_atexit(litellm)


def _disable_litellm_logging(litellm: Any) -> None:
    global _LOGGING_PATCHED
    if _LOGGING_PATCHED:
        return

    try:
        litellm.success_callback = []
        litellm.failure_callback = []
        litellm.suppress_debug_info = True
        litellm.set_verbose = False
        setattr(litellm, "logging", False)

        utils = getattr(litellm, "utils", None)
        helper = getattr(utils, "_client_async_logging_helper", None)
        if callable(helper):

            async def _noop_logging_helper(*_args, **_kwargs) -> None:
                return None

            utils._client_async_logging_helper = _noop_logging_helper
    except Exception:
        # Best-effort logging suppression; failures are non-fatal.
        pass

    _LOGGING_PATCHED = True


def _ensure_async_client(litellm: Any, httpx: Any, max_concurrency: int, keepalive: int, timeout: Any) -> None:
    existing = getattr(litellm, "aclient_session", None)
    if existing is not None and not getattr(existing, "is_closed", False):
        if getattr(existing, _CLIENT_ATTR, None) == max_concurrency:
            return
        _close_async_client(existing)

    limits = httpx.Limits(
        max_connections=max_concurrency,
        max_keepalive_connections=keepalive,
        keepalive_expiry=30.0,
    )
    client = httpx.AsyncClient(limits=limits, timeout=timeout)
    setattr(client, _CLIENT_ATTR, max_concurrency)
    litellm.aclient_session = client


def _ensure_sync_client(litellm: Any, httpx: Any, max_concurrency: int, keepalive: int, timeout: Any) -> None:
    existing = getattr(litellm, "client_session", None)
    if existing is not None and not getattr(existing, "is_closed", False):
        if getattr(existing, _CLIENT_ATTR, None) == max_concurrency:
            return
        try:
            existing.close()
        except Exception:
            pass

    limits = httpx.Limits(
        max_connections=max_concurrency,
        max_keepalive_connections=keepalive,
        keepalive_expiry=30.0,
    )
    client = httpx.Client(limits=limits, timeout=timeout)
    setattr(client, _CLIENT_ATTR, max_concurrency)
    litellm.client_session = client


def _register_atexit(litellm: Any) -> None:
    global _ATEXIT_REGISTERED
    if _ATEXIT_REGISTERED:
        return

    def _close_clients() -> None:
        for attr in ("aclient_session", "client_session"):
            client = getattr(litellm, attr, None)
            if client is None:
                continue
            try:
                if hasattr(client, "aclose"):
                    _close_async_client_blocking(client)
                else:
                    client.close()
            except Exception:
                pass

    atexit.register(_close_clients)
    _ATEXIT_REGISTERED = True


def _close_async_client(client: Any) -> None:
    if getattr(client, "is_closed", False):
        return

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        loop.create_task(client.aclose())
    else:
        _close_async_client_blocking(client)


def _close_async_client_blocking(client: Any) -> None:
    if getattr(client, "is_closed", False):
        return

    try:
        asyncio.run(client.aclose())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(client.aclose())
        finally:
            loop.close()

