"""Debug output controls for the Glean integration.

Library callers can set ``GEPA_DEBUG=1`` or call :func:`set_debug`; the Glean
CLI exposes the same behavior through ``--debug``.
"""

from __future__ import annotations

import os
from typing import Any


def _env_flag(value: str | None) -> bool:
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


_debug_enabled = _env_flag(os.environ.get("GEPA_DEBUG"))


def set_debug(enabled: bool) -> None:
    """Enable or disable diagnostic output for the current process."""
    global _debug_enabled
    _debug_enabled = enabled


def debug_print(*args: Any, **kwargs: Any) -> None:
    """Print diagnostic output only when debugging is enabled."""
    if _debug_enabled:
        print(*args, **kwargs)


__all__ = ["debug_print", "set_debug"]
