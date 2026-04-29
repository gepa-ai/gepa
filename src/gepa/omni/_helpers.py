"""Small shared helpers for backend implementations."""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Any


def warn_unknown_config_keys(backend_name: str, raw: dict[str, Any], known: Iterable[str]) -> None:
    """Warn when ``raw`` contains keys outside ``known`` so typos surface.

    Backends call this from their ``__init__`` after reading the keys they
    consume from ``OmniConfig.config``.
    """
    known_set = frozenset(known)
    unknown = set(raw) - known_set
    if unknown:
        warnings.warn(
            f"{backend_name}: unknown keys in OmniConfig.config: {sorted(unknown)}. Known keys: {sorted(known_set)}",
            stacklevel=3,
        )
