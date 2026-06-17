"""Small shared helpers for engine implementations."""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Iterable, Mapping
from typing import Any


def warn_unknown_config_keys(engine_name: str, raw: dict[str, Any], known: Iterable[str]) -> None:
    """Warn when ``raw`` contains keys outside ``known`` so typos surface.

    Engines call this from their ``__init__`` after reading the keys they
    consume from ``OptimizeAnythingConfig.engine_config``.
    """
    known_set = frozenset(known)
    unknown = set(raw) - known_set
    if unknown:
        warnings.warn(
            f"{engine_name}: unknown keys in OptimizeAnythingConfig.engine_config: {sorted(unknown)}. "
            f"Known keys: {sorted(known_set)}",
            stacklevel=3,
        )


def example_to_json(eid: str, item: Any) -> dict[str, Any]:
    """Best-effort JSON-friendly dict for a dataset item, with ``id`` injected.

    - Mapping → ``{**item, "id": eid}``
    - dataclass → ``{**asdict(item), "id": eid}``
    - object with ``inputs``/``expected`` attrs → preserve that shape
    - anything else → ``{"id": eid, "data": item}``
    """
    if isinstance(item, Mapping):
        return {**item, "id": eid}
    if dataclasses.is_dataclass(item) and not isinstance(item, type):
        return {**dataclasses.asdict(item), "id": eid}
    inputs = getattr(item, "inputs", None)
    if inputs is not None:
        data: dict[str, Any] = {"id": eid, "inputs": inputs}
        expected = getattr(item, "expected", None)
        if expected is not None:
            data["expected"] = expected
        return data
    return {"id": eid, "data": item}
