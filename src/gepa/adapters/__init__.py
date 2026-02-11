"""Package-level helpers for the GEPA adapter collection."""

from __future__ import annotations

from .opik_adapter.opik_adapter import OpikAdapter, OpikDataInst  # noqa: F401

__all__ = ["OpikAdapter", "OpikDataInst"]
