"""Built-in backends. Importing this package registers each backend by name."""

from gepa.omni.backends.best_of_n import BestOfNBackend
from gepa.omni.backends.claude_code import ClaudeCodeBackend
from gepa.omni.backends.gepa import GepaBackend
from gepa.omni.backends.meta_harness import MetaHarnessBackend
from gepa.omni.registry import register_backend

register_backend("gepa", GepaBackend)
register_backend("claude_code", ClaudeCodeBackend)
register_backend("meta_harness", MetaHarnessBackend)
register_backend("best_of_n", BestOfNBackend)

__all__ = ["BestOfNBackend", "ClaudeCodeBackend", "GepaBackend", "MetaHarnessBackend"]
