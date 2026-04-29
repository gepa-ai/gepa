"""Built-in backends. Importing this package registers each backend by name."""

from gepa.omni.backends.claude_code import ClaudeCodeBackend
from gepa.omni.backends.gepa import GepaBackend
from gepa.omni.backends.meta_harness import MetaHarnessBackend
from gepa.omni.registry import register_backend

register_backend("gepa", GepaBackend)
register_backend("claude_code", ClaudeCodeBackend)
register_backend("meta_harness", MetaHarnessBackend)

__all__ = ["GepaBackend", "ClaudeCodeBackend", "MetaHarnessBackend"]
