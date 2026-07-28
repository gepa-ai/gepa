"""Agent-harness registry for optimize_anything.

``OptimizeAnythingConfig.harness`` selects the runtime that executes agent
sessions for the subprocess engines:

- ``"claude-code"`` (default) — the existing ``claude --print`` CLI path.
- ``"omnigent"`` — any Omnigent-wrapped backend; pass a dict for options,
  e.g. ``{"type": "omnigent", "backend": "codex", "sandbox_type": "bwrap"}``.
- an :class:`~gepa.oa.harness.base.AgentHarness` instance — used as-is.
"""

from __future__ import annotations

from typing import Any

from gepa.oa.harness.base import AgentHarness, AgentRunResult, AgentRunSpec

__all__ = ["AgentHarness", "AgentRunResult", "AgentRunSpec", "get_harness"]


def get_harness(
    spec: str | dict[str, Any] | AgentHarness,
    *,
    sandbox: bool = True,
) -> AgentHarness:
    """Build the harness named by ``spec``.

    Args:
        spec: Harness name, an options dict with a ``"type"`` key, or a
            ready :class:`AgentHarness` instance (returned unchanged).
        sandbox: The engine-level ``OptimizeAnythingConfig.sandbox`` value;
            forwarded as each harness's default unless the options dict
            overrides it.
    """
    if isinstance(spec, AgentHarness):
        return spec
    if isinstance(spec, str):
        kind, options = spec, {}
    else:
        options = dict(spec)
        kind = options.pop("type", "claude-code")
    options.setdefault("sandbox", sandbox)

    if kind in ("claude-code", "claude"):
        from gepa.oa.harness.claude_code import ClaudeCodeHarness

        return ClaudeCodeHarness(sandbox=bool(options.get("sandbox", True)))
    if kind == "omnigent":
        from gepa.oa.harness.omnigent import OmnigentHarness

        return OmnigentHarness(**options)
    raise ValueError(f"Unknown harness {kind!r}: expected 'claude-code', 'omnigent', or an AgentHarness instance")
