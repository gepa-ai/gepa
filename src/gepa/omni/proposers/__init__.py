"""Custom GEPA candidate proposers exposed by the omni package.

Currently ships :class:`ClaudeCodeAgentProposer` — a file-based proposer that
launches ``claude --print`` once per GEPA reflection step, with a sandboxed
view of the run directory. Plugged into GEPA via
``ReflectionConfig.custom_candidate_proposer`` (the
:class:`gepa.core.adapter.ProposalFn` slot), so it slots straight into the
:class:`gepa.omni.backends.gepa.GepaBackend` config under the
``claude_code_agent`` key.
"""

from gepa.omni.proposers.claude_code_agent import ClaudeCodeAgentProposer

__all__ = ["ClaudeCodeAgentProposer"]
