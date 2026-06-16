"""Custom GEPA candidate proposers exposed by the optimize_anything package.

Currently ships :class:`ClaudeCodeAgentProposer` — a file-based proposer that
launches ``claude --print`` once per GEPA reflection step, with a sandboxed
view of the run directory. Plugged into GEPA via
``ReflectionConfig.custom_candidate_proposer`` (the
:class:`gepa.core.adapter.ProposalFn` slot), so it slots straight into the
:class:`gepa.oa.engines.gepa.GepaEngine` config under the
``claude_code_agent`` key.
"""

from gepa.oa.proposers.claude_code_agent import ClaudeCodeAgentProposer

__all__ = ["ClaudeCodeAgentProposer"]
