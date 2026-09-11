"""Custom GEPA candidate proposers exposed by the optimize_anything package.

Ships:

* :class:`ClaudeCodeAgentProposer` — a file-based proposer that launches
  ``claude --print`` once per GEPA reflection step, with a sandboxed view of the
  run directory.
* :class:`GitAgentProposer` — an agent-agnostic proposer for *git-commit* mode
  (candidate = a git commit SHA). It leases a worktree at the parent SHA, runs a
  caller-supplied coding agent to edit files, and mints the child commit.

Both plug into GEPA via ``ReflectionConfig.custom_candidate_proposer`` (the
:class:`gepa.core.adapter.ProposalFn` slot), so they slot straight into the
:class:`gepa.oa.engines.gepa.GepaEngine` config.
"""

from gepa.oa.proposers.claude_code_agent import ClaudeCodeAgentProposer
from gepa.oa.proposers.git_agent import GitAgentProposer, ProposalContext

__all__ = ["ClaudeCodeAgentProposer", "GitAgentProposer", "ProposalContext"]
