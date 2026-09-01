"""Agent-agnostic git-commit candidate proposer.

In git-commit mode a GEPA candidate is a git commit SHA (see
:mod:`gepa.oa.repo_pool`). :class:`GitAgentProposer` is the
:class:`~gepa.core.adapter.ProposalFn` that turns a coding agent's edits into
the next candidate commit. Plug it into GEPA via
``ReflectionConfig.custom_candidate_proposer`` (the same slot
:class:`~gepa.oa.proposers.claude_code_agent.ClaudeCodeAgentProposer` uses).

Each proposal:

1. reads the parent commit SHA from the current candidate (the git-mode
   component — a single component whose value is a SHA string),
2. leases that SHA's worktree slot from the pool with ``exclusive=True`` so the
   agent owns the working tree for the whole edit,
3. runs ``agent(slot_dir, context)`` — a caller-supplied callable that edits
   files *in place* within ``manifest_globs``,
4. mints the child commit via ``pool.commit_worktree(...)`` (which stages the
   manifest, commits with repo hooks disabled, and rejects any out-of-manifest
   path / symlink / gitlink), and
5. returns ``{component: child_sha}``.

The *agent* is any callable ``(slot_dir: Path, ctx: ProposalContext) -> None``.
This class has no LLM dependency: bind your own coding CLI (Claude Code, Aider,
a shell script, …) or, in tests, pass a scripted mutation. When the agent makes
no in-manifest change (a no-op proposal) the pool returns the parent SHA and
GEPA's acceptance test simply rejects the unchanged candidate. When an edit
violates the manifest allowlist, the proposal is treated as a rejected no-op
(returns the parent) rather than aborting the run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from gepa.oa.repo_pool import RepoCandidatePool

# GEPA's internal component key for a single-string (non-dict) seed candidate.
_STR_CANDIDATE_KEY = "current_candidate"


@dataclass
class ProposalContext:
    """Everything an agent needs to decide its edits for one proposal.

    Passed to the ``agent`` callable alongside the leased working directory. The
    agent reads whatever it needs and edits files in ``slot_dir`` in place.
    """

    slot_dir: Path
    parent_sha: str
    component: str
    manifest_globs: Sequence[str]
    reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]] = field(default_factory=dict)
    components_to_update: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


# An agent edits the working tree in place; it returns nothing.
AgentFn = Callable[[Path, ProposalContext], None]


class GitAgentProposer:
    """Turn a coding agent's edits into the next git-commit candidate.

    Args:
        agent: Callable ``(slot_dir, ctx) -> None`` that edits files in place
            inside the leased worktree. Any coding tool or scripted mutation.
        pool: A :class:`~gepa.oa.repo_pool.RepoCandidatePool` (the portable
            :class:`~gepa.oa.repo_pool.GitWorktreePool`, or a custom one).
        manifest_globs: The editable surface. Only these paths are staged into
            the candidate commit, and the pool rejects edits outside them.
        component: Candidate-dict key holding the repo SHA. Defaults to GEPA's
            single-string candidate key ``"current_candidate"``; ignored when the
            candidate has exactly one component (that sole key is used).
        commit_message: Commit message, or a callable ``(ctx) -> str``.
    """

    def __init__(
        self,
        agent: AgentFn,
        pool: RepoCandidatePool,
        manifest_globs: Sequence[str],
        *,
        component: str = _STR_CANDIDATE_KEY,
        commit_message: str | Callable[[ProposalContext], str] = "gepa candidate",
    ) -> None:
        if not manifest_globs:
            raise ValueError("GitAgentProposer requires a non-empty manifest_globs")
        self.agent = agent
        self.pool = pool
        self.manifest_globs = list(manifest_globs)
        self.component = component
        self._commit_message = commit_message
        # Cost source hook read by GepaEngine: mirrors the agent's spend when the
        # agent exposes ``total_cost``; stays 0.0 for scripted/free agents.
        self.total_cost: float = 0.0

    def _pick_component(self, candidate: Mapping[str, str], components_to_update: Sequence[str]) -> str:
        """Resolve which candidate component carries the repo SHA to evolve."""
        if len(candidate) == 1:
            return next(iter(candidate))
        for name in components_to_update:
            if name in candidate:
                return name
        if self.component in candidate:
            return self.component
        raise KeyError(
            f"GitAgentProposer cannot locate the git-mode component; candidate keys "
            f"{list(candidate)}, components_to_update {list(components_to_update)}, "
            f"configured component {self.component!r}"
        )

    def _message(self, ctx: ProposalContext) -> str:
        msg = self._commit_message
        return msg(ctx) if callable(msg) else msg

    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, str]:
        component = self._pick_component(candidate, components_to_update)
        parent_sha = candidate[component]
        lease = self.pool.lease(parent_sha, exclusive=True)
        try:
            ctx = ProposalContext(
                slot_dir=Path(lease.slot_dir),
                parent_sha=parent_sha,
                component=component,
                manifest_globs=self.manifest_globs,
                reflective_dataset=reflective_dataset,
                components_to_update=components_to_update,
                metadata=metadata or {},
            )
            self.agent(Path(lease.slot_dir), ctx)
            self.total_cost = float(getattr(self.agent, "total_cost", self.total_cost) or self.total_cost)
            try:
                child_sha = self.pool.commit_worktree(lease.slot_dir, self._message(ctx), self.manifest_globs)
            except ValueError:
                # An out-of-manifest / symlink / gitlink edit. Treat as a rejected
                # no-op (return the parent) so a single bad proposal doesn't abort
                # the run; GEPA's acceptance test discards the unchanged candidate.
                child_sha = parent_sha
        finally:
            self.pool.release(lease)
        return {component: child_sha}
