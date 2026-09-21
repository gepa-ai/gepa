"""Shared helpers for git-commit candidate mode.

In git-commit mode a candidate is a git commit SHA rather than a text string
(see :mod:`gepa.oa.repo_pool`). Every engine that supports the mode reads the
same ``OptimizeAnythingConfig.git_commit`` dict and drives the same pool. This
module centralizes:

* :class:`RepoCandidate` — the parsed handles (repo dir, editable manifest, base
  commit, pool, optional coding-agent callable),
* :func:`resolve_repo_candidate` — parse/validate ``config.git_commit`` (returns
  ``None`` for ordinary text mode), and
* :func:`commit_agent_edit` — the host-commit flow used by the ``gepa`` and
  ``meta_harness`` engines: lease the parent SHA, run a coding agent that edits
  the leased worktree in place, mint the child commit, and return its SHA.

The ``autoresearch`` engine uses a *self-commit* flow instead (the agent commits
inside a persistent leased slot via shell), so it uses
:func:`resolve_repo_candidate` and the pool directly rather than
:func:`commit_agent_edit`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gepa.oa.repo_pool import RepoCandidatePool

_REQUIRED_KEYS = ("repo_dir", "manifest_globs", "worktree_pool")


@dataclass
class RepoCandidate:
    """Parsed ``config.git_commit`` handles for git-commit candidate mode."""

    repo_dir: Path
    manifest_globs: list[str]
    pool: RepoCandidatePool
    base_commit: str
    agent: Any = None


def resolve_repo_candidate(
    git_commit: dict[str, Any] | None,
    *,
    seed: str | None = None,
) -> RepoCandidate | None:
    """Parse ``config.git_commit`` into a :class:`RepoCandidate`, or ``None``.

    Returns ``None`` when ``git_commit`` is absent/empty (ordinary text mode).
    Raises ``ValueError`` on a half-wired dict (missing ``repo_dir`` /
    ``manifest_globs`` / ``worktree_pool``, or no base commit). ``base_commit``
    defaults to ``seed`` (typically the task's seed candidate).
    """
    if not git_commit:
        return None
    missing = [k for k in _REQUIRED_KEYS if not git_commit.get(k)]
    if missing:
        raise ValueError(f"config.git_commit is missing required keys: {missing}")
    base = git_commit.get("base_commit") or seed
    if not base:
        raise ValueError("config.git_commit needs a 'base_commit' (or a seed_candidate to default from)")
    return RepoCandidate(
        repo_dir=Path(git_commit["repo_dir"]).resolve(),
        manifest_globs=list(git_commit["manifest_globs"]),
        pool=git_commit["worktree_pool"],
        base_commit=base,
        agent=git_commit.get("agent"),
    )


def commit_agent_edit(
    pool: RepoCandidatePool,
    parent_sha: str,
    manifest_globs: Sequence[str],
    edit_fn: Callable[[Path], None],
    message: str,
) -> str:
    """Host-commit flow: run a coding agent against a leased slot, mint a commit.

    Leases ``parent_sha`` (``exclusive=True`` so the agent owns the working
    tree), calls ``edit_fn(slot_dir)`` to edit files in place within
    ``manifest_globs``, then mints the child commit via
    ``pool.commit_worktree`` and returns its SHA. A no-op edit (nothing staged)
    or a manifest-violating edit (out-of-manifest path / symlink / gitlink,
    which ``commit_worktree`` raises ``ValueError`` on) returns ``parent_sha``
    unchanged, so a single bad proposal never aborts the run.
    """
    lease = pool.lease(parent_sha, exclusive=True)
    try:
        edit_fn(Path(lease.slot_dir))
        try:
            return pool.commit_worktree(lease.slot_dir, message, list(manifest_globs))
        except ValueError:
            return parent_sha
    finally:
        pool.release(lease)
