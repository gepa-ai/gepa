# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for the agent-agnostic git-commit proposer (:mod:`gepa.oa.proposers.git_agent`).

Drives :class:`GitAgentProposer` with scripted (LLM-free) agents against a real
throwaway git repo, covering the happy path, no-op proposals, the
reward-hack/out-of-manifest rejection falling back to the parent, component
resolution, and agent cost mirroring.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from gepa.oa.proposers.git_agent import GitAgentProposer, ProposalContext
from gepa.oa.repo_pool import GitCheckoutHelper, GitWorktreePool


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> tuple[Path, str]:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    _git(repo_dir, "init", "-q")
    _git(repo_dir, "config", "user.email", "t@example.com")
    _git(repo_dir, "config", "user.name", "Test")
    (repo_dir / "src").mkdir()
    (repo_dir / "src" / "value.txt").write_text("0\n")
    _git(repo_dir, "add", "-A")
    _git(repo_dir, "commit", "-qm", "base")
    base = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    return repo_dir, base


@pytest.fixture
def pool(repo: tuple[Path, str]) -> GitWorktreePool:
    repo_dir, base = repo
    helper = GitCheckoutHelper(repo_dir)
    slots = [repo_dir / ".gepa_worktrees" / f"slot{i}" for i in range(2)]
    p = GitWorktreePool(helper, slots, base).start()
    yield p
    p.teardown()


def _bump_agent(slot_dir: Path, ctx: ProposalContext) -> None:
    """Scripted coding agent: increment the integer in src/value.txt."""
    f = slot_dir / "src" / "value.txt"
    f.write_text(f"{int(f.read_text().strip()) + 1}\n")


def test_proposer_commits_edit(repo, pool) -> None:
    _repo_dir, base = repo
    proposer = GitAgentProposer(_bump_agent, pool, manifest_globs=["src"])
    out = proposer({"current_candidate": base}, {}, ["current_candidate"])
    child = out["current_candidate"]
    assert child != base
    reader = pool.lease(child)
    assert (reader.slot_dir / "src" / "value.txt").read_text() == "1\n"
    pool.release(reader)


def test_proposer_noop_returns_parent(repo, pool) -> None:
    _repo_dir, base = repo

    def do_nothing(slot_dir: Path, ctx: ProposalContext) -> None:
        pass

    proposer = GitAgentProposer(do_nothing, pool, manifest_globs=["src"])
    out = proposer({"current_candidate": base}, {}, ["current_candidate"])
    assert out["current_candidate"] == base


def test_reward_hack_edit_falls_back_to_parent(repo, pool) -> None:
    """A symlink inside the manifest is rejected by the pool; the proposer must
    fall back to the parent SHA rather than abort the run."""
    _repo_dir, base = repo

    def sneaky(slot_dir: Path, ctx: ProposalContext) -> None:
        os.symlink("/etc/passwd", slot_dir / "src" / "leak")

    proposer = GitAgentProposer(sneaky, pool, manifest_globs=["src"])
    out = proposer({"current_candidate": base}, {}, ["current_candidate"])
    assert out["current_candidate"] == base


def test_component_resolution_single_key(repo, pool) -> None:
    """A single-component candidate uses its sole key, even when it differs from
    the default component name."""
    _repo_dir, base = repo
    proposer = GitAgentProposer(_bump_agent, pool, manifest_globs=["src"])
    out = proposer({"repo": base}, {}, ["repo"])
    assert set(out) == {"repo"}
    assert out["repo"] != base


def test_total_cost_mirrors_agent(repo, pool) -> None:
    """When the agent object exposes ``total_cost``, the proposer surfaces it for
    the GepaEngine cost source."""
    _repo_dir, base = repo

    class CostingAgent:
        total_cost = 0.0

        def __call__(self, slot_dir: Path, ctx: ProposalContext) -> None:
            _bump_agent(slot_dir, ctx)
            self.total_cost += 0.25

    agent = CostingAgent()
    proposer = GitAgentProposer(agent, pool, manifest_globs=["src"])
    proposer({"current_candidate": base}, {}, ["current_candidate"])
    assert proposer.total_cost == pytest.approx(0.25)


def test_empty_manifest_rejected(pool) -> None:
    with pytest.raises(ValueError, match="manifest_globs"):
        GitAgentProposer(_bump_agent, pool, manifest_globs=[])
