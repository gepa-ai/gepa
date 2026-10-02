# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""End-to-end test of git-commit candidate mode through the real GEPA engine.

No LLM and no network: a scripted coding agent edits a file, the pool mints the
candidate commit, and a trivial evaluator scores the checked-out worktree. This
exercises the whole path — ``OptimizeAnythingConfig.git_commit`` -> GepaEngine
wiring -> GitAgentProposer (lease/edit/commit) -> evaluate a SHA -> a commit SHA
as the winning candidate.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from gepa.oa.config import OptimizeAnythingConfig
from gepa.oa.proposers.git_agent import ProposalContext
from gepa.oa.repo_pool import GitCheckoutHelper, GitWorktreePool
from gepa.optimize_anything import optimize_anything

_HEX40 = re.compile(r"^[0-9a-f]{40}$")


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


def test_git_commit_mode_evolves_a_repo(repo: tuple[Path, str]) -> None:
    repo_dir, base = repo
    helper = GitCheckoutHelper(repo_dir)
    slots = [repo_dir / ".gepa_worktrees" / f"slot{i}" for i in range(4)]
    pool = GitWorktreePool(helper, slots, base).start()

    def bump_agent(slot_dir: Path, ctx: ProposalContext) -> None:
        """Increment the integer in src/value.txt — a monotonically improving edit."""
        f = slot_dir / "src" / "value.txt"
        f.write_text(f"{int(f.read_text().strip()) + 1}\n")

    def evaluate(candidate: str, example=None) -> tuple[float, dict]:
        """Score = the integer at the candidate SHA (checked out via the pool)."""
        lease = pool.lease(candidate)
        try:
            value = int((lease.slot_dir / "src" / "value.txt").read_text().strip())
        finally:
            pool.release(lease)
        return float(value), {"value": value}

    config = OptimizeAnythingConfig(
        engine="gepa",
        max_evals=25,
        git_commit={
            "repo_dir": str(repo_dir),
            "manifest_globs": ["src"],
            "worktree_pool": pool,
            "agent": bump_agent,
        },
        engine_config={"reflection": {"reflection_minibatch_size": 1, "reflection_lm": None}},
    )

    try:
        result = optimize_anything(
            seed_candidate=base,
            evaluator=evaluate,
            dataset=[{"id": "t0"}],
            valset=[{"id": "v0"}],
            objective="Maximize the integer in src/value.txt",
            config=config,
        )
    finally:
        pool.teardown()

    best = result.best_candidate
    assert isinstance(best, str) and _HEX40.match(best), f"best candidate must be a SHA, got {best!r}"
    assert best != base, "the optimizer should have moved off the seed commit"
    # seed value is 0; any accepted candidate has a strictly larger value.
    assert result.val_aggregate_scores[result.best_idx] >= 1.0
    # the winning SHA is a real commit reachable in the repo
    reachable = subprocess.run(["git", "-C", str(repo_dir), "cat-file", "-e", f"{best}^{{commit}}"])
    assert reachable.returncode == 0
