# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Git-commit candidate mode for the autoresearch engine (LLM-free plumbing test).

autoresearch drives a shell agent that self-commits inside a persistent leased
worktree and scores SHAs via a git-aware ``eval.sh``. A full run needs a live
coding CLI, so this test monkeypatches the ``_run_claude`` seam to simulate the
agent's effect (a committed candidate + a server best) and verifies the
git-specific plumbing: the persistent slot is leased and released, ``eval.sh`` /
``program.md`` are materialized in git-commit form, and the winning candidate is
the best-scoring commit SHA.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

import gepa.oa.engines.autoresearch as ar
from gepa.oa.budget import BudgetTracker
from gepa.oa.config import OptimizeAnythingConfig
from gepa.oa.repo_pool import GitCheckoutHelper, GitWorktreePool
from gepa.oa.task import Task

_HEX40 = re.compile(r"^[0-9a-f]{40}$")


@pytest.fixture(autouse=True)
def _skip_claude_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ar, "preflight_claude_engine", lambda *a, **k: None)


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True)


class _FakeServer:
    """Minimal EvalServer stand-in (no HTTP); the fake agent sets best_* directly."""

    def __init__(self) -> None:
        self.budget = BudgetTracker(max_evals=10)
        self.url = "http://127.0.0.1:9"
        self.best_score: float | None = None
        self.best_candidate: str | None = None
        self.eval_log: list[dict] = []


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


def test_autoresearch_git_commit_plumbing(repo: tuple[Path, str], tmp_path: Path) -> None:
    repo_dir, base = repo
    slots = [repo_dir / ".gepa_worktrees" / f"slot{i}" for i in range(2)]
    pool = GitWorktreePool(GitCheckoutHelper(repo_dir), slots, base).start()
    run_dir = tmp_path / "run"
    server = _FakeServer()
    task = Task(name="ar-git", seed_candidate=base, objective="Maximize src/value.txt")

    engine = ar.AutoResearchEngine(
        OptimizeAnythingConfig(
            engine="autoresearch",
            sandbox=False,
            run_dir=str(run_dir),
            max_evals=10,
            git_commit={
                "repo_dir": str(repo_dir),
                "manifest_globs": ["src"],
                "worktree_pool": pool,
                "base_commit": base,
            },
            engine_config={"ralph": False},
        )
    )

    def fake_run_claude(*, work_dir, session_id, prompt, budget, adapter_cost, resume, env, repo=None):
        # Simulate the agent: commit an improved candidate (no worktree needed —
        # the run already holds the slot) and record it as the server's best.
        child = pool._helper.commit_index_only(base, {"src/value.txt": "1\n"})
        server.best_candidate = child
        server.best_score = 1.0
        server.eval_log.append({"score": 1.0, "candidate": child})
        return subprocess.CompletedProcess([], 0, json.dumps({"total_cost_usd": 0.1}), "")

    engine._run_claude = fake_run_claude  # type: ignore[method-assign]

    try:
        result = engine.run(task, server)
    finally:
        pool.teardown()

    # Winning candidate is the best-scoring commit SHA.
    assert isinstance(result.best_candidate, str) and _HEX40.match(result.best_candidate)
    assert result.best_candidate != base
    assert result.best_score == 1.0

    # The persistent slot lease was released (no dangling refcount).
    assert all(slot.refcount == 0 for slot in pool._slots)

    # eval.sh and program.md were materialized in git-commit form.
    eval_sh = (run_dir / "eval.sh").read_text()
    assert "rev-parse" in eval_sh and ".gepa_worktrees" in eval_sh  # resolves a SHA in the leased worktree
    program_md = (run_dir / "program.md").read_text()
    assert "git-commit mode" in program_md
