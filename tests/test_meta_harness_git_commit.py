# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Git-commit candidate mode for the meta_harness engine (LLM-free).

Monkeypatches the engine's ``_invoke_proposer`` seam so no real ``claude`` runs:
the fake agent edits the leased worktree slot and writes ``pending_eval.json``.
The engine then commits the slot via the pool and scores the resulting SHA, so
this exercises the whole host-commit path — lease -> agent edit -> commit ->
score a SHA -> a commit SHA as the winning candidate.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

import gepa.oa.engines.meta_harness as mh
from gepa.oa.budget import BudgetTracker
from gepa.oa.config import OptimizeAnythingConfig
from gepa.oa.eval_server import EvalServer
from gepa.oa.repo_pool import GitCheckoutHelper, GitWorktreePool
from gepa.oa.task import Task

_HEX40 = re.compile(r"^[0-9a-f]{40}$")


@pytest.fixture(autouse=True)
def _skip_claude_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mh, "preflight_claude_engine", lambda *a, **k: None)


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


def test_meta_harness_git_commit_mode(repo: tuple[Path, str], tmp_path: Path) -> None:
    repo_dir, base = repo
    slots = [repo_dir / ".gepa_worktrees" / f"slot{i}" for i in range(2)]
    pool = GitWorktreePool(GitCheckoutHelper(repo_dir), slots, base).start()

    def evaluate(candidate: str, example=None) -> tuple[float, dict]:
        lease = pool.lease(candidate)
        try:
            value = int((lease.slot_dir / "src" / "value.txt").read_text().strip())
        finally:
            pool.release(lease)
        return float(value), {"value": value}

    task = Task(name="mh-git", seed_candidate=base, objective="Maximize src/value.txt")
    server = EvalServer(task, evaluate=evaluate, budget=BudgetTracker(max_evals=20))

    engine = mh.MetaHarnessEngine(
        OptimizeAnythingConfig(
            engine="meta_harness",
            sandbox=False,
            run_dir=str(tmp_path / "run"),
            max_evals=20,
            git_commit={
                "repo_dir": str(repo_dir),
                "manifest_globs": ["src"],
                "worktree_pool": pool,
                "base_commit": base,
            },
            engine_config={"max_iterations": 2, "max_candidates_per_iter": 1},
        )
    )

    def fake_invoke(repo_c, *, work_dir, slot_dir, iteration, max_budget_usd, pending_path, sessions_dir):
        # The "coding agent": edit the leased slot in place (set value=iteration),
        # then declare one candidate. The host commits + records the SHA.
        (Path(slot_dir) / "src" / "value.txt").write_text(f"{iteration}\n")
        pending_path.write_text(
            json.dumps(
                {
                    "iteration": iteration,
                    "candidates": [{"name": f"cand{iteration}", "file": f"agents/iter{iteration}_cand.txt"}],
                }
            )
        )
        return 0, 0.0, f"session-{iteration}"

    engine._invoke_proposer = fake_invoke  # type: ignore[method-assign]

    try:
        result = engine.run(task, server)
    finally:
        pool.teardown()

    # baseline (value 0) plus iter1 (value 1) and iter2 (value 2) were scored;
    # the winner is the iter-2 commit.
    assert isinstance(result.best_candidate, str) and _HEX40.match(result.best_candidate)
    assert result.best_candidate != base
    assert result.best_score == 2.0
    reachable = subprocess.run(["git", "-C", str(repo_dir), "cat-file", "-e", f"{result.best_candidate}^{{commit}}"])
    assert reachable.returncode == 0
