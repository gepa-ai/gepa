# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Git-commit candidate mode: optimize a repository, one commit at a time.

In git mode a GEPA candidate is a *git commit SHA* rather than a text string.
GEPA evolves the repo by committing edits from a coding agent, and your
evaluator scores a real working tree checked out at each candidate SHA. The
winning candidate is a commit you can ``git checkout``.

Three pieces:

* a **pool** (:class:`gepa.oa.repo_pool.GitWorktreePool`) that keeps K warm
  worktrees and moves them between SHAs cheaply, isolating concurrent
  proposers/evaluators and gc-pinning scored commits;
* a **proposer** (:class:`gepa.oa.proposers.git_agent.GitAgentProposer`) that
  leases the parent SHA, runs *your* coding agent to edit files, and mints the
  child commit — rejecting any edit outside the manifest;
* your **evaluator**, which leases a candidate SHA and scores its worktree.

This script uses a scripted agent (increment an integer in a file) so it runs
with no API key. Swap ``agent`` for a real coding-CLI binding to optimize actual
code; the rest is unchanged.

Run it::

    python -m gepa.examples.git_mode.optimize_repo    # from a source checkout
    # or: python examples/git_mode/optimize_repo.py
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from gepa.oa.config import OptimizeAnythingConfig
from gepa.oa.proposers.git_agent import ProposalContext
from gepa.oa.repo_pool import GitCheckoutHelper, GitWorktreePool
from gepa.optimize_anything import optimize_anything


def _make_demo_repo(root: Path) -> str:
    """Create a throwaway git repo with ``src/value.txt`` and return its base SHA."""
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "demo@example.com"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Demo"], check=True)
    (root / "src").mkdir()
    (root / "src" / "value.txt").write_text("0\n")
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "base"], check=True)
    out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    return out.stdout.strip()


def agent(slot_dir: Path, ctx: ProposalContext) -> None:
    """A scripted 'coding agent': improve the candidate by bumping the integer.

    A real agent would inspect ``ctx.reflective_dataset`` (per-example scores and
    feedback from the last evaluation) and edit source files in ``slot_dir``
    accordingly — e.g. shell out to a coding CLI with ``cwd=slot_dir``. It edits
    in place and returns nothing; the pool commits the manifest afterwards.
    """
    f = slot_dir / "src" / "value.txt"
    f.write_text(f"{int(f.read_text().strip()) + 1}\n")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="gepa-git-mode-") as tmp:
        repo_dir = Path(tmp) / "repo"
        repo_dir.mkdir()
        base_sha = _make_demo_repo(repo_dir)

        # A K-bound pool of warm worktrees over the repo's own object store.
        pool = GitWorktreePool(
            GitCheckoutHelper(repo_dir),
            slot_dirs=[repo_dir / ".gepa_worktrees" / f"slot{i}" for i in range(4)],
            base_commit=base_sha,
        ).start()

        # The evaluator scores a candidate SHA by checking it out via the pool.
        def evaluate(candidate: str, example=None) -> tuple[float, dict]:
            lease = pool.lease(candidate)
            try:
                value = int((lease.slot_dir / "src" / "value.txt").read_text().strip())
            finally:
                pool.release(lease)
            return float(value), {"value": value}

        config = OptimizeAnythingConfig(
            engine="gepa",
            max_evals=25,
            # Opt into git-commit mode. The engine builds a GitAgentProposer from
            # `agent` and turns off text-merge; your evaluator closes over `pool`.
            git_commit={
                "repo_dir": str(repo_dir),
                "manifest_globs": ["src"],  # the editable surface (and anti-hacking gate)
                "worktree_pool": pool,
                "agent": agent,
            },
            engine_config={"reflection": {"reflection_minibatch_size": 1, "reflection_lm": None}},
        )

        try:
            result = optimize_anything(
                seed_candidate=base_sha,  # the candidate is a commit SHA
                evaluator=evaluate,
                dataset=[{"id": "t0"}],
                valset=[{"id": "v0"}],
                objective="Maximize the integer in src/value.txt",
                config=config,
            )
        finally:
            pool.teardown()

        best_sha = result.best_candidate
        best_score = result.val_aggregate_scores[result.best_idx]
        print(f"\nseed commit : {base_sha}")
        print(f"best commit : {best_sha}  (score {best_score})")
        print(f"inspect it  : git -C <repo> show {best_sha}:src/value.txt")


if __name__ == "__main__":
    main()
