# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for the git-worktree candidate pool (:mod:`gepa.oa.repo_pool`).

All tests run against a real throwaway git repo under ``tmp_path`` — no network,
no LLM. They cover the pool contract (lease/release/commit_worktree), the
manifest diff-allowlist gate, index-only commits, gc-pinning, and concurrency.
"""

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path

import pytest

from gepa.oa.repo_pool import GitCheckoutHelper, GitWorktreePool


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True)


def _sha(cwd: Path, *rev: str) -> str:
    out = subprocess.run(["git", "-C", str(cwd), "rev-parse", *rev], check=True, capture_output=True, text=True)
    return out.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> tuple[Path, str]:
    """A git repo with ``src/value.txt`` + ``README.md``; returns (repo_dir, base_sha)."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    _git(repo_dir, "init", "-q")
    _git(repo_dir, "config", "user.email", "t@example.com")
    _git(repo_dir, "config", "user.name", "Test")
    (repo_dir / "src").mkdir()
    (repo_dir / "src" / "value.txt").write_text("0\n")
    (repo_dir / "README.md").write_text("readme\n")
    _git(repo_dir, "add", "-A")
    _git(repo_dir, "commit", "-qm", "base")
    return repo_dir, _sha(repo_dir, "HEAD")


def _pool(repo_dir: Path, base: str, k: int = 2) -> GitWorktreePool:
    helper = GitCheckoutHelper(repo_dir)
    slots = [repo_dir / ".gepa_worktrees" / f"slot{i}" for i in range(k)]
    return GitWorktreePool(helper, slots, base)


def test_commit_worktree_mints_readable_child(repo: tuple[Path, str]) -> None:
    repo_dir, base = repo
    pool = _pool(repo_dir, base).start()
    try:
        writer = pool.lease(base, exclusive=True)
        (writer.slot_dir / "src" / "value.txt").write_text("1\n")
        child = pool.commit_worktree(writer.slot_dir, "cand: bump", ["src"])
        pool.release(writer)

        assert child != base
        reader = pool.lease(child)
        assert (reader.slot_dir / "src" / "value.txt").read_text() == "1\n"
        pool.release(reader)
        # scored candidate is pinned so gc cannot prune the detached commit
        assert any(child in ref for ref in pool._helper.list_candidate_refs())
    finally:
        pool.teardown()


def test_noop_proposal_returns_parent(repo: tuple[Path, str]) -> None:
    """No in-manifest change must yield the parent SHA, not a failed commit."""
    repo_dir, base = repo
    pool = _pool(repo_dir, base).start()
    try:
        writer = pool.lease(base, exclusive=True)
        # touch only an out-of-manifest file; nothing in "src" changes
        (writer.slot_dir / "README.md").write_text("changed but out of manifest\n")
        result = pool.commit_worktree(writer.slot_dir, "noop", ["src"])
        pool.release(writer)
        assert result == base
    finally:
        pool.teardown()


def test_out_of_manifest_edit_rejected(repo: tuple[Path, str]) -> None:
    """A self-committed change outside the manifest fails the diff allowlist."""
    repo_dir, base = repo
    helper = GitCheckoutHelper(repo_dir)
    pool = _pool(repo_dir, base).start()
    try:
        writer = pool.lease(base, exclusive=True)
        (writer.slot_dir / "README.md").write_text("sneaky\n")
        _git(writer.slot_dir, "add", "README.md")
        _git(writer.slot_dir, "-c", "core.hooksPath=/dev/null", "commit", "--no-verify", "-qm", "self")
        self_sha = _sha(writer.slot_dir, "HEAD")
        with pytest.raises(ValueError, match="out-of-manifest"):
            helper.check_diff_allowlist(base, self_sha, ["src"])
        pool.release(writer)
    finally:
        pool.teardown()


def test_symlink_in_manifest_rejected(repo: tuple[Path, str]) -> None:
    repo_dir, base = repo
    pool = _pool(repo_dir, base).start()
    try:
        writer = pool.lease(base, exclusive=True)
        os.symlink("/etc/passwd", writer.slot_dir / "src" / "link")
        with pytest.raises(ValueError, match="symlink/gitlink"):
            pool.commit_worktree(writer.slot_dir, "sym", ["src"])
        pool.release(writer)
    finally:
        pool.teardown()


def test_commit_index_only_no_worktree(repo: tuple[Path, str]) -> None:
    """Index-only commits build a child with no working tree (thread-safe path)."""
    repo_dir, base = repo
    pool = _pool(repo_dir, base).start()
    try:
        child = pool._helper.commit_index_only(base, {"src/generated.txt": "made via index\n"})
        assert child != base
        reader = pool.lease(child)
        assert (reader.slot_dir / "src" / "generated.txt").read_text() == "made via index\n"
        # the pre-existing file is preserved (seeded from parent tree)
        assert (reader.slot_dir / "src" / "value.txt").read_text() == "0\n"
        pool.release(reader)
    finally:
        pool.teardown()


def test_readers_share_slot_writer_does_not(repo: tuple[Path, str]) -> None:
    """Two readers at the same SHA share one slot; an exclusive writer never shares."""
    repo_dir, base = repo
    pool = _pool(repo_dir, base, k=2).start()
    try:
        r1 = pool.lease(base)
        r2 = pool.lease(base)  # exact-match reuse of the same slot
        assert r1.slot_dir == r2.slot_dir
        pool.release(r1)
        pool.release(r2)

        w = pool.lease(base, exclusive=True)
        r3 = pool.lease(base)  # cannot share the writer's slot
        assert r3.slot_dir != w.slot_dir
        pool.release(r3)
        pool.release(w)
    finally:
        pool.teardown()


def test_teardown_drops_candidate_refs(repo: tuple[Path, str]) -> None:
    repo_dir, base = repo
    helper = GitCheckoutHelper(repo_dir)
    pool = GitWorktreePool(helper, [repo_dir / ".gepa_worktrees" / "slot0"], base).start()
    writer = pool.lease(base, exclusive=True)
    (writer.slot_dir / "src" / "value.txt").write_text("9\n")
    pool.commit_worktree(writer.slot_dir, "c", ["src"])
    pool.release(writer)
    assert helper.list_candidate_refs()  # pinned during the run
    pool.teardown()
    assert helper.list_candidate_refs() == []  # cleaned on teardown


def test_concurrent_leases_never_exceed_k(repo: tuple[Path, str]) -> None:
    """K threads may hold leases at once; the (K+1)th blocks until one frees."""
    repo_dir, base = repo
    k = 3
    pool = _pool(repo_dir, base, k=k).start()
    live = 0
    peak = 0
    lock = threading.Lock()
    start = threading.Event()

    def worker() -> None:
        nonlocal live, peak
        start.wait()
        lease = pool.lease(base)
        with lock:
            live += 1
            peak = max(peak, live)
        # hold briefly so contention is real
        threading.Event().wait(0.02)
        with lock:
            live -= 1
        pool.release(lease)

    try:
        threads = [threading.Thread(target=worker) for _ in range(k * 3)]
        for t in threads:
            t.start()
        start.set()
        for t in threads:
            t.join(timeout=30)
        assert peak <= k
    finally:
        pool.teardown()
