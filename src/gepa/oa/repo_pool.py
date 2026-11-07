"""Repo-candidate pool — the pluggable surface for repo-level GEPA candidates.

When a GEPA candidate is a *git commit SHA* rather than a text string, scoring it
needs a real working tree checked out at that SHA so a build / run / judge
pipeline can execute against it. The git-commit engines consume a single
duck-typed pool handle. This module formalizes that handle as the
:class:`RepoCandidatePool` Protocol and ships a portable, dependency-free default
implementation (:class:`GitWorktreePool` + :class:`GitCheckoutHelper`) built on
plain ``git worktree`` over the repo's own object store.

The Protocol is intentionally minimal — exactly the three methods the engines
duck-type:

* ``lease(sha) -> Lease`` (Lease exposes ``.slot_dir`` and ``.sha``),
* ``release(lease) -> None``,
* ``commit_worktree(slot_dir, message, manifest_globs) -> child_sha``.

``commit_worktree`` is used by the "host commits the slot" flow (an outer loop
that edits a leased slot and then mints the child commit for it); a coding agent
that self-commits via shell uses only ``lease`` / ``release``. Both live on one
Protocol because the injected handle is a single object.

Lifecycle (``start`` / ``teardown`` / context-manager) and the richer
:class:`GitCheckoutHelper` surface (``checkout`` / ``restore`` /
``check_diff_allowlist`` / ``commit_index_only`` / ref-pinning / gc control) are
caller-driven and live on the concrete classes, NOT the Protocol — an
alternative implementation (e.g. a sparse-checkout- or overlay-backed pool) may
set up slots completely differently while still satisfying the
lease/release/commit_worktree contract.

This module is pure stdlib so it stays a portable default that any caller can
use.

Implementation notes for the default :class:`GitWorktreePool`:

* Creating a worktree from scratch is expensive on a large repo (checking out
  the full tree), so the pool keeps a small K-bound set of long-lived worktrees
  ("slots") and moves them between SHAs with a cheap incremental
  ``git checkout --detach`` (gitignored build caches survive the checkout).
* Worktrees are co-located UNDER ``repo_dir`` (``repo_dir/.gepa_worktrees/slotN``)
  so a single writable bind of ``repo_dir`` covers BOTH the worktree dir AND the
  shared ``.git`` common dir. A worktree's ``.git`` is a *file* pointing at
  ``<repo>/.git/worktrees/<name>`` whose ``commondir`` points back at
  ``<repo>/.git``; if a sandbox binds only the slot it severs that link and
  ``git`` breaks, so the bind must cover the resolved git-common-dir too.
* Commits use ``git -c core.hooksPath=/dev/null commit --no-verify`` because
  candidate commits are machine-generated: repo pre-commit hooks (formatters,
  linters, secret scanners) may be absent, slow, interactive, or reject
  nondeterministically, so bypassing them keeps commit minting deterministic.
* ``gc.auto`` is set to 0 for the duration of the run and every scored SHA is
  pinned with an atomic ``update-ref refs/gepa/cand/<sha>`` so detached
  candidate commits are not pruned. Teardown restores ``gc.auto`` and deletes the
  pinned refs.
* ``add_worktree`` is serialized through a process-wide lock (``git worktree add``
  takes the repo's worktree lock; concurrent invocations race) and is preceded by
  ``worktree prune`` + stale-slot removal so re-runs do not collide with orphans.
* ``commit_worktree`` enforces a diff allowlist (``git diff --raw base..sha``):
  any path outside ``manifest_globs`` or any introduced/changed symlink (mode
  ``120000``) or gitlink (mode ``160000``) raises, catching stray or
  out-of-scope edits cheaply.

The pool is K-bound (RAM/disk) and DECOUPLED from any evaluator concurrency via a
``Semaphore(K)``; ``lease`` picks the slot whose HEAD is the nearest ancestor of
the requested SHA (``git rev-list --count <slot_head>..sha``) for a warm
parent->child handoff, and never round-robins. A slot with refcount > 0 is never
re-checked-out (no thrash).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
from collections import OrderedDict
from collections.abc import Sequence
from fnmatch import fnmatch
from pathlib import Path
from typing import Protocol

log = logging.getLogger(__name__)

# Untracked paths that must SURVIVE a restore()'s `git clean`. Matched as
# repo-root-relative paths (NOT bare basenames). Empty by default; a caller with
# expensive build caches (e.g. a JS repo's node_modules) passes its own list so
# those caches persist between candidates instead of being rebuilt from cold.
_DEFAULT_CLEAN_KEEP: tuple[str, ...] = ()

# Ref namespace used to pin scored candidate commits so gc cannot prune them.
_CAND_REF_PREFIX = "refs/gepa/cand/"

# Process-wide lock serializing `git worktree add` across all helpers/pools in
# this process (the repo's worktree lock is per-repo, but a module-level lock is
# simplest and the add path is the slow one-time cost anyway).
_WORKTREE_ADD_LOCK = threading.Lock()


class Lease:
    """A held slot in a repo-candidate pool, checked out to a candidate SHA."""

    def __init__(self, slot_dir: Path, sha: str) -> None:
        self.slot_dir = slot_dir
        self.sha = sha

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Lease(slot={self.slot_dir.name}, sha={self.sha[:12]})"


class RepoCandidatePool(Protocol):
    """The minimal pool surface the git-commit engines duck-type.

    A candidate is a git commit SHA. An engine receives a concrete pool and calls
    only these methods on it; the caller chooses the implementation (the portable
    :class:`GitWorktreePool` default, or a custom sparse/overlay-backed pool).

    ``commit_worktree`` is used only by the flow where the host commits a slot it
    edited; a self-committing coding agent uses only ``lease`` / ``release``. All
    three live on one Protocol because the injected handle is a single object.

    A plain (non-``runtime_checkable``) Protocol because every call site relies on
    duck typing; any object exposing these three methods conforms.
    """

    def lease(self, sha: str, *, exclusive: bool = False) -> Lease:
        """Acquire a worktree slot checked out at ``sha`` (blocks on a K-bound
        semaphore). The returned :class:`Lease` exposes ``.slot_dir`` (the working
        dir to edit/build) and ``.sha``.

        ``exclusive=True`` (used by *writers* — a proposer that edits + commits
        the slot) takes a slot no other lease will share for the duration, so two
        writers (or a reader scoring the same SHA) can never co-occupy one working
        tree. The default (read-only scoring) may share a slot already at ``sha``.
        """
        ...

    def release(self, lease: Lease) -> None:
        """Return a leased slot to the pool. Always called in a ``finally``."""
        ...

    def commit_worktree(
        self,
        slot_dir: Path | str,
        message: str,
        manifest_globs: Sequence[str],
    ) -> str:
        """Stage exactly ``manifest_globs`` in ``slot_dir``, commit, pin the
        resulting SHA, gate the diff against ``manifest_globs`` (raising
        ``ValueError`` on out-of-manifest paths / symlinks / gitlinks), and return
        the child commit SHA."""
        ...


def _run_git(
    cwd: Path | str,
    *args: str,
    config: Sequence[str] = (),
    check: bool = True,
    stdin: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Run ``git -C <cwd> [-c k=v ...] <args>`` and return the completed process.

    ``config`` entries are passed as ``-c k=v`` BEFORE the subcommand (the only
    position git accepts global ``-c`` flags). Output is captured as bytes.
    """
    cmd = ["git", "-C", str(cwd)]
    for entry in config:
        cmd.extend(["-c", entry])
    cmd.extend(args)
    return subprocess.run(cmd, input=stdin, capture_output=True, check=check)


def _git_out(cwd: Path | str, *args: str, config: Sequence[str] = ()) -> str:
    """Run git and return stripped stdout (raises on non-zero exit)."""
    return _run_git(cwd, *args, config=config).stdout.decode().strip()


class GitCheckoutHelper:
    """Stateless git plumbing around a single repo's shared object store.

    All methods operate by path; the helper holds no per-slot state so it can be
    shared across pool slots and threads. Mutating worktree operations that take
    the repo worktree lock (``add_worktree``) are serialized via a process-wide
    lock; the cheaper per-slot operations rely on the pool's slot refcounting to
    avoid concurrent use of the same slot.
    """

    def __init__(self, repo_dir: Path | str) -> None:
        self._repo_dir = Path(repo_dir).resolve()

    @property
    def repo_dir(self) -> Path:
        return self._repo_dir

    # ------------------------------------------------------------------
    # Worktree lifecycle
    # ------------------------------------------------------------------

    def add_worktree(self, slot_dir: Path | str, at_commit: str) -> Path:
        """Create a detached worktree at ``slot_dir`` checked out to ``at_commit``.

        Serialized process-wide (the repo's worktree lock). Before adding we
        ``worktree prune`` (drop bookkeeping for vanished worktrees) and forcibly
        remove any stale slot already registered/existing at ``slot_dir`` so
        re-runs do not collide with orphans from a crashed prior run.
        """
        slot = Path(slot_dir).resolve()
        with _WORKTREE_ADD_LOCK:
            _run_git(self._repo_dir, "worktree", "prune")
            self._remove_stale_slot(slot)
            slot.parent.mkdir(parents=True, exist_ok=True)
            _run_git(self._repo_dir, "worktree", "add", "--detach", str(slot), at_commit)
        return slot

    def _remove_stale_slot(self, slot: Path) -> None:
        """Remove a worktree registration and/or directory left at ``slot``.

        Tries ``worktree remove --force`` first (cleans the registration), then
        deletes the directory if it still exists. Both are best-effort: a slot
        that was never registered just needs the directory removed.
        """
        _run_git(self._repo_dir, "worktree", "remove", "--force", str(slot), check=False)
        _run_git(self._repo_dir, "worktree", "prune")
        if slot.exists():
            shutil.rmtree(slot, ignore_errors=True)

    def remove_worktree(self, slot_dir: Path | str) -> None:
        """Force-remove a worktree (registration + directory) and prune."""
        with _WORKTREE_ADD_LOCK:
            self._remove_stale_slot(Path(slot_dir).resolve())

    # ------------------------------------------------------------------
    # Cheap per-slot SHA movement
    # ------------------------------------------------------------------

    def checkout(self, slot_dir: Path | str, sha: str) -> None:
        """Incrementally move ``slot_dir`` to ``sha`` (detached, forced)."""
        _run_git(slot_dir, "checkout", "--detach", "--force", sha)

    def restore(
        self,
        slot_dir: Path | str,
        sha: str,
        clean_keep: Sequence[str] = _DEFAULT_CLEAN_KEEP,
    ) -> None:
        """Hard-reset ``slot_dir`` to ``sha`` and clean untracked files.

        ``git clean -fdx`` removes everything not tracked (including gitignored
        build artifacts) EXCEPT the ``clean_keep`` allowlist, which is passed
        verbatim as ``-e`` excludes so expensive build caches survive between
        candidates.
        """
        _run_git(slot_dir, "reset", "--hard", sha)
        clean_args = ["clean", "-fdx"]
        for pattern in clean_keep:
            clean_args.extend(["-e", pattern])
        _run_git(slot_dir, *clean_args)

    def slot_head(self, slot_dir: Path | str) -> str:
        """Return the current HEAD commit SHA of a slot."""
        return _git_out(slot_dir, "rev-parse", "HEAD")

    # ------------------------------------------------------------------
    # Commit minting
    # ------------------------------------------------------------------

    def commit_worktree(
        self,
        slot_dir: Path | str,
        message: str,
        manifest_globs: Sequence[str],
    ) -> str:
        """Stage manifest paths in a worktree, commit, pin, and gate the diff.

        Stages exactly ``manifest_globs`` (relative to the slot), commits with
        repo hooks disabled (``core.hooksPath=/dev/null`` + ``--no-verify``), then
        atomically pins ``refs/gepa/cand/<sha>`` so the detached commit is not
        garbage-collected. Finally enforces the diff allowlist against the
        *parent*: if ``git diff --raw base..sha`` touches a path outside
        ``manifest_globs`` OR introduces/changes a symlink (mode ``120000``) or
        gitlink (mode ``160000``), the candidate is rejected with ``ValueError``
        AND the offending commit's pin ref is removed (it must not become a leased
        candidate).
        """
        slot = Path(slot_dir).resolve()
        # Stage each manifest pathspec INDEPENDENTLY, tolerating ones that match
        # nothing yet. `git add -- a b` is atomic: if ANY pathspec matches no file
        # (e.g. a manifest dir the seed doesn't have — the agent is expected to
        # CREATE it), git exits fatal and stages NOTHING, silently dropping the
        # real edits. Per-path with check=False stages what exists and skips
        # absent pathspecs, so a not-yet-created editable dir doesn't wipe the
        # commit. The diff allowlist below still gates every staged path against
        # manifest_globs.
        for spec in manifest_globs:
            _run_git(slot, "add", "--", spec, check=False)
        # No staged in-manifest change → a no-op proposal. Return the parent HEAD
        # unchanged rather than letting `git commit` fail on an empty diff; the
        # optimizer's acceptance test then rejects the unchanged candidate.
        if _run_git(slot, "diff", "--cached", "--quiet", check=False).returncode == 0:
            return _git_out(slot, "rev-parse", "HEAD")
        _run_git(
            slot,
            "commit",
            "--no-verify",
            "-m",
            message,
            config=["core.hooksPath=/dev/null"],
        )
        sha = _git_out(slot, "rev-parse", "HEAD")
        # Pin BEFORE validating so a concurrent gc cannot prune mid-check; remove
        # the pin if validation fails.
        self.pin_candidate(sha)
        try:
            self._enforce_diff_allowlist(slot, sha, manifest_globs)
        except ValueError:
            self.unpin_candidate(sha)
            raise
        return sha

    def _enforce_diff_allowlist(
        self,
        slot_dir: Path,
        sha: str,
        manifest_globs: Sequence[str],
    ) -> None:
        """Raise if ``base..sha`` touches paths outside manifest or any symlink.

        ``base`` is the commit's first parent. Delegates the diff parse + gate to
        :meth:`check_diff_allowlist` so the same logic backs both commit minting
        (here) and an evaluator's out-of-scope gate against the run base commit.
        """
        base = self._first_parent(slot_dir, sha)
        if base is None:
            # Root commit: compare against the empty tree.
            base = _git_out(slot_dir, "hash-object", "-t", "tree", "/dev/null")
        self.check_diff_allowlist(base, sha, manifest_globs, cwd=slot_dir)

    def check_diff_allowlist(
        self,
        base: str,
        sha: str,
        manifest_globs: Sequence[str],
        *,
        cwd: Path | str | None = None,
    ) -> None:
        """Raise if ``base..sha`` touches paths outside manifest or any symlink.

        Public so an evaluator can gate a candidate SHA against the *run base
        commit* without re-implementing the ``git diff --raw`` parse. Uses
        ``git diff --raw`` which emits old/new file modes; mode ``120000`` is a
        symlink and ``160000`` a gitlink/submodule, both rejected whether they
        appear as the old or new mode (i.e. introduced OR changed). Any path not
        matching ``manifest_globs`` is rejected. ``cwd`` defaults to ``repo_dir``
        (any worktree resolves the same SHAs from the shared object store).
        """
        where = cwd if cwd is not None else self._repo_dir
        # --no-renames so a rename is emitted as a delete + add (two separate
        # single-path lines) rather than one "R<score>\t<old>\t<new>" line —
        # otherwise the second tab smuggles an out-of-manifest destination past
        # the partition("\t") parse below. --no-textconv keeps the raw format.
        raw = _git_out(where, "diff", "--raw", "--no-renames", "--no-textconv", f"{base}..{sha}")
        if not raw:
            return
        # Regular blobs only. 000000 = absent (one side of an add/delete).
        allowed_modes = ("000000", "100644", "100755")
        for line in raw.splitlines():
            # Format: ":<old_mode> <new_mode> <old_sha> <new_sha> <status>\t<path>"
            meta, _, path = line.partition("\t")
            fields = meta.lstrip(":").split()
            if len(fields) < 2:
                continue
            old_mode, new_mode = fields[0], fields[1]
            # Reject symlinks (120000) and gitlinks/submodules (160000) whether
            # introduced or changed — both are out-of-tree side channels into the
            # editable surface.
            if old_mode not in allowed_modes or new_mode not in allowed_modes:
                raise ValueError(
                    f"candidate {sha[:12]} introduces/changes a non-regular file "
                    f"(symlink/gitlink, modes {old_mode}->{new_mode}): {path!r}"
                )
            if not _path_matches_any(path, manifest_globs):
                raise ValueError(
                    f"candidate {sha[:12]} edits out-of-manifest path {path!r} (allowed: {list(manifest_globs)})"
                )

    @staticmethod
    def _first_parent(slot_dir: Path | str, sha: str) -> str | None:
        proc = _run_git(slot_dir, "rev-parse", "--verify", "--quiet", f"{sha}^", check=False)
        if proc.returncode != 0:
            return None
        return proc.stdout.decode().strip() or None

    # ------------------------------------------------------------------
    # Index-only (worktree-less) child commits — pure plumbing
    # ------------------------------------------------------------------

    def commit_index_only(self, parent_sha: str, files: dict[str, bytes | str]) -> str:
        """Create a child commit of ``parent_sha`` overlaying ``files``, no worktree.

        Pure plumbing (hash-object / read-tree-via-update-index / write-tree /
        commit-tree) using a temporary index file so no working tree or the shared
        index is touched. This makes it safe to call from parallel worker threads.
        The resulting commit is pinned under ``refs/gepa/cand/<sha>``.

        ``files`` maps repo-relative paths to blob content. Nested paths are
        supported; ``update-index --add --cacheinfo`` writes them into the index
        which ``write-tree`` materializes (creating subtrees as needed).
        """
        parent = _git_out(self._repo_dir, "rev-parse", "--verify", f"{parent_sha}^{{commit}}")
        parent_tree = _git_out(self._repo_dir, "rev-parse", f"{parent}^{{tree}}")

        # Temp index seeded from the parent tree so existing files are preserved.
        tmp_index = self._repo_dir / ".git" / f"gepa-index-{parent[:12]}-{threading.get_ident()}"
        try:
            self._run_with_index(tmp_index, "read-tree", parent_tree)
            for relpath, content in files.items():
                data = content.encode("utf-8") if isinstance(content, str) else content
                blob = self._hash_object(data)
                self._run_with_index(
                    tmp_index,
                    "update-index",
                    "--add",
                    "--cacheinfo",
                    f"100644,{blob},{relpath}",
                )
            tree_sha = self._git_out_with_index(tmp_index, "write-tree")
        finally:
            tmp_index.unlink(missing_ok=True)

        commit_sha = _git_out(
            self._repo_dir,
            "commit-tree",
            tree_sha,
            "-p",
            parent,
            "-m",
            "gepa index-only candidate",
        )
        self.pin_candidate(commit_sha)
        return commit_sha

    def _hash_object(self, data: bytes) -> str:
        return _run_git(self._repo_dir, "hash-object", "-w", "--stdin", stdin=data).stdout.decode().strip()

    def _run_with_index(self, index_file: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
        cmd = ["git", "-C", str(self._repo_dir), *args]
        env = {**os.environ, "GIT_INDEX_FILE": str(index_file)}
        return subprocess.run(cmd, capture_output=True, check=True, env=env)

    def _git_out_with_index(self, index_file: Path, *args: str) -> str:
        return self._run_with_index(index_file, *args).stdout.decode().strip()

    # ------------------------------------------------------------------
    # Candidate ref pinning + gc control
    # ------------------------------------------------------------------

    def pin_candidate(self, sha: str) -> str:
        """Atomically point ``refs/gepa/cand/<sha>`` at ``sha`` (idempotent)."""
        ref = f"{_CAND_REF_PREFIX}{sha}"
        _run_git(self._repo_dir, "update-ref", ref, sha)
        return ref

    def unpin_candidate(self, sha: str) -> None:
        """Delete a candidate pin ref (best-effort)."""
        _run_git(self._repo_dir, "update-ref", "-d", f"{_CAND_REF_PREFIX}{sha}", check=False)

    def list_candidate_refs(self) -> list[str]:
        """Return all ``refs/gepa/cand/*`` ref names currently present."""
        out = _git_out(self._repo_dir, "for-each-ref", "--format=%(refname)", _CAND_REF_PREFIX)
        return [line for line in out.splitlines() if line]

    def delete_all_candidate_refs(self) -> None:
        """Delete every ``refs/gepa/cand/*`` ref (teardown)."""
        for ref in self.list_candidate_refs():
            _run_git(self._repo_dir, "update-ref", "-d", ref, check=False)

    def get_gc_auto(self) -> str | None:
        """Return the repo-local ``gc.auto`` value, or None if unset."""
        proc = _run_git(self._repo_dir, "config", "--local", "--get", "gc.auto", check=False)
        if proc.returncode != 0:
            return None
        return proc.stdout.decode().strip()

    def set_gc_auto(self, value: str) -> None:
        _run_git(self._repo_dir, "config", "--local", "gc.auto", value)

    def unset_gc_auto(self) -> None:
        _run_git(self._repo_dir, "config", "--local", "--unset", "gc.auto", check=False)


def _path_matches_any(path: str, globs: Sequence[str]) -> bool:
    """True if ``path`` matches any pathspec in ``globs``.

    Matching mirrors ``git add -- <pathspec>`` leading-directory semantics: a
    directory pathspec (``manifest`` or ``manifest/``) matches everything beneath
    it, and ``fnmatch`` handles glob patterns (``manifest/*.txt``). Plain file
    paths match exactly.
    """
    norm = path.strip("/")
    for raw in globs:
        spec = raw.strip("/")
        if not spec:
            continue
        if norm == spec:
            return True
        # Directory-prefix match (everything under `spec/`).
        if norm.startswith(spec + "/"):
            return True
        if fnmatch(norm, spec):
            return True
        # Glob with a trailing recursive component, e.g. `manifest/**`.
        if fnmatch(norm, spec + "/*") or fnmatch(norm, spec.rstrip("*").rstrip("/") + "/*"):
            return True
    return False


class _Slot:
    """Internal pool slot bookkeeping."""

    def __init__(self, slot_dir: Path) -> None:
        self.slot_dir = slot_dir
        self.head: str = ""
        self.refcount: int = 0
        # True while held by an exclusive (writer) lease — no other lease may
        # share this slot until it is released back to refcount 0.
        self.exclusive: bool = False


class GitWorktreePool:
    """K-bound pool of long-lived worktrees with nearest-ancestor lease affinity.

    The portable default :class:`RepoCandidatePool`. Concurrency is bounded by a
    ``Semaphore(K)`` where ``K = len(slot_dirs)``, DECOUPLED from any evaluator
    concurrency. ``lease`` blocks until a slot is free, picks the slot whose HEAD
    is the nearest ancestor of the requested SHA (minimizing
    ``rev-list --count <head>..sha``) for warm parent->child handoff, checks it out
    only if its HEAD differs, and bumps the refcount. A slot with refcount > 0 is
    reused in place and never re-checked-out under a different SHA (no thrash);
    identical SHAs share a slot.

    ``commit_worktree`` delegates to the underlying :class:`GitCheckoutHelper` so
    the pool is a single object exposing the full lease/release/commit_worktree
    contract the engines duck-type.
    """

    def __init__(
        self,
        helper: GitCheckoutHelper,
        slot_dirs: Sequence[Path | str],
        base_commit: str,
    ) -> None:
        if not slot_dirs:
            raise ValueError("GitWorktreePool requires at least one slot dir")
        self._helper = helper
        self._base_commit = base_commit
        self._slots: list[_Slot] = [_Slot(Path(d).resolve()) for d in slot_dirs]
        self._k = len(self._slots)
        self._sem = threading.Semaphore(self._k)
        self._lock = threading.Lock()
        # LRU recency of slot indices (most-recently-released at the end).
        self._lru: OrderedDict[int, None] = OrderedDict()
        self._prev_gc_auto: str | None = None
        self._gc_was_set: bool = False
        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> GitWorktreePool:
        """Create all worktrees at ``base_commit`` and disable gc for the run."""
        self._prev_gc_auto = self._helper.get_gc_auto()
        self._helper.set_gc_auto("0")
        self._gc_was_set = True
        for idx, slot in enumerate(self._slots):
            self._helper.add_worktree(slot.slot_dir, self._base_commit)
            slot.head = self._helper.slot_head(slot.slot_dir)
            self._lru[idx] = None
        self._started = True
        return self

    def teardown(self) -> None:
        """Force-remove run-owned worktrees, drop cand refs, restore gc.auto."""
        for slot in self._slots:
            self._helper.remove_worktree(slot.slot_dir)
        self._helper.delete_all_candidate_refs()
        if self._gc_was_set:
            if self._prev_gc_auto is None:
                self._helper.unset_gc_auto()
            else:
                self._helper.set_gc_auto(self._prev_gc_auto)
            self._gc_was_set = False
        self._started = False

    def __enter__(self) -> GitWorktreePool:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.teardown()

    # ------------------------------------------------------------------
    # Commit minting (delegated to the helper so the pool is one handle)
    # ------------------------------------------------------------------

    def commit_worktree(
        self,
        slot_dir: Path | str,
        message: str,
        manifest_globs: Sequence[str],
    ) -> str:
        """Delegate to the checkout helper so the pool exposes all three of the
        :class:`RepoCandidatePool` methods on one object."""
        return self._helper.commit_worktree(slot_dir, message, manifest_globs)

    # ------------------------------------------------------------------
    # Leasing
    # ------------------------------------------------------------------

    def lease(self, sha: str, *, exclusive: bool = False) -> Lease:
        """Acquire a slot for ``sha`` (blocks on the K-semaphore).

        Selection AND reservation (the refcount bump) happen atomically under
        ``self._lock``, so two concurrent leases can never pick the same free
        slot. The slow ``git checkout`` then runs *outside* the lock with the slot
        already reserved (refcount > 0), so no other lease re-picks it or
        re-checks-it-out under a different SHA — closing the TOCTOU window where a
        slot's working tree could diverge from the SHA it was leased for.

        ``exclusive=True`` reserves a slot that no other lease will share for the
        duration (a writer that edits + commits the working tree). Non-exclusive
        leases (read-only scoring) may share a slot already at ``sha`` but never
        one held exclusively.
        """
        if not self._started:
            raise RuntimeError("GitWorktreePool.lease called before start()")
        self._sem.acquire()
        slot = None
        try:
            with self._lock:
                idx = self._select_slot_locked(sha, exclusive)
                slot = self._slots[idx]
                self._lru.pop(idx, None)
                need_checkout = slot.refcount == 0 and slot.head != sha
                slot.refcount += 1  # reserve before releasing the lock
                if exclusive:
                    slot.exclusive = True
                if need_checkout:
                    # Invalidate the head WHILE the checkout is in flight so no
                    # concurrent lease can exact-match or distance-select this
                    # slot on a now-stale head — its working tree is about to
                    # change on disk. Restored to the real head after checkout.
                    slot.head = ""
            if need_checkout:
                self._helper.checkout(slot.slot_dir, sha)
                new_head = self._helper.slot_head(slot.slot_dir)
                with self._lock:
                    slot.head = new_head
            # Invariant guard: the reserved slot's HEAD is exactly `sha`. The
            # atomic select+reserve makes divergence impossible; this turns any
            # future regression into a loud failure rather than a silent wrong
            # score.
            if slot.head != sha:
                raise RuntimeError(f"GitWorktreePool slot {slot.slot_dir} HEAD {slot.head!r} != leased {sha!r}")
            return Lease(slot.slot_dir, sha)
        except BaseException:
            if slot is not None:
                with self._lock:
                    if slot.refcount > 0:
                        slot.refcount -= 1
                    if slot.refcount == 0:
                        slot.exclusive = False
            self._sem.release()
            raise

    def _select_slot_locked(self, sha: str, exclusive: bool) -> int:
        """Pick a slot index for ``sha``. MUST be called with ``self._lock`` held.

        For a NON-exclusive (read-only) lease: prefer (a) a slot already AT ``sha``
        (distance 0, shareable in place regardless of refcount) UNLESS it is held
        exclusively (a writer is mutating it), then (b) a FREE slot
        (refcount == 0) whose HEAD is the nearest ancestor of ``sha``.

        For an EXCLUSIVE (writer) lease: never share — skip the exact-match reuse
        entirely and only take a FREE slot, so the writer owns the working tree.
        Never returns a busy slot under a different SHA and never round-robins.
        """
        best_idx: int | None = None
        best_dist: int | None = None
        for idx, slot in enumerate(self._slots):
            if not exclusive and slot.head == sha and not slot.exclusive:
                # Exact match: a reader reuses a shareable slot in place regardless
                # of refcount (but never one a writer holds).
                return idx
            if slot.refcount > 0:
                continue
            dist = self._distance(slot.head, sha)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx is not None:
            return best_idx
        # Semaphore(K) admits at most K concurrent leases and there are K slots,
        # each reserved under this lock, so a free or exact-match slot is always
        # available here. Reaching this means slot bookkeeping is corrupt.
        raise RuntimeError("GitWorktreePool: no free slot despite semaphore admission")

    def _distance(self, head: str, sha: str) -> int:
        """Commits on ``head..sha`` (how far ``sha`` is ahead of ``head``).

        A small distance means ``head`` is a near ancestor of ``sha`` so the
        incremental checkout touches few files. If the two are unrelated git still
        returns a count (every commit reachable from ``sha`` but not ``head``); on
        any error we return a large sentinel so the slot is de-prioritized but
        still usable.
        """
        if not head:
            return 1 << 30
        proc = _run_git(self._helper.repo_dir, "rev-list", "--count", f"{head}..{sha}", check=False)
        if proc.returncode != 0:
            return 1 << 30
        try:
            return int(proc.stdout.decode().strip())
        except ValueError:
            return 1 << 30

    def release(self, lease: Lease) -> None:
        """Release a lease: decrement refcount, mark LRU, free the semaphore.

        A slot whose refcount remains > 0 (still held by another lease at the same
        SHA) is NOT re-checked-out by anyone — the next lease for a different SHA
        will pick a different free slot.
        """
        with self._lock:
            for idx, slot in enumerate(self._slots):
                if slot.slot_dir == lease.slot_dir:
                    if slot.refcount > 0:
                        slot.refcount -= 1
                    if slot.refcount == 0:
                        slot.exclusive = False
                        self._lru[idx] = None
                    break
        self._sem.release()
