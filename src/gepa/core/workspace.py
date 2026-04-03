# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class WorkspaceManager(Protocol):
    """Manages isolated workspaces for code candidate mutations.

    The agent works in a workspace directory.  The manager handles creating,
    forking, resolving, diffing, and cleaning up workspaces.

    Concrete implementations decide *how* isolation is achieved:

    - ``GitWorktreeWorkspaceManager`` uses git branches + lightweight worktrees.
    - ``DirectoryCopyWorkspaceManager`` uses plain directory copies.
    - ``NullWorkspaceManager`` is a no-op for text-mode.

    **Typical flow**::

        ref = manager.create_from_seed(Path("/path/to/project"))
        child = manager.fork(ref)
        path = manager.get_path(child)        # agent works here
        snap = manager.snapshot(child)         # commit current state
        delta = manager.diff(ref, child)       # what changed
        manager.cleanup(child)                 # discard rejected candidate
    """

    def create_from_seed(self, seed: Path | str) -> str:
        """Initialize a workspace from a seed path or git ref.

        Returns a workspace reference string (e.g. branch name, directory id).
        """
        ...

    def fork(self, parent_ref: str, label: str = "") -> str:
        """Create an isolated workspace copy from *parent_ref*.

        Returns a new workspace reference.
        """
        ...

    def get_path(self, ref: str) -> Path:
        """Resolve a workspace reference to a working directory path.

        This is where the agent reads/writes files.
        """
        ...

    def snapshot(self, ref: str) -> str:
        """Capture the current state of a workspace (e.g. ``git commit``).

        Returns a snapshot identifier (commit SHA, timestamp, etc.).
        """
        ...

    def diff(self, parent_ref: str, child_ref: str) -> str:
        """Return a human-readable diff between two workspace states."""
        ...

    def cleanup(self, ref: str) -> None:
        """Remove a workspace and reclaim resources."""
        ...


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------


class NullWorkspaceManager:
    """No-op workspace manager for text-mode optimization.

    All operations are safe to call but do nothing meaningful.  ``get_path()``
    returns a fixed empty directory so callers that unconditionally resolve
    workspace paths don't crash.
    """

    def create_from_seed(self, seed: Path | str) -> str:
        return "null"

    def fork(self, parent_ref: str, label: str = "") -> str:
        return f"{parent_ref}_fork_{label or 'null'}"

    def get_path(self, ref: str) -> Path:
        return Path(".")

    def snapshot(self, ref: str) -> str:
        return "null"

    def diff(self, parent_ref: str, child_ref: str) -> str:
        return ""

    def cleanup(self, ref: str) -> None:
        pass


class DirectoryCopyWorkspaceManager:
    """Workspace manager that uses plain directory copies.

    Each ``fork()`` creates a full ``shutil.copytree`` of the parent workspace.
    Simple and universal — works for non-git projects.

    Parameters
    ----------
    base_dir:
        Root directory under which all workspaces are created.
        Defaults to a ``gepa_workspaces`` directory alongside the seed.
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir
        self._workspaces: dict[str, Path] = {}

    def _ensure_base_dir(self) -> Path:
        if self._base_dir is None:
            self._base_dir = Path(".gepa_workspaces")
        self._base_dir.mkdir(parents=True, exist_ok=True)
        return self._base_dir

    def create_from_seed(self, seed: Path | str) -> str:
        seed_path = Path(seed)
        ref = "seed"
        self._workspaces[ref] = seed_path
        return ref

    def fork(self, parent_ref: str, label: str = "") -> str:
        parent_path = self._workspaces[parent_ref]
        new_ref = label or f"c{uuid.uuid4().hex[:8]}"
        dest = self._ensure_base_dir() / new_ref
        shutil.copytree(parent_path, dest, dirs_exist_ok=False)
        self._workspaces[new_ref] = dest
        return new_ref

    def get_path(self, ref: str) -> Path:
        return self._workspaces[ref]

    def snapshot(self, ref: str) -> str:
        return ref

    def diff(self, parent_ref: str, child_ref: str) -> str:
        return f"Directory copy diff between {parent_ref} and {child_ref} (not implemented)"

    def cleanup(self, ref: str) -> None:
        path = self._workspaces.pop(ref, None)
        if path is not None and path.exists():
            shutil.rmtree(path, ignore_errors=True)
