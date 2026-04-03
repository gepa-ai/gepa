# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for gepa.core.workspace — WorkspaceManager protocol and implementations."""

from __future__ import annotations

from pathlib import Path

from gepa.core.workspace import DirectoryCopyWorkspaceManager, NullWorkspaceManager, WorkspaceManager


class TestNullWorkspaceManager:
    def test_protocol_compliance(self) -> None:
        manager = NullWorkspaceManager()
        assert isinstance(manager, WorkspaceManager)

    def test_create_from_seed(self) -> None:
        manager = NullWorkspaceManager()
        ref = manager.create_from_seed("/some/path")
        assert ref == "null"

    def test_fork(self) -> None:
        manager = NullWorkspaceManager()
        ref = manager.fork("parent", "child")
        assert "child" in ref

    def test_get_path(self) -> None:
        manager = NullWorkspaceManager()
        path = manager.get_path("anything")
        assert isinstance(path, Path)

    def test_snapshot(self) -> None:
        manager = NullWorkspaceManager()
        assert manager.snapshot("ref") == "null"

    def test_diff(self) -> None:
        manager = NullWorkspaceManager()
        assert manager.diff("a", "b") == ""

    def test_cleanup_noop(self) -> None:
        manager = NullWorkspaceManager()
        manager.cleanup("ref")  # should not raise


class TestDirectoryCopyWorkspaceManager:
    def test_protocol_compliance(self) -> None:
        manager = DirectoryCopyWorkspaceManager()
        assert isinstance(manager, WorkspaceManager)

    def test_create_and_fork(self, tmp_path: Path) -> None:
        seed_dir = tmp_path / "seed"
        seed_dir.mkdir()
        (seed_dir / "main.py").write_text("print('hello')")

        manager = DirectoryCopyWorkspaceManager(base_dir=tmp_path / "workspaces")
        seed_ref = manager.create_from_seed(seed_dir)
        assert seed_ref == "seed"
        assert manager.get_path(seed_ref) == seed_dir

        child_ref = manager.fork(seed_ref, "c001")
        child_path = manager.get_path(child_ref)
        assert child_path.exists()
        assert (child_path / "main.py").read_text() == "print('hello')"

    def test_fork_is_isolated(self, tmp_path: Path) -> None:
        seed_dir = tmp_path / "seed"
        seed_dir.mkdir()
        (seed_dir / "file.txt").write_text("original")

        manager = DirectoryCopyWorkspaceManager(base_dir=tmp_path / "ws")
        seed_ref = manager.create_from_seed(seed_dir)
        child_ref = manager.fork(seed_ref, "child")
        child_path = manager.get_path(child_ref)

        # Modify child — should not affect seed
        (child_path / "file.txt").write_text("modified")
        assert (seed_dir / "file.txt").read_text() == "original"

    def test_cleanup_removes_directory(self, tmp_path: Path) -> None:
        seed_dir = tmp_path / "seed"
        seed_dir.mkdir()
        (seed_dir / "f.txt").write_text("x")

        manager = DirectoryCopyWorkspaceManager(base_dir=tmp_path / "ws")
        seed_ref = manager.create_from_seed(seed_dir)
        child_ref = manager.fork(seed_ref, "to_delete")
        child_path = manager.get_path(child_ref)
        assert child_path.exists()

        manager.cleanup(child_ref)
        assert not child_path.exists()

    def test_multiple_forks(self, tmp_path: Path) -> None:
        seed_dir = tmp_path / "seed"
        seed_dir.mkdir()
        (seed_dir / "f.txt").write_text("base")

        manager = DirectoryCopyWorkspaceManager(base_dir=tmp_path / "ws")
        seed_ref = manager.create_from_seed(seed_dir)
        ref1 = manager.fork(seed_ref, "a")
        ref2 = manager.fork(seed_ref, "b")

        assert manager.get_path(ref1) != manager.get_path(ref2)
        assert (manager.get_path(ref1) / "f.txt").read_text() == "base"
        assert (manager.get_path(ref2) / "f.txt").read_text() == "base"
