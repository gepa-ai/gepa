"""
Migration backends for sharing elites between islands.

Supports in-process queue migrations (default) and filesystem-backed migrations
usable across processes or Modal workers sharing a volume.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from turbo_gepa.islands import IslandContext

from turbo_gepa.interfaces import Candidate


def _serialize_candidate(candidate: Candidate) -> dict[str, object]:
    return {
        "text": candidate.text,
        "meta": dict(candidate.meta),
    }


def _deserialize_candidate(payload: dict[str, object]) -> Candidate:
    return Candidate(
        text=str(payload.get("text", "")),
        meta=dict(payload.get("meta", {})),
    )


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


class MigrationBackend(Protocol):
    """Interface for sending/receiving elites across islands."""

    def publish(self, from_island: int, candidates: Sequence[Candidate]) -> None:
        ...

    def consume(self, island_id: int) -> list[Candidate]:
        ...


class NullMigrationBackend(MigrationBackend):
    """No-op backend when migration is disabled."""

    def publish(self, from_island: int, candidates: Sequence[Candidate]) -> None:  # pragma: no cover - trivial
        return

    def consume(self, island_id: int) -> list[Candidate]:  # pragma: no cover - trivial
        return []


@dataclass
class LocalQueueMigrationBackend(MigrationBackend):
    """Wraps IslandContext queues for legacy in-process behavior."""

    context: "IslandContext"

    def publish(self, from_island: int, candidates: Sequence[Candidate]) -> None:
        for candidate in candidates:
            try:
                self.context.outbound.put_nowait(candidate)
            except Exception:
                break

    def consume(self, island_id: int) -> list[Candidate]:
        received: list[Candidate] = []
        while True:
            try:
                received.append(self.context.inbound.get_nowait())
            except Exception:
                break
        return received


class FileMigrationBackend(MigrationBackend):
    """
    Filesystem-backed migration queue. Each island has a JSONL file where other
    workers append elites. Consumers atomically rename the file to avoid races.
    """

    def __init__(self, root_dir: str, total_islands: int) -> None:
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.total_islands = max(1, total_islands)

    def _file_for(self, island_id: int) -> Path:
        return self.root / f"island_{island_id}.jsonl"

    def _tmp_for(self, island_id: int) -> Path:
        return self.root / f".island_{island_id}.{os.getpid()}.tmp"

    def _lock_path(self, target: Path) -> Path:
        return target.with_suffix(target.suffix + ".lock")

    @contextmanager
    def _file_lock(self, target: Path):
        lock_path = self._lock_path(target)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
        try:
            if os.name == "nt":  # pragma: no cover - windows specific
                import msvcrt

                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            else:  # pragma: no cover - unix specific
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            try:
                if os.name == "nt":  # pragma: no cover - windows specific
                    import msvcrt

                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                else:  # pragma: no cover - unix specific
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def publish(self, from_island: int, candidates: Sequence[Candidate]) -> None:
        if not candidates:
            return
        dest = (from_island + 1) % self.total_islands
        island_file = self._file_for(dest)
        island_file.parent.mkdir(parents=True, exist_ok=True)
        with self._file_lock(island_file):
            with island_file.open("a", encoding="utf-8") as handle:
                for cand in candidates:
                    handle.write(json.dumps(_serialize_candidate(cand)) + "\n")

    def consume(self, island_id: int) -> list[Candidate]:
        island_file = self._file_for(island_id)
        if not island_file.exists():
            return []
        tmp = self._tmp_for(island_id)
        with self._file_lock(island_file):
            try:
                island_file.replace(tmp)
            except FileNotFoundError:
                return []
        results: list[Candidate] = []
        try:
            with tmp.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        payload = json.loads(line)
                        results.append(_deserialize_candidate(payload))
                    except json.JSONDecodeError:
                        continue
        finally:
            _safe_unlink(tmp)
        return results
