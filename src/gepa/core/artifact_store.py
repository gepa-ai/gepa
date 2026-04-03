# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Artifact store protocol and implementations for candidate persistence.

The ``ArtifactStore`` is a pluggable persistence layer for candidate metadata,
lineage, and evaluation results.  It is **separate from** ``GEPAState`` (which
tracks the optimisation loop's internal bookkeeping).  ``ArtifactStore`` is
meant for richer, queryable persistence of the full candidate history including
scores, side-info, and arbitrary metadata.

Integration with the engine is via ``ArtifactStoreCallback`` — a
``GEPACallback`` that listens to engine events and forwards them to the store.
The engine itself is unchanged.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class ArtifactStore(Protocol):
    """Pluggable persistence for candidates and evaluation results."""

    def save_candidate(
        self,
        idx: int,
        candidate: dict[str, str],
        *,
        parents: list[int],
        iteration: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist a candidate and its lineage."""
        ...

    def save_evaluation(
        self,
        candidate_idx: int,
        data_id: Any,
        score: float,
        *,
        output: Any = None,
        side_info: dict[str, Any] | None = None,
    ) -> None:
        """Persist a single evaluation result."""
        ...

    def load_candidate(self, idx: int) -> dict[str, str]:
        """Retrieve a candidate by index."""
        ...

    def get_lineage(self, idx: int) -> list[int]:
        """Return the parent indices for a candidate."""
        ...

    def list_candidates(self) -> list[int]:
        """Return all stored candidate indices."""
        ...


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------


class InMemoryArtifactStore:
    """In-memory artifact store — useful for testing and short-lived runs."""

    def __init__(self) -> None:
        self._candidates: dict[int, dict[str, str]] = {}
        self._parents: dict[int, list[int]] = {}
        self._metadata: dict[int, dict[str, Any]] = {}
        self._evaluations: list[dict[str, Any]] = []

    def save_candidate(
        self,
        idx: int,
        candidate: dict[str, str],
        *,
        parents: list[int],
        iteration: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._candidates[idx] = dict(candidate)
        self._parents[idx] = list(parents)
        self._metadata[idx] = {"iteration": iteration, **(metadata or {})}

    def save_evaluation(
        self,
        candidate_idx: int,
        data_id: Any,
        score: float,
        *,
        output: Any = None,
        side_info: dict[str, Any] | None = None,
    ) -> None:
        self._evaluations.append(
            {
                "candidate_idx": candidate_idx,
                "data_id": data_id,
                "score": score,
                "output": output,
                "side_info": side_info,
            }
        )

    def load_candidate(self, idx: int) -> dict[str, str]:
        return dict(self._candidates[idx])

    def get_lineage(self, idx: int) -> list[int]:
        return list(self._parents.get(idx, []))

    def list_candidates(self) -> list[int]:
        return sorted(self._candidates.keys())


class FileSystemArtifactStore:
    """File-system-backed artifact store.

    Layout::

        {root}/
          candidates/
            c00000/
              candidate.json    # candidate dict
              meta.json         # parents, iteration, metadata
              evals/
                {data_id}.json  # per-example evaluation
            c00001/
              ...

    Parameters
    ----------
    root:
        Base directory for all artifacts.
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._candidates_dir = self._root / "candidates"
        self._candidates_dir.mkdir(parents=True, exist_ok=True)

    def _candidate_dir(self, idx: int) -> Path:
        d = self._candidates_dir / f"c{idx:05d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_candidate(
        self,
        idx: int,
        candidate: dict[str, str],
        *,
        parents: list[int],
        iteration: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        d = self._candidate_dir(idx)
        (d / "candidate.json").write_text(json.dumps(candidate, indent=2))
        meta = {"parents": parents, "iteration": iteration, **(metadata or {})}
        (d / "meta.json").write_text(json.dumps(meta, indent=2))

    def save_evaluation(
        self,
        candidate_idx: int,
        data_id: Any,
        score: float,
        *,
        output: Any = None,
        side_info: dict[str, Any] | None = None,
    ) -> None:
        d = self._candidate_dir(candidate_idx) / "evals"
        d.mkdir(parents=True, exist_ok=True)
        entry = {"data_id": data_id, "score": score, "output": output, "side_info": side_info}
        fname = str(data_id).replace("/", "_").replace(" ", "_")
        (d / f"{fname}.json").write_text(json.dumps(entry, indent=2, default=str))

    def load_candidate(self, idx: int) -> dict[str, str]:
        d = self._candidates_dir / f"c{idx:05d}"
        return json.loads((d / "candidate.json").read_text())

    def get_lineage(self, idx: int) -> list[int]:
        d = self._candidates_dir / f"c{idx:05d}"
        meta_path = d / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            return meta.get("parents", [])
        return []

    def list_candidates(self) -> list[int]:
        if not self._candidates_dir.exists():
            return []
        indices = []
        for d in sorted(self._candidates_dir.iterdir()):
            if d.is_dir() and d.name.startswith("c"):
                try:
                    indices.append(int(d.name[1:]))
                except ValueError:
                    pass
        return indices


# ---------------------------------------------------------------------------
# Callback integration
# ---------------------------------------------------------------------------


class ArtifactStoreCallback:
    """GEPACallback that forwards engine events to an ArtifactStore.

    Usage::

        store = FileSystemArtifactStore(root="runs/my_run/artifacts")
        result = optimize_anything(..., callbacks=[ArtifactStoreCallback(store)])
    """

    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    def on_valset_evaluated(self, event: dict[str, Any]) -> None:
        candidate_idx = event["candidate_idx"]
        candidate = event["candidate"]
        parent_ids = list(event.get("parent_ids", []))
        iteration = event.get("iteration", 0)

        self._store.save_candidate(
            idx=candidate_idx,
            candidate=candidate,
            parents=parent_ids,
            iteration=iteration,
        )

        scores_by_val_id = event.get("scores_by_val_id", {})
        for data_id, score in scores_by_val_id.items():
            self._store.save_evaluation(
                candidate_idx=candidate_idx,
                data_id=data_id,
                score=score,
            )
