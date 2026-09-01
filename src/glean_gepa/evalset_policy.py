"""Training-eval scheduling for Glean's evolutionary proposer."""

from __future__ import annotations

from typing import Any

from gepa.core.data_loader import DataLoader


class UnseenEvalSetPolicy:
    """Select one previously unseen training eval-set item per generation.

    Validation remains the engine's full-evaluation policy: every configured
    validation version is evaluated for each accepted candidate.
    """

    def __init__(self) -> None:
        self._ordered_ids: list[Any] | None = None
        self._next_index = 0

    def _ids(self, loader: DataLoader) -> list[Any]:
        ids = list(loader.all_ids())
        if self._ordered_ids is None:
            self._ordered_ids = ids
        elif ids != self._ordered_ids:
            raise ValueError("UnseenEvalSetPolicy must be shared with loaders containing the same eval sets")
        return self._ordered_ids

    def take_unseen(self, loader: DataLoader, *, purpose: str) -> list[Any]:
        ids = self._ids(loader)
        if self._next_index >= len(ids):
            raise RuntimeError(
                f"No unseen eval sets remain for {purpose}; configured {len(ids)} eval-set versions"
            )
        selected = ids[self._next_index]
        self._next_index += 1
        print(f"[Eval set schedule] {purpose}: selected id {selected} ({self._next_index}/{len(ids)})")
        return [selected]
