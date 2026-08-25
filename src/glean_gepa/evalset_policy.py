"""Evaluation policy that reveals configured eval sets one at a time."""

from __future__ import annotations

from typing import Any

from gepa.core.data_loader import DataLoader
from gepa.core.state import GEPAState, ProgramIdx
from gepa.strategies.eval_policy import FullEvaluationPolicy


class UnseenEvalSetPolicy(FullEvaluationPolicy):
    """Use one previously unseen eval-set item for each full-screen round."""

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

    def get_seed_eval_batch(self, loader: DataLoader) -> list[Any]:
        return self.take_unseen(loader, purpose="root candidate")

    def get_eval_batch(
        self,
        loader: DataLoader,
        state: GEPAState,
        target_program_idx: ProgramIdx | None = None,
    ) -> list[Any]:
        del state, target_program_idx
        return self.take_unseen(loader, purpose="accepted-candidate full screen")

