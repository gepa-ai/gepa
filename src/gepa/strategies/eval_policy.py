"""Validation evaluation policy protocols and helpers."""

from __future__ import annotations

from abc import abstractmethod
from typing import Protocol, runtime_checkable

from gepa.core.data_loader import DataId, DataInst, DataLoader
from gepa.core.state import GEPAState, ProgramIdx


@runtime_checkable
class EvaluationPolicy(Protocol[DataId, DataInst]):  # type: ignore
    """Strategy for choosing validation ids to evaluate and identifying best programs for validation instances."""

    @abstractmethod
    def get_eval_batch(
        self, loader: DataLoader[DataId, DataInst], state: GEPAState, target_program_idx: ProgramIdx | None = None
    ) -> list[DataId]:
        """Select examples for evaluation for a program"""
        ...

    @abstractmethod
    def get_best_program(self, state: GEPAState) -> ProgramIdx:
        """Return "best" program given all validation results so far across candidates"""
        ...

    @abstractmethod
    def get_valset_score(self, program_idx: ProgramIdx, state: GEPAState) -> float:
        """Return the score of the program on the valset"""
        ...


class FullEvaluationPolicy(EvaluationPolicy[DataId, DataInst]):
    """Policy that evaluates all validation instances every time."""

    def get_eval_batch(
        self, loader: DataLoader[DataId, DataInst], state: GEPAState, target_program_idx: ProgramIdx | None = None
    ) -> list[DataId]:
        """Always return the full ordered list of validation ids."""
        return list(loader.all_ids())

    def get_best_program(self, state: GEPAState) -> ProgramIdx:
        """Pick the program whose evaluated validation scores achieve the highest average."""
        best_idx, best_score, best_coverage = -1, float("-inf"), -1
        for program_idx, scores in enumerate(state.prog_candidate_val_subscores):
            coverage = len(scores)
            avg = sum(scores.values()) / coverage if coverage else float("-inf")
            if avg > best_score or (avg == best_score and coverage > best_coverage):
                best_score = avg
                best_idx = program_idx
                best_coverage = coverage
        return best_idx

    def get_valset_score(self, program_idx: ProgramIdx, state: GEPAState) -> float:
        """Return the score of the program on the valset"""
        return state.get_program_average_val_subset(program_idx)[0]


class RotatingFoldEvaluationPolicy(FullEvaluationPolicy[DataId, DataInst]):
    """Evaluate each candidate on a single fold of the valset, rotating through
    ``num_folds`` folds across iterations (rotating k-fold cross-validation).

    Each iteration's candidates are scored on one fold only, so a full valset
    pass costs ``1/num_folds`` of the metric calls. Folds advance when a
    candidate is *accepted* (the engine appends an entry to
    ``state.prog_candidate_val_subscores``), which is what makes the policy
    safe for the engine's per-program calls: every ``get_eval_batch`` call
    within one iteration sees the same state length, so **all programs
    evaluated in the same iteration land on the same fold**.

    Two properties to keep in mind when opting in:

    - **Fold bias.** Candidates accepted in different iterations are compared
      on different folds, so their averages are not measured on a common
      benchmark. ``get_best_program``/``get_valset_score`` reuse the full
      policy's coverage-aware semantics (higher average wins; ties prefer the
      wider val coverage), and ``full_eval_every`` periodically re-anchors the
      comparison on the full valset.
    - **Opt-in tool, not an overfitting fix.** The point is to save metric
      calls during training, mirroring the discussion in #133; the default
      ``FullEvaluationPolicy`` remains unchanged.

    The policy is stateless: the fold index is derived entirely from
    ``len(state.prog_candidate_val_subscores)``, so interrupted runs resume
    into the correct fold.
    """

    def __init__(self, num_folds: int = 4, full_eval_every: int | None = None):
        if num_folds < 1:
            raise ValueError("num_folds must be a positive integer")
        if full_eval_every is not None and full_eval_every < 1:
            raise ValueError("full_eval_every must be a positive integer when set")
        self.num_folds = num_folds
        self.full_eval_every = full_eval_every

    def get_eval_batch(
        self, loader: DataLoader[DataId, DataInst], state: GEPAState, target_program_idx: ProgramIdx | None = None
    ) -> list[DataId]:
        """Return the ids of the current fold, or all ids on full-eval anchors.

        The current fold is ``fold_index = (candidates in state - 1) %
        num_folds``, computed from the loader's stable id order, so all
        programs of an iteration share it.
        """
        all_ids = list(loader.all_ids())
        eval_count = len(state.prog_candidate_val_subscores) - 1
        if self.full_eval_every is not None and eval_count % self.full_eval_every == 0:
            return all_ids
        return self._fold_ids(all_ids, eval_count % self.num_folds)

    def get_seed_eval_batch(self, loader: DataLoader[DataId, DataInst]) -> list[DataId]:
        """Give the seed candidate a full-valset baseline via the optional hook.

        The seed is the common ancestor of every candidate; scoring it on the
        whole valset keeps the coverage-aware comparison meaningful.
        """
        return list(loader.all_ids())

    def _fold_ids(self, all_ids: list[DataId], fold_index: int) -> list[DataId]:
        """Slice ``fold_index`` out of ``num_folds`` near-equal, order-preserving parts."""
        base, remainder = divmod(len(all_ids), self.num_folds)
        start = fold_index * base + min(fold_index, remainder)
        size = base + (1 if fold_index < remainder else 0)
        return all_ids[start : start + size]


__all__ = [
    "DataLoader",
    "EvaluationPolicy",
    "FullEvaluationPolicy",
    "RotatingFoldEvaluationPolicy",
]
