"""Offline tests for the rotating k-fold val_evaluation_policy (#133)."""

from __future__ import annotations

from itertools import pairwise
from typing import ClassVar

import pytest

import gepa
from gepa.core.data_loader import DataId, ListDataLoader
from gepa.strategies.eval_policy import FullEvaluationPolicy, RotatingFoldEvaluationPolicy


class _FakeState:
    """Just what the policy reads: the per-candidate val subscores list."""

    def __init__(self, num_candidates: int):
        self.prog_candidate_val_subscores: list[dict[DataId, float]] = [{f"id{i}": 1.0} for i in range(num_candidates)]

    def accept_candidate(self) -> None:
        """Simulate the engine appending subscores for an accepted candidate."""
        self.prog_candidate_val_subscores.append({f"id{len(self.prog_candidate_val_subscores)}": 1.0})

    def get_program_average_val_subset(self, program_idx: int) -> tuple[float, int]:
        scores = self.prog_candidate_val_subscores[program_idx]
        if not scores:
            return float("-inf"), 0
        return sum(scores.values()) / len(scores), len(scores)


def _loader(num_items: int) -> ListDataLoader:
    return ListDataLoader([{"id": i, "split": "val"} for i in range(num_items)])


def test_rejects_non_positive_fold_counts() -> None:
    with pytest.raises(ValueError):
        RotatingFoldEvaluationPolicy(num_folds=0)
    with pytest.raises(ValueError):
        RotatingFoldEvaluationPolicy(num_folds=2, full_eval_every=0)


def test_folds_partition_valset_exactly_once() -> None:
    """Folds are disjoint and together cover every id exactly once; repeated
    calls with the same state return the identical (deterministic) batch."""
    policy = RotatingFoldEvaluationPolicy(num_folds=3)
    loader = _loader(10)
    state = _FakeState(num_candidates=1)

    calls = [policy.get_eval_batch(loader, state) for _ in range(3)]
    assert calls[0] == calls[1] == calls[2]

    folds = [policy.get_eval_batch(loader, _FakeState(num_candidates=1 + i)) for i in range(3)]
    flat = [val_id for fold in folds for val_id in fold]
    assert sorted(flat) == list(range(10))
    assert len(flat) == len(set(flat))
    assert all(fold for fold in folds)


def test_uneven_split_gives_remainder_to_leading_folds() -> None:
    policy = RotatingFoldEvaluationPolicy(num_folds=4)
    loader = _loader(10)
    sizes = [len(policy.get_eval_batch(loader, _FakeState(num_candidates=1 + i))) for i in range(4)]
    assert sizes == [3, 3, 2, 2]


def test_same_iteration_all_programs_see_the_same_fold() -> None:
    """Engine invariant (#133): ``get_eval_batch`` runs once per candidate
    program within an iteration; the fold cursor must be keyed on accepted
    candidates (``len(state.prog_candidate_val_subscores)``), never on call
    count — so every call inside one iteration returns the same fold."""
    policy = RotatingFoldEvaluationPolicy(num_folds=2)
    loader = _loader(4)
    state = _FakeState(num_candidates=1)

    per_program_batches = [policy.get_eval_batch(loader, state) for _ in range(4)]
    assert all(batch == per_program_batches[0] for batch in per_program_batches)

    # ... and a fresh fold only appears once a candidate has actually been accepted.
    state.accept_candidate()
    assert policy.get_eval_batch(loader, state) != per_program_batches[0]


def test_rotation_advances_on_acceptance_and_wraps() -> None:
    policy = RotatingFoldEvaluationPolicy(num_folds=3)
    loader = _loader(9)
    state = _FakeState(num_candidates=1)  # seed candidate already in state

    seen = [tuple(policy.get_eval_batch(loader, state))]
    for _ in range(4):
        state.accept_candidate()
        seen.append(tuple(policy.get_eval_batch(loader, state)))

    assert seen[0] == (0, 1, 2)
    assert seen[1] == (3, 4, 5)
    assert seen[2] == (6, 7, 8)
    assert seen[3] == (0, 1, 2)  # wrapped back to the first fold
    assert seen[4] == (3, 4, 5)


def test_full_eval_every_anchors_on_the_full_valset() -> None:
    policy = RotatingFoldEvaluationPolicy(num_folds=2, full_eval_every=2)
    loader = _loader(4)

    batches = [policy.get_eval_batch(loader, _FakeState(num_candidates=1 + i)) for i in range(4)]
    sizes = [len(batch) for batch in batches]
    # eval_count = (candidates in state) - 1: full on 0, fold on 1, full on 2, fold on 3.
    assert sizes == [4, 2, 4, 2]


def test_get_seed_eval_batch_returns_full_valset() -> None:
    policy = RotatingFoldEvaluationPolicy(num_folds=4)
    loader = _loader(6)
    assert list(policy.get_seed_eval_batch(loader)) == list(range(6))


def test_best_program_reuses_full_policy_comparison_semantics() -> None:
    policy = RotatingFoldEvaluationPolicy(num_folds=2)

    class _State:
        prog_candidate_val_subscores: ClassVar[list[dict[str, float]]] = [
            {"a": 0.4, "b": 0.4},  # avg 0.4, coverage 2
            {"a": 0.5},  # avg 0.5, coverage 1
        ]

        def get_program_average_val_subset(self, program_idx: int) -> tuple[float, int]:
            scores = self.prog_candidate_val_subscores[program_idx]
            if not scores:
                return float("-inf"), 0
            return sum(scores.values()) / len(scores), len(scores)

    # Higher average wins even with lower coverage.
    assert policy.get_best_program(_State()) == 1
    assert policy.get_valset_score(1, _State()) == 0.5

    class _TiedState(_State):
        prog_candidate_val_subscores: ClassVar[list[dict[str, float]]] = [
            {"a": 1.0, "b": 0.0},  # avg 0.5, coverage 2
            {"a": 0.5},  # avg 0.5, coverage 1
        ]

    # Equal average: the candidate with the wider val coverage wins.
    assert policy.get_best_program(_TiedState()) == 0
    # Agreement with the full policy on the same scores.
    assert policy.get_best_program(_TiedState()) == FullEvaluationPolicy().get_best_program(_TiedState())


class _CountingAdapter:
    """Deterministic adapter (no LLM) that records which val ids were scored.

    ``evaluate`` scores ``min(1, (weight + 1) / difficulty)`` and
    ``propose_new_texts`` always bumps the integer weight, so the optimization
    trajectory is fully deterministic (constant difficulty + equal-acceptance
    keeps every iteration accepted).
    """

    def __init__(self):
        self.val_ids_seen: list[list[int]] = []
        self.propose_new_texts = self._propose_new_texts

    def evaluate(self, batch: list[dict], candidate: dict[str, str], capture_traces: bool = False):
        from gepa.core.adapter import EvaluationBatch

        if batch and batch[0].get("split") == "val":
            self.val_ids_seen.append(sorted(item["id"] for item in batch))

        weight = int(candidate["system_prompt"].split("=")[-1])
        scores = [min(1.0, (weight + 1) / item["difficulty"]) for item in batch]
        trajectories = [{"score": score} for score in scores] if capture_traces else None
        return EvaluationBatch(outputs=list(batch), scores=scores, trajectories=trajectories)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        records = [{"Score": score} for score in eval_batch.scores]
        return dict.fromkeys(components_to_update, records)

    def _propose_new_texts(self, candidate, reflective_dataset, components_to_update):
        weight = int(candidate["system_prompt"].split("=")[-1])
        return dict.fromkeys(components_to_update, f"weight={weight + 1}")


_TRAINSET = [{"id": 100 + i, "difficulty": 10, "split": "train"} for i in range(3)]
# Constant difficulty keeps every fold average identical across folds, and the
# weight-increment proposals improve strictly for the whole (budget-limited)
# run, so the trajectory is deterministic and every iteration accepts.
_VALSET = [{"id": i, "difficulty": 10, "split": "val"} for i in range(4)]


def _run_optimize(policy, run_name, tmp_path):
    adapter = _CountingAdapter()
    result = gepa.optimize(
        seed_candidate={"system_prompt": "weight=0"},
        trainset=_TRAINSET,
        valset=_VALSET,
        adapter=adapter,
        reflection_lm=None,
        candidate_selection_strategy="current_best",
        max_metric_calls=40,
        run_dir=str(tmp_path / run_name),
        val_evaluation_policy=policy,
    )
    return result, adapter


def test_e2e_rotating_policy_scores_each_candidate_on_a_single_fold(tmp_path) -> None:
    """End-to-end with a mock adapter: every post-seed candidate is evaluated
    on exactly one fold (half the valset here), consecutive candidates rotate
    folds, and the optimizer still selects a best candidate."""
    result, _ = _run_optimize(RotatingFoldEvaluationPolicy(num_folds=2), "rotating", tmp_path)

    # The seed keeps its full-valset baseline.
    assert len(result.val_subscores[0]) == 4
    # Every accepted candidate afterwards was scored on exactly one fold of 2.
    assert len(result.val_subscores) >= 3, "expected several accepted candidates"
    for scores in result.val_subscores[1:]:
        assert len(scores) == 2
    # Rotation: consecutive accepted candidates saw different folds.
    fold_ids = [frozenset(scores) for scores in result.val_subscores[1:]]
    assert all(a != b for a, b in pairwise(fold_ids))
    # The optimizer still produced a best candidate consistent with the
    # coverage-aware comparison (seed avg 0.5, accepted candidates 1.0).
    assert result.best_idx >= 1


def test_e2e_rotating_policy_uses_fewer_val_metric_calls_than_full(tmp_path) -> None:
    """Same deterministic setup, two policies: rotating folds must spend fewer
    val metric calls than full evaluation while still completing."""
    rotating_result, rotating_adapter = _run_optimize(RotatingFoldEvaluationPolicy(num_folds=2), "rotating", tmp_path)
    full_result, full_adapter = _run_optimize(FullEvaluationPolicy(), "full", tmp_path)

    assert rotating_result.val_subscores and full_result.val_subscores
    rotating_val_ids = sum(len(ids) for ids in rotating_adapter.val_ids_seen)
    full_val_ids = sum(len(ids) for ids in full_adapter.val_ids_seen)
    assert rotating_val_ids < full_val_ids


def test_e2e_merge_path_with_rotating_policy_does_not_crash(tmp_path) -> None:
    """``merge_val_overlap_floor`` gating reads parents' (possibly disjoint)
    val coverage under a non-full policy; the merge-enabled path must run to
    completion with rotating folds (including a full_eval_every anchor)."""
    result = gepa.optimize(
        seed_candidate={"system_prompt": "weight=0"},
        trainset=_TRAINSET,
        valset=_VALSET,
        adapter=_CountingAdapter(),
        reflection_lm=None,
        candidate_selection_strategy="current_best",
        max_metric_calls=40,
        use_merge=True,
        max_merge_invocations=2,
        merge_val_overlap_floor=1,
        run_dir=str(tmp_path / "merge-explicit"),
        val_evaluation_policy=RotatingFoldEvaluationPolicy(num_folds=2, full_eval_every=3),
    )
    assert result.val_subscores
    assert result.best_idx >= 0
