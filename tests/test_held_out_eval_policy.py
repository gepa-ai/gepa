"""Tests for HeldOutSetEvaluationPolicy and the held-out evaluation flow."""

from unittest.mock import patch

import pytest

import gepa
from gepa.core.adapter import EvaluationBatch
from gepa.core.data_loader import ListDataLoader
from gepa.core.state import GEPAState, HeldOutEvaluation, ValsetEvaluation
from gepa.strategies.eval_policy import FullEvaluationPolicy, HeldOutSetEvaluationPolicy

# ---------------------------------------------------------------------------
# Minimal fake state for unit tests
# ---------------------------------------------------------------------------


class _FakeState:
    """Minimal GEPAState stub for policy unit tests."""

    def __init__(
        self,
        val_subscores: list[dict],
        held_out_subscores: list[dict],
        program_candidates: list[dict],
    ):
        self.prog_candidate_val_subscores = val_subscores
        self.prog_candidate_held_out_subscores = held_out_subscores
        self.program_candidates = program_candidates

    def get_program_average_val_subset(self, idx: int) -> tuple[float, int]:
        scores = self.prog_candidate_val_subscores[idx]
        if not scores:
            return float("-inf"), 0
        vals = list(scores.values())
        return sum(vals) / len(vals), len(vals)


# ---------------------------------------------------------------------------
# Unit tests for HeldOutSetEvaluationPolicy
# ---------------------------------------------------------------------------


def test_get_best_program_falls_back_to_valset_when_no_held_out_scores():
    """Returns valset leader when no held-out evaluations have happened yet."""
    policy = HeldOutSetEvaluationPolicy()
    state = _FakeState(
        val_subscores=[{0: 0.5, 1: 0.7}, {0: 0.9, 1: 0.8}],
        held_out_subscores=[{}, {}],
        program_candidates=[{"p": "seed"}, {"p": "v1"}],
    )
    # Candidate 1 has higher valset average (0.85 vs 0.6)
    assert policy.get_best_program(state) == 1  # type: ignore[arg-type]


def test_get_best_program_uses_held_out_scores_when_available():
    """Returns the candidate with the best held-out score."""
    policy = HeldOutSetEvaluationPolicy()
    state = _FakeState(
        val_subscores=[{0: 0.9}, {0: 0.5}],
        # candidate 0 leads valset but candidate 1 leads held-out
        held_out_subscores=[{0: 0.4, 1: 0.5}, {0: 0.8, 1: 0.9}],
        program_candidates=[{"p": "seed"}, {"p": "v1"}],
    )
    assert policy.get_best_program(state) == 1  # type: ignore[arg-type]


def test_get_best_program_ignores_unevaluated_candidates_for_held_out():
    """Only candidates with held-out scores compete for held-out best."""
    policy = HeldOutSetEvaluationPolicy()
    state = _FakeState(
        val_subscores=[{0: 0.3}, {0: 0.9}, {0: 0.8}],
        # only candidate 0 has been evaluated on held-out
        held_out_subscores=[{0: 0.7}, {}, {}],
        program_candidates=[{"p": "seed"}, {"p": "v1"}, {"p": "v2"}],
    )
    assert policy.get_best_program(state) == 0  # type: ignore[arg-type]


def test_get_valset_leader_ignores_held_out_scores():
    """get_valset_leader always uses valset scores regardless of held-out."""
    policy = HeldOutSetEvaluationPolicy()
    state = _FakeState(
        val_subscores=[{0: 0.3}, {0: 0.9}],
        held_out_subscores=[{0: 0.99}, {}],  # seed has great held-out, but candidate 1 leads valset
        program_candidates=[{"p": "seed"}, {"p": "v1"}],
    )
    assert policy.get_valset_leader(state) == 1  # type: ignore[arg-type]


def test_get_valset_score_returns_valset_average():
    """get_valset_score delegates to state.get_program_average_val_subset."""
    policy = HeldOutSetEvaluationPolicy()
    state = _FakeState(
        val_subscores=[{0: 0.6, 1: 0.8}],
        held_out_subscores=[{}],
        program_candidates=[{"p": "seed"}],
    )
    assert policy.get_valset_score(0, state) == 0.7  # type: ignore[arg-type]


def test_get_eval_batch_returns_all_val_ids():
    """get_eval_batch returns the full ordered list of validation ids."""
    policy = HeldOutSetEvaluationPolicy()
    loader = ListDataLoader([{"id": 0}, {"id": 1}, {"id": 2}])
    state = _FakeState(val_subscores=[], held_out_subscores=[], program_candidates=[])
    assert policy.get_eval_batch(loader, state) == [0, 1, 2]  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Integration tests using gepa.optimize()
# ---------------------------------------------------------------------------


class _DummyAdapter:
    """Deterministic adapter: score = min(1.0, weight / difficulty)."""

    def __init__(self, val_scores_fn):
        self.val_scores_fn = val_scores_fn
        self.propose_new_texts = self._propose

    def evaluate(self, batch, candidate, capture_traces=False):
        weight = int(candidate["weight"])
        scores = [self.val_scores_fn(item, weight) for item in batch]
        trajectories = [{"score": s} for s in scores] if capture_traces else None
        return EvaluationBatch(outputs=scores, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        return dict.fromkeys(components_to_update, [{"score": s} for s in eval_batch.scores])

    def _propose(self, candidate, reflective_dataset, components_to_update):
        weight = int(candidate["weight"]) + 1
        return dict.fromkeys(components_to_update, str(weight))


def test_held_out_eval_policy_integration(tmp_path):
    """End-to-end: held-out evaluations happen and result uses held-out best."""
    trainset = [{"id": 0, "difficulty": 2}, {"id": 1, "difficulty": 3}]
    valset = [{"id": 0, "difficulty": 2}, {"id": 1, "difficulty": 3}]
    held_out_data = [{"id": 0, "difficulty": 4}, {"id": 1, "difficulty": 5}]

    def val_score(item, weight):
        return min(1.0, weight / item["difficulty"])

    adapter = _DummyAdapter(val_scores_fn=val_score)

    result = gepa.optimize(
        seed_candidate={"weight": "0"},
        trainset=trainset,
        valset=valset,
        held_out=held_out_data,
        adapter=adapter,  # type: ignore[arg-type]
        val_evaluation_policy="heldout_eval",
        max_metric_calls=15,
        run_dir=str(tmp_path / "run"),
    )

    # Held-out scores must be populated
    assert result.held_out_scores is not None
    assert len(result.held_out_scores) > 0

    # num_held_out_evals must be tracked separately
    assert result.num_held_out_evals is not None
    assert result.num_held_out_evals > 0

    # best_idx reflects held-out best
    best = result.best_idx
    assert best in result.held_out_scores

    # valset_best_idx reflects valset best
    assert result.valset_best_idx == max(
        range(len(result.val_aggregate_scores)),
        key=lambda i: result.val_aggregate_scores[i],
    )


def test_held_out_auto_policy_selection(tmp_path):
    """When held_out is provided without explicit policy, HeldOutSetEvaluationPolicy is used."""
    trainset = [{"id": 0, "difficulty": 2}]
    valset = [{"id": 0, "difficulty": 2}]
    held_out_data = [{"id": 0, "difficulty": 3}]

    def val_score(item, weight):
        return min(1.0, weight / item["difficulty"])

    adapter = _DummyAdapter(val_scores_fn=val_score)
    tracker = _RecordingTracker()

    with patch("gepa.api.create_experiment_tracker", return_value=tracker):
        result = gepa.optimize(
            seed_candidate={"weight": "0"},
            trainset=trainset,
            valset=valset,
            held_out=held_out_data,
            adapter=adapter,  # type: ignore[arg-type]
            max_metric_calls=5,
        )

    assert tracker.config is not None
    assert tracker.config["val_evaluation_policy"] == "HeldOutSetEvaluationPolicy"
    assert result.held_out_scores is not None


def test_held_out_string_policy_selection(tmp_path):
    """The explicit 'heldout_eval' string selects HeldOutSetEvaluationPolicy."""
    trainset = [{"id": 0, "difficulty": 2}]
    valset = [{"id": 0, "difficulty": 2}]
    held_out_data = [{"id": 0, "difficulty": 3}]

    def val_score(item, weight):
        return min(1.0, weight / item["difficulty"])

    adapter = _DummyAdapter(val_scores_fn=val_score)

    result = gepa.optimize(
        seed_candidate={"weight": "0"},
        trainset=trainset,
        valset=valset,
        held_out=held_out_data,
        adapter=adapter,  # type: ignore[arg-type]
        max_metric_calls=5,
        val_evaluation_policy="heldout_eval",
    )

    assert result.held_out_scores is not None


def test_heldout_eval_without_held_out_warns(tmp_path):
    """Selecting heldout_eval without a held_out set should warn explicitly."""
    trainset = [{"id": 0, "difficulty": 2}]
    valset = [{"id": 0, "difficulty": 2}]

    def val_score(item, weight):
        return min(1.0, weight / item["difficulty"])

    adapter = _DummyAdapter(val_scores_fn=val_score)

    with pytest.warns(UserWarning, match="no held_out set was provided"):
        gepa.optimize(
            seed_candidate={"weight": "0"},
            trainset=trainset,
            valset=valset,
            adapter=adapter,  # type: ignore[arg-type]
            max_metric_calls=5,
            val_evaluation_policy="heldout_eval",
        )


def test_held_out_evals_do_not_consume_budget():
    """num_held_out_evals is tracked separately and does not count toward the budget."""
    # With max_metric_calls=1 the optimization stops after seeding (1 valset eval).
    # But the seed's held-out eval should still have happened.
    trainset = [{"id": 0, "difficulty": 2}]
    valset = [{"id": 0, "difficulty": 2}]
    held_out_data = [{"id": 0, "difficulty": 3}, {"id": 1, "difficulty": 4}]

    def val_score(item, weight):
        return min(1.0, weight / item["difficulty"])

    adapter = _DummyAdapter(val_scores_fn=val_score)

    result = gepa.optimize(
        seed_candidate={"weight": "0"},
        trainset=trainset,
        valset=valset,
        held_out=held_out_data,
        adapter=adapter,  # type: ignore[arg-type]
        val_evaluation_policy="heldout_eval",
        max_metric_calls=1,
    )

    # Budget counter only counts valset + trainset evals
    assert result.total_metric_calls == 1

    # Held-out evals are tracked separately: seed was evaluated on 2 held-out examples
    assert result.num_held_out_evals == 2

    # total_metric_calls does NOT include held-out
    assert result.total_metric_calls != result.total_metric_calls + result.num_held_out_evals


def test_held_out_only_evaluated_for_valset_leader(tmp_path):
    """Candidates that never lead the valset are never evaluated on held-out."""
    # Proposer always increments weight. Seed (weight=0) scores 0 on valset.
    # First candidate (weight=1) becomes valset leader and gets held-out eval.
    # If a second candidate (weight=2) also leads, it gets a held-out eval too.
    # Candidates that never lead the valset should have no held-out scores.
    trainset = [{"id": 0, "difficulty": 2}, {"id": 1, "difficulty": 3}]
    valset = [{"id": 0, "difficulty": 2}, {"id": 1, "difficulty": 3}]
    held_out_data = [{"id": 0, "difficulty": 10}]

    def val_score(item, weight):
        return min(1.0, weight / item["difficulty"])

    adapter = _DummyAdapter(val_scores_fn=val_score)

    result = gepa.optimize(
        seed_candidate={"weight": "0"},
        trainset=trainset,
        valset=valset,
        held_out=held_out_data,
        adapter=adapter,  # type: ignore[arg-type]
        val_evaluation_policy="heldout_eval",
        max_metric_calls=20,
        run_dir=str(tmp_path / "run"),
    )

    assert result.held_out_scores is not None
    # Every candidate in held_out_scores was a valset leader at some point
    # and every valset leader must have a held-out score
    valset_scores = result.val_aggregate_scores
    valset_leader = max(range(len(valset_scores)), key=lambda i: valset_scores[i])
    assert valset_leader in result.held_out_scores

    # The seed (weight=0) scores 0 on valset — if it was immediately displaced,
    # it should still be evaluated on held-out as the initial leader
    assert 0 in result.held_out_scores


def test_get_program_average_held_out_returns_inf_when_unevaluated():
    """get_program_average_held_out returns (-inf, 0) for candidates with no held-out scores."""
    from gepa.core.state import ValsetEvaluation

    state = GEPAState(
        {"p": "seed"},
        ValsetEvaluation(outputs_by_val_id={0: "o"}, scores_by_val_id={0: 0.5}),
    )
    avg, count = state.get_program_average_held_out(0)
    assert avg == float("-inf")
    assert count == 0


def test_empty_held_out_loader_skips_evaluation():
    """Engine handles an empty held-out loader without crashing (exercises the early-return path)."""
    trainset = [{"id": 0, "difficulty": 2}]
    valset = [{"id": 0, "difficulty": 2}]

    def val_score(item, weight):
        return min(1.0, weight / item["difficulty"])

    adapter = _DummyAdapter(val_scores_fn=val_score)

    result = gepa.optimize(
        seed_candidate={"weight": "0"},
        trainset=trainset,
        valset=valset,
        held_out=[],  # empty loader triggers the early-return in _evaluate_on_held_out_set
        adapter=adapter,  # type: ignore[arg-type]
        val_evaluation_policy="heldout_eval",
        max_metric_calls=5,
    )

    assert result.num_held_out_evals == 0
    assert result.held_out_scores is None


def test_result_to_dict_and_from_dict_roundtrip_with_held_out_scores():
    """to_dict/from_dict round-trip preserves held_out_scores and num_held_out_evals."""
    from gepa.core.result import GEPAResult

    original = GEPAResult(
        candidates=[{"p": "seed"}, {"p": "v1"}],
        parents=[[None], [0]],
        val_aggregate_scores=[0.4, 0.7],
        val_subscores=[{0: 0.4}, {0: 0.7}],
        per_val_instance_best_candidates={0: {1}},
        discovery_eval_counts=[0, 5],
        held_out_scores={0: 0.3, 1: 0.9},
        num_held_out_evals=4,
    )

    restored = GEPAResult.from_dict(original.to_dict())

    assert restored.held_out_scores == {0: 0.3, 1: 0.9}
    assert restored.num_held_out_evals == 4
    # best_idx uses held-out: candidate 1 has score 0.9
    assert restored.best_idx == 1
    # valset_best_idx uses valset: candidate 1 also has 0.7
    assert restored.valset_best_idx == 1


def test_result_best_idx_uses_held_out_over_valset():
    """best_idx picks held-out winner even when valset winner differs."""
    from gepa.core.result import GEPAResult

    result = GEPAResult(
        candidates=[{"p": "seed"}, {"p": "v1"}],
        parents=[[None], [0]],
        val_aggregate_scores=[0.9, 0.4],  # candidate 0 leads valset
        val_subscores=[{0: 0.9}, {0: 0.4}],
        per_val_instance_best_candidates={0: {0}},
        discovery_eval_counts=[0, 5],
        held_out_scores={0: 0.3, 1: 0.8},  # candidate 1 leads held-out
    )

    assert result.best_idx == 1  # held-out winner
    assert result.valset_best_idx == 0  # valset winner


def test_no_held_out_leaves_result_unchanged():
    """Without held_out, result behaves exactly as before (held_out_scores is None)."""
    trainset = [{"id": 0, "difficulty": 2}, {"id": 1, "difficulty": 3}]

    def val_score(item, weight):
        return min(1.0, weight / item["difficulty"])

    adapter = _DummyAdapter(val_scores_fn=val_score)

    result = gepa.optimize(
        seed_candidate={"weight": "0"},
        trainset=trainset,
        adapter=adapter,  # type: ignore[arg-type]
        max_metric_calls=5,
    )

    assert result.held_out_scores is None
    assert result.num_held_out_evals == 0
    # best_idx falls back to valset
    assert result.best_idx == result.valset_best_idx


# ---------------------------------------------------------------------------
# 1. Callback eventing tests
# ---------------------------------------------------------------------------


class _RecordingCallback:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def _record(self, name, event):
        self.calls.append((name, dict(event)))

    def get_calls(self, name):
        return [e for n, e in self.calls if n == name]

    def on_held_out_evaluated(self, event):
        self._record("on_held_out_evaluated", event)

    def on_valset_evaluated(self, event):
        self._record("on_valset_evaluated", event)


class _RecordingTracker:
    def __init__(self):
        self.summary: dict | None = None
        self.config: dict | None = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def log_config(self, config):
        self.config = dict(config)

    def log_metrics(self, metrics, step=None):
        pass

    def log_table(self, name, columns, data):
        pass

    def log_summary(self, summary):
        self.summary = dict(summary)

    def log_html(self, html_content, key):
        pass


def test_engine_emits_held_out_callback_for_seed_at_iteration_0():
    """on_held_out_evaluated is emitted for the seed candidate at iteration 0."""
    trainset = [{"id": 0, "difficulty": 2}]
    valset = [{"id": 0, "difficulty": 2}]
    held_out_data = [{"id": 0, "difficulty": 3}, {"id": 1, "difficulty": 4}]

    def val_score(item, weight):
        return min(1.0, weight / item["difficulty"])

    adapter = _DummyAdapter(val_scores_fn=val_score)
    cb = _RecordingCallback()

    gepa.optimize(
        seed_candidate={"weight": "0"},
        trainset=trainset,
        valset=valset,
        held_out=held_out_data,
        adapter=adapter,  # type: ignore[arg-type]
        val_evaluation_policy="heldout_eval",
        max_metric_calls=1,
        callbacks=[cb],
    )

    held_out_calls = cb.get_calls("on_held_out_evaluated")
    # Seed must always fire at iteration 0
    assert any(c["iteration"] == 0 and c["candidate_idx"] == 0 for c in held_out_calls)

    seed_call = next(c for c in held_out_calls if c["iteration"] == 0)
    assert seed_call["candidate_idx"] == 0
    assert seed_call["num_examples_evaluated"] == len(held_out_data)
    assert seed_call["total_held_out_size"] == len(held_out_data)
    assert set(seed_call["scores_by_id"].keys()) == {0, 1}


def test_engine_emits_held_out_callback_when_valset_leader_changes():
    """on_held_out_evaluated fires for a newly promoted valset leader, not just the seed.

    Seed has weight=0 → valset score 0.0. The first accepted candidate (weight=1)
    scores higher and becomes the new valset leader, which must trigger its own
    held-out callback. We assert that a candidate with idx > 0 appears in the
    held-out callbacks, directly proving the leader-change path fired.
    """
    trainset = [{"id": 0, "difficulty": 2}, {"id": 1, "difficulty": 3}]
    valset = [{"id": 0, "difficulty": 2}, {"id": 1, "difficulty": 3}]
    held_out_data = [{"id": 0, "difficulty": 10}]

    def val_score(item, weight):
        return min(1.0, weight / item["difficulty"])

    adapter = _DummyAdapter(val_scores_fn=val_score)
    cb = _RecordingCallback()

    gepa.optimize(
        seed_candidate={"weight": "0"},
        trainset=trainset,
        valset=valset,
        held_out=held_out_data,
        adapter=adapter,  # type: ignore[arg-type]
        val_evaluation_policy="heldout_eval",
        max_metric_calls=20,
        callbacks=[cb],
    )

    held_out_calls = cb.get_calls("on_held_out_evaluated")
    candidate_ids_evaluated = {c["candidate_idx"] for c in held_out_calls}

    # Seed fires at iteration 0
    assert 0 in candidate_ids_evaluated

    # At least one non-seed candidate must have been evaluated on held-out,
    # proving a new valset leader triggered the callback
    assert any(c > 0 for c in candidate_ids_evaluated), (
        "Expected a non-seed candidate to become valset leader and trigger a held-out callback"
    )

    # All payloads have consistent fields
    for call in held_out_calls:
        assert "scores_by_id" in call
        assert "average_score" in call
        assert "total_held_out_size" in call
        assert call["total_held_out_size"] == len(held_out_data)


def test_on_valset_evaluated_is_best_program_tracks_valset_leader_with_held_out():
    """Held-out winner must not change the valset callback's is_best_program semantics."""
    trainset = [{"id": 0, "kind": "train"}]
    valset = [{"id": 0, "kind": "val"}]
    held_out_data = [{"id": 0, "kind": "held_out"}]

    def score(item, weight):
        if item["kind"] == "held_out":
            return 1.0 / (weight + 1)
        return float(weight)

    adapter = _DummyAdapter(val_scores_fn=score)
    cb = _RecordingCallback()

    gepa.optimize(
        seed_candidate={"weight": "0"},
        trainset=trainset,
        valset=valset,
        held_out=held_out_data,
        adapter=adapter,  # type: ignore[arg-type]
        val_evaluation_policy="heldout_eval",
        max_metric_calls=2,
        callbacks=[cb],
    )

    candidate_call = next(c for c in cb.get_calls("on_valset_evaluated") if c["candidate_idx"] == 1)
    assert candidate_call["average_score"] == 1.0
    assert candidate_call["is_best_program"] is True


def test_final_summary_reports_policy_selected_candidate_scores():
    """Summary should report the policy-selected candidate's valset and held-out scores."""
    trainset = [{"id": 0, "kind": "train"}]
    valset = [{"id": 0, "kind": "val"}]
    held_out_data = [{"id": 0, "kind": "held_out"}]

    def score(item, weight):
        if item["kind"] == "held_out":
            return 1.0 / (weight + 1)
        return float(weight)

    adapter = _DummyAdapter(val_scores_fn=score)
    tracker = _RecordingTracker()

    with patch("gepa.api.create_experiment_tracker", return_value=tracker):
        gepa.optimize(
            seed_candidate={"weight": "0"},
            trainset=trainset,
            valset=valset,
            held_out=held_out_data,
            adapter=adapter,  # type: ignore[arg-type]
            val_evaluation_policy="heldout_eval",
            max_metric_calls=2,
        )

    assert tracker.summary is not None
    assert tracker.summary["best_candidate_idx"] == 0
    assert tracker.summary["best_score_on_valset"] == 0.0
    assert tracker.summary["best_score_on_held_out"] == 1.0


# ---------------------------------------------------------------------------
# 2. Persistence / migration tests
# ---------------------------------------------------------------------------


def test_held_out_state_round_trips_through_save_load(tmp_path):
    """prog_candidate_held_out_subscores and num_held_out_evals survive save/load."""
    import gepa.core.state as state_mod

    seed = {"p": "seed"}
    valset_eval = ValsetEvaluation(
        outputs_by_val_id={0: "o"},
        scores_by_val_id={0: 0.5},
    )
    held_out_eval = HeldOutEvaluation(
        outputs_by_id={0: "h"},
        scores_by_id={0: 0.25, 1: 0.75},
    )

    state = state_mod.GEPAState(seed, valset_eval, seed_held_out_evaluation=held_out_eval)
    state.num_full_ds_evals = 1
    state.total_num_evals = 1
    state.num_held_out_evals = 2

    run_dir = str(tmp_path / "run")
    import os

    os.makedirs(run_dir)
    state.save(run_dir)

    loaded = state_mod.GEPAState.load(run_dir)

    assert loaded.prog_candidate_held_out_subscores == [{0: 0.25, 1: 0.75}]
    assert loaded.num_held_out_evals == 2
    assert loaded.is_consistent()


def test_legacy_state_migration_fills_held_out_defaults(tmp_path):
    """A state saved without held-out fields gets empty defaults on load."""
    import os
    import pickle

    import gepa.core.state as state_mod

    run_dir = str(tmp_path / "run")
    os.makedirs(run_dir)

    # Build a minimal legacy state dict missing the new fields
    seed = {"p": "seed"}
    valset_eval = ValsetEvaluation(outputs_by_val_id={0: "o"}, scores_by_val_id={0: 0.5})
    state = state_mod.GEPAState(seed, valset_eval)
    state.num_full_ds_evals = 1
    state.total_num_evals = 1
    state.num_held_out_evals = 0

    # Manually drop the new fields to simulate a legacy file
    legacy_dict = {
        k: v for k, v in state.__dict__.items() if k not in ("prog_candidate_held_out_subscores", "num_held_out_evals")
    }
    legacy_dict["validation_schema_version"] = 2  # pre-held-out version

    with open(os.path.join(run_dir, "gepa_state.bin"), "wb") as f:
        pickle.dump(legacy_dict, f)

    loaded = state_mod.GEPAState.load(run_dir)

    assert hasattr(loaded, "prog_candidate_held_out_subscores")
    assert loaded.prog_candidate_held_out_subscores == [{}]
    assert hasattr(loaded, "num_held_out_evals")
    assert loaded.num_held_out_evals == 0
    assert loaded.is_consistent()


# ---------------------------------------------------------------------------
# 3. Negative policy test
# ---------------------------------------------------------------------------


def test_incompatible_policy_with_held_out_raises():
    """Passing held_out with a non-HeldOutSetEvaluationPolicy raises at runtime."""
    trainset = [{"id": 0, "difficulty": 2}]
    valset = [{"id": 0, "difficulty": 2}]
    held_out_data = [{"id": 0, "difficulty": 3}]

    def val_score(item, weight):
        return min(1.0, weight / item["difficulty"])

    adapter = _DummyAdapter(val_scores_fn=val_score)

    with pytest.raises(ValueError, match="held_out requires HeldOutSetEvaluationPolicy"):
        gepa.optimize(
            seed_candidate={"weight": "0"},
            trainset=trainset,
            valset=valset,
            held_out=held_out_data,
            adapter=adapter,  # type: ignore[arg-type]
            val_evaluation_policy=FullEvaluationPolicy(),  # incompatible
            max_metric_calls=5,
        )


def test_full_eval_string_with_held_out_raises():
    """Passing held_out with the 'full_eval' string path raises."""
    trainset = [{"id": 0, "difficulty": 2}]
    valset = [{"id": 0, "difficulty": 2}]
    held_out_data = [{"id": 0, "difficulty": 3}]

    def val_score(item, weight):
        return min(1.0, weight / item["difficulty"])

    adapter = _DummyAdapter(val_scores_fn=val_score)

    with pytest.raises(ValueError, match="held_out requires HeldOutSetEvaluationPolicy"):
        gepa.optimize(
            seed_candidate={"weight": "0"},
            trainset=trainset,
            valset=valset,
            held_out=held_out_data,
            adapter=adapter,  # type: ignore[arg-type]
            val_evaluation_policy="full_eval",
            max_metric_calls=5,
        )
