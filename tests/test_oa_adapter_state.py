# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for OptimizeAnythingAdapter state persistence via adapter_state."""

import os
import pickle

import pytest

import gepa.core.state as state_mod
from gepa.adapters.optimize_anything_adapter.optimize_anything_adapter import OptimizeAnythingAdapter
from gepa.core.state import ValsetEvaluation
from gepa.optimize_anything import OptimizationState


@pytest.fixture
def run_dir(tmp_path):
    os.makedirs(tmp_path / "run")
    return tmp_path / "run"


def _noop_evaluator(candidate, **kwargs):
    return 0.0, None, {}


def _make_adapter(**kwargs) -> OptimizeAnythingAdapter:
    return OptimizeAnythingAdapter(evaluator=_noop_evaluator, **kwargs)


class TestGetAdapterState:
    """Tests for get_adapter_state()."""

    def test_empty_on_fresh_adapter(self):
        adapter = _make_adapter()
        state = adapter.get_adapter_state()
        assert isinstance(state["opt_state"], OptimizationState)
        assert state["opt_state"].best_evals_by_example == {}

    def test_contains_populated_evals(self):
        adapter = _make_adapter(best_example_evals_k=2)
        # Simulate evaluations
        adapter._update_best_example_evals("ex1", 0.9, {"info": "a"})
        adapter._update_best_example_evals("ex1", 0.7, {"info": "b"})
        adapter._update_best_example_evals("ex2", 0.5, {"info": "c"})

        state = adapter.get_adapter_state()
        opt = state["opt_state"]
        assert len(opt.best_evals_by_example) == 2  # 2 example hashes

        # ex1 hash should have 2 evals (k=2), sorted desc
        ex1_hash = adapter._example_hash("ex1")
        assert len(opt.best_evals_by_example[ex1_hash]) == 2
        assert opt.best_evals_by_example[ex1_hash][0]["score"] == 0.9


class TestSetAdapterState:
    """Tests for set_adapter_state()."""

    def test_restores_from_snapshot(self):
        adapter = _make_adapter()
        adapter._update_best_example_evals("ex1", 0.9, {"info": "a"})

        # Get snapshot
        snapshot = adapter.get_adapter_state()

        # Fresh adapter, restore
        adapter2 = _make_adapter()
        adapter2.set_adapter_state(snapshot)

        ex1_hash = adapter._example_hash("ex1")
        assert adapter2._opt_state.best_evals_by_example[ex1_hash][0]["score"] == 0.9

    def test_empty_state_on_fresh_run(self):
        adapter = _make_adapter()
        adapter.set_adapter_state({})
        assert adapter._opt_state.best_evals_by_example == {}

    def test_invalid_opt_state_resets_to_default(self):
        adapter = _make_adapter()
        adapter._update_best_example_evals("ex1", 0.9, {"info": "a"})
        # Setting with invalid opt_state type resets to default
        adapter.set_adapter_state({"opt_state": "not an OptimizationState"})
        assert adapter._opt_state.best_evals_by_example == {}


class TestRoundTrip:
    """Tests for full get → set → get round-trip."""

    def test_round_trip_preserves_data(self):
        adapter = _make_adapter(best_example_evals_k=3)
        adapter._update_best_example_evals("ex1", 0.9, {"key": "val1"})
        adapter._update_best_example_evals("ex1", 0.7, {"key": "val2"})
        adapter._update_best_example_evals("ex2", 0.5, {"key": "val3"})

        snapshot = adapter.get_adapter_state()

        adapter2 = _make_adapter(best_example_evals_k=3)
        adapter2.set_adapter_state(snapshot)

        snapshot2 = adapter2.get_adapter_state()
        assert (
            snapshot["opt_state"].best_evals_by_example
            == snapshot2["opt_state"].best_evals_by_example
        )

    def test_build_opt_state_after_restore(self):
        """After restoring, _build_opt_state returns populated OptimizationState."""
        adapter = _make_adapter(best_example_evals_k=2)
        adapter._update_best_example_evals("ex1", 0.9, {"info": "best"})
        adapter._update_best_example_evals("ex1", 0.3, {"info": "worst"})

        snapshot = adapter.get_adapter_state()

        adapter2 = _make_adapter(best_example_evals_k=2)
        adapter2.set_adapter_state(snapshot)

        opt_state = adapter2._build_opt_state("ex1")
        assert len(opt_state.best_example_evals) == 2
        assert opt_state.best_example_evals[0]["score"] == 0.9


class TestPickleRoundTrip:
    """Tests that OptimizationState survives pickle (as used by GEPAState)."""

    def test_pickle_round_trip(self):
        opt = OptimizationState(
            best_evals_by_example={"h1": [{"score": 0.9, "side_info": {"k": "v"}}]},
            best_example_evals=[{"score": 0.5, "side_info": {}}],
        )
        restored = pickle.loads(pickle.dumps(opt))
        assert restored.best_evals_by_example == opt.best_evals_by_example
        assert restored.best_example_evals == opt.best_example_evals

    def test_setstate_adds_missing_fields(self):
        """Simulate loading an old pickle that lacks best_evals_by_example."""
        # Create an object, then strip the new field to simulate old pickle
        opt = OptimizationState(best_example_evals=[{"score": 1.0, "side_info": {}}])
        raw = pickle.dumps(opt)

        # Tamper: remove best_evals_by_example from the pickled __dict__
        restored_broken = pickle.loads(raw)
        del restored_broken.__dict__["best_evals_by_example"]

        # Now re-pickle and load — __setstate__ should add the default
        state_dict = restored_broken.__dict__.copy()
        fresh = OptimizationState.__new__(OptimizationState)
        fresh.__setstate__(state_dict)
        assert fresh.best_evals_by_example == {}
        assert fresh.best_example_evals == [{"score": 1.0, "side_info": {}}]


class TestE2EGEPAStateSaveLoad:
    """End-to-end: adapter → GEPAState.save() → GEPAState.load() → adapter restored."""

    def test_adapter_state_survives_gepa_state_save_load(self, run_dir):
        """OptimizationState round-trips through GEPAState pickle on disk."""
        # 1. Adapter accumulates evals
        adapter = _make_adapter(best_example_evals_k=3)
        adapter._update_best_example_evals("ex1", 0.9, {"info": "best"})
        adapter._update_best_example_evals("ex1", 0.5, {"info": "mid"})
        adapter._update_best_example_evals("ex2", 0.7, {"info": "only"})

        # 2. Snapshot adapter state into GEPAState and save
        gepa_state = state_mod.GEPAState(
            {"model": "m"},
            ValsetEvaluation(outputs_by_val_id={0: "out"}, scores_by_val_id={0: 0.5}),
        )
        gepa_state.num_full_ds_evals = 1
        gepa_state.total_num_evals = 1
        gepa_state.adapter_state = adapter.get_adapter_state()
        gepa_state.save(str(run_dir))

        # 3. Load from disk and restore into a fresh adapter
        loaded_state = state_mod.GEPAState.load(str(run_dir))
        adapter2 = _make_adapter(best_example_evals_k=3)
        adapter2.set_adapter_state(loaded_state.adapter_state)

        # 4. Verify evals survived
        ex1_hash = adapter._example_hash("ex1")
        ex2_hash = adapter._example_hash("ex2")
        assert len(adapter2._opt_state.best_evals_by_example[ex1_hash]) == 2
        assert adapter2._opt_state.best_evals_by_example[ex1_hash][0]["score"] == 0.9
        assert adapter2._opt_state.best_evals_by_example[ex2_hash][0]["score"] == 0.7

        # 5. Verify _build_opt_state works after restore
        opt_state = adapter2._build_opt_state("ex1")
        assert opt_state.best_example_evals[0]["score"] == 0.9
