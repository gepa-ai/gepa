import json
import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import gepa
import gepa.core.state as state_mod
from gepa.core.adapter import EvaluationBatch
from gepa.core.state import ValsetEvaluation
from gepa.strategies.eval_policy import EvaluationPolicy


@pytest.fixture
def run_dir(tmp_path):
    os.makedirs(tmp_path / "run")
    return tmp_path / "run"


def test_initialize_gepa_state_fresh_init_writes_and_counts(run_dir):
    """With a run dir but no state, the state is initialized from scratch and the eval output is written to the run dir."""
    seed = {"model": "m"}
    valset_out = ValsetEvaluation(
        outputs_by_val_id={0: "out0", 1: {"k": "out1"}},
        scores_by_val_id={0: 0.1, 1: 0.2},
        objective_scores_by_val_id=None,
    )

    fake_logger = MagicMock()

    result = state_mod.initialize_gepa_state(
        run_dir=str(run_dir),
        logger=fake_logger,
        seed_candidate=seed,
        seed_valset_evaluation=valset_out,
        track_best_outputs=False,
    )

    assert isinstance(result, state_mod.GEPAState)
    assert result.num_full_ds_evals == 1
    assert result.total_num_evals == len(valset_out.scores_by_val_id)
    fake_logger.log.assert_not_called()

    # Files written for each task with outputs (not scores)
    base = run_dir / "generated_best_outputs_valset"
    p0 = base / "task_0" / "iter_0_prog_0.json"
    p1 = base / "task_1" / "iter_0_prog_0.json"
    assert p0.exists() and p1.exists()
    assert json.loads(p0.read_text()) == "out0"
    assert json.loads(p1.read_text()) == {"k": "out1"}


def test_initialize_gepa_state_no_run_dir():
    """Without a run dir, the state is initialized from scratch and not saved."""
    seed = {"model": "m"}
    valset_out = ValsetEvaluation(
        outputs_by_val_id={0: "out"},
        scores_by_val_id={0: 0.5},
        objective_scores_by_val_id=None,
    )
    fake_logger = MagicMock()

    result = state_mod.initialize_gepa_state(
        run_dir=None,
        logger=fake_logger,
        seed_candidate=seed,
        seed_valset_evaluation=valset_out,
        track_best_outputs=False,
    )

    assert isinstance(result, state_mod.GEPAState)
    assert result.num_full_ds_evals == 1
    assert result.total_num_evals == len(valset_out.scores_by_val_id)
    fake_logger.log.assert_not_called()


def test_gepa_state_save_and_initialize(run_dir):
    """With a run dir that contains a saved state, the state is saved and initialized from it."""
    seed = {"model": "m"}
    valset_out = ValsetEvaluation(
        outputs_by_val_id={0: {"x": 1}, 1: {"y": 2}},
        scores_by_val_id={0: 0.3, 1: 0.7},
        objective_scores_by_val_id=None,
    )
    fake_logger = MagicMock()

    state = state_mod.GEPAState(seed, valset_out)
    state.num_full_ds_evals = 3
    state.total_num_evals = 10
    assert state.is_consistent()

    # Ensure both regular pickle and cloudpickle save and restore equivalent state
    state.save(run_dir)
    result = state_mod.initialize_gepa_state(
        run_dir=str(run_dir),
        logger=fake_logger,
        seed_candidate=seed,
        seed_valset_evaluation=valset_out,
        track_best_outputs=False,
    )

    assert state.__dict__ == result.__dict__

    state.save(run_dir, use_cloudpickle=True)
    result = state_mod.initialize_gepa_state(
        run_dir=str(run_dir),
        logger=fake_logger,
        seed_candidate=seed,
        seed_valset_evaluation=valset_out,
        track_best_outputs=False,
    )

    assert state.__dict__ == result.__dict__


def test_agent_state_directory_layout(run_dir):
    """save(write_agent_state=True) writes a directory tree of small files."""
    seed = {"system_prompt": "You are helpful."}
    valset_out = ValsetEvaluation(
        outputs_by_val_id={0: "out0", 1: "out1", 2: "out2"},
        scores_by_val_id={0: 0.3, 1: 0.7, 2: 0.5},
        objective_scores_by_val_id={
            0: {"accuracy": 0.3, "latency": 0.9},
            1: {"accuracy": 0.7, "latency": 0.8},
            2: {"accuracy": 0.5, "latency": 0.7},
        },
    )

    state = state_mod.GEPAState(seed, valset_out, frontier_type="hybrid")
    state.total_num_evals = 10
    state.num_full_ds_evals = 1
    state.save(run_dir, write_agent_state=True)

    run_dir_path = Path(str(run_dir))

    # Top-level index is the small navigation file.
    with open(run_dir_path / "gepa_state.json") as f:
        index = json.load(f)
    assert index["schema_version"] == 2
    assert index["frontier_type"] == "hybrid"
    assert index["iteration"] == state.i
    assert index["component_names"] == ["system_prompt"]
    assert index["summary"]["num_candidates"] == 1
    assert index["summary"]["best_candidate_idx"] == 0
    assert index["summary"]["hardest_examples"][0]["best_score"] == 0.3
    assert "candidates_dir" in index["layout"]
    # Bulky payloads must not live in the index.
    assert "candidates" not in index
    assert "iteration_log" not in index
    assert "rejected_proposals" not in index

    # Candidate subtree.
    cand_meta_path = run_dir_path / "candidates" / "00000" / "meta.json"
    assert cand_meta_path.exists()
    with open(cand_meta_path) as f:
        meta = json.load(f)
    assert meta["idx"] == 0
    assert meta["num_val_scored"] == 3
    assert meta["parent_ids"] == []
    assert meta["avg_val_score"] == pytest.approx(0.5)

    with open(run_dir_path / "candidates" / "00000" / "val_scores.json") as f:
        val_scores = json.load(f)
    assert val_scores == {"0": 0.3, "1": 0.7, "2": 0.5}

    # Component text lives as raw .txt, mapped from real name via _index.json.
    with open(run_dir_path / "candidates" / "00000" / "components" / "_index.json") as f:
        comp_index = json.load(f)
    assert comp_index == {"system_prompt": "system_prompt.txt"}
    comp_path = run_dir_path / "candidates" / "00000" / "components" / "system_prompt.txt"
    assert comp_path.read_text() == "You are helpful."

    # Pareto subtree — only files matching frontier_type should exist.
    assert (run_dir_path / "pareto" / "instance_front.json").exists()
    assert (run_dir_path / "pareto" / "objective_front.json").exists()
    assert not (run_dir_path / "pareto" / "cartesian_front.json").exists()


def test_agent_state_default_off(run_dir):
    """save() without write_agent_state=True does not create the agent tree."""
    seed = {"system_prompt": "hi"}
    valset_out = ValsetEvaluation(
        outputs_by_val_id={0: "out"},
        scores_by_val_id={0: 0.5},
        objective_scores_by_val_id=None,
    )
    state = state_mod.GEPAState(seed, valset_out)
    state.total_num_evals = 1
    state.num_full_ds_evals = 1
    state.save(run_dir)  # default: write_agent_state=False

    run_dir_path = Path(str(run_dir))
    assert not (run_dir_path / "gepa_state.json").exists()
    assert not (run_dir_path / "candidates").exists()
    assert not (run_dir_path / "pareto").exists()
    # Legacy outputs still produced.
    assert (run_dir_path / "gepa_state.bin").exists()


def test_agent_state_cache_and_rejected_proposals(run_dir):
    """eval_cache/ and rejected_proposals/ get their own per-item files."""
    seed = {"prompt": "hello"}
    valset_out = ValsetEvaluation(
        outputs_by_val_id={0: "out0", 1: "out1"},
        scores_by_val_id={0: 0.5, 1: 0.5},
        objective_scores_by_val_id=None,
    )
    state = state_mod.GEPAState(seed, valset_out)
    state.total_num_evals = 10
    state.num_full_ds_evals = 1

    state.evaluation_cache = state_mod.EvaluationCache()
    state.evaluation_cache.put({"prompt": "hello"}, 0, "cached_output_0", 0.5)
    state.evaluation_cache.put({"prompt": "hello"}, 1, "cached_output_1", 0.5)

    state.full_program_trace.append({
        "i": 7,
        "selected_program_candidate": 0,
        "subsample_ids": [0],
        "subsample_scores": [0.5],
        "new_subsample_scores": [0.3],
        "proposal_accepted": False,
        "proposed_candidate": {"prompt": "rejected attempt"},
        "eval_before": {"scores": [0.5], "trajectories": ["trace_before"]},
        "eval_after": {"scores": [0.3], "trajectories": ["trace_after"]},
    })

    state.save(run_dir, write_agent_state=True)

    run_dir_path = Path(str(run_dir))

    # Eval cache: one file per candidate, keyed by val_id inside.
    cache_file = run_dir_path / "eval_cache" / "00000.json"
    assert cache_file.exists()
    with open(cache_file) as f:
        cache = json.load(f)
    assert cache["0"]["score"] == 0.5
    assert cache["0"]["output"] == "cached_output_0"
    assert cache["1"]["output"] == "cached_output_1"

    # Rejected proposal: one file per rejecting iteration.
    rej_file = run_dir_path / "rejected_proposals" / "00007.json"
    assert rej_file.exists()
    with open(rej_file) as f:
        rej = json.load(f)
    assert rej["iteration"] == 7
    assert rej["candidate"] == {"prompt": "rejected attempt"}
    assert rej["parent_ids"] == [0]
    assert rej["subsample_scores_before"] == [0.5]
    assert rej["subsample_scores_after"] == [0.3]
    assert rej["eval_before"]["trajectories"] == ["trace_before"]
    assert rej["eval_after"]["trajectories"] == ["trace_after"]

    # Iteration trace file for the same iteration also written.
    iter_file = run_dir_path / "iterations" / "00007.json"
    assert iter_file.exists()
    with open(iter_file) as f:
        iter_entry = json.load(f)
    assert iter_entry["i"] == 7
    assert iter_entry["proposal_accepted"] is False


def test_agent_state_sanitizes_component_names(run_dir):
    """Component names with unsafe filesystem chars get sanitized; _index.json preserves the real name."""
    seed = {"sys/prompt v1": "hello", "sys prompt": "world"}
    valset_out = ValsetEvaluation(
        outputs_by_val_id={0: "out"},
        scores_by_val_id={0: 0.5},
        objective_scores_by_val_id=None,
    )
    state = state_mod.GEPAState(seed, valset_out)
    state.total_num_evals = 1
    state.num_full_ds_evals = 1
    state.save(run_dir, write_agent_state=True)

    comp_dir = Path(str(run_dir)) / "candidates" / "00000" / "components"
    with open(comp_dir / "_index.json") as f:
        index = json.load(f)
    # Real names preserved as keys; slashes and spaces become underscores in filenames.
    assert set(index.keys()) == {"sys/prompt v1", "sys prompt"}
    for real_name, fname in index.items():
        assert "/" not in fname
        assert " " not in fname
        assert (comp_dir / fname).read_text() == seed[real_name]


def test_budget_hooks_excluded_from_serialization(run_dir):
    """Budget hooks are runtime-only and should not be serialized."""
    seed = {"model": "m"}
    valset_out = ValsetEvaluation(
        outputs_by_val_id={0: {"x": 1}, 1: {"y": 2}},
        scores_by_val_id={0: 0.3, 1: 0.7},
        objective_scores_by_val_id=None,
    )

    state = state_mod.GEPAState(seed, valset_out)
    state.num_full_ds_evals = 3
    state.total_num_evals = 10

    # Register a budget hook
    hook_calls = []
    state.add_budget_hook(lambda total, delta: hook_calls.append((total, delta)))

    # Verify hook works
    state.increment_evals(5)
    assert hook_calls == [(15, 5)]
    assert state.total_num_evals == 15

    # Save state (should not include _budget_hooks)
    state.save(run_dir)

    # Load state
    loaded_state = state_mod.GEPAState.load(run_dir)

    # Loaded state should not have _budget_hooks attribute
    assert not hasattr(loaded_state, "_budget_hooks")

    # But increment_evals should still work (no hooks to call)
    loaded_state.increment_evals(3)
    assert loaded_state.total_num_evals == 18

    # And we can add hooks to the loaded state
    loaded_hook_calls = []
    loaded_state.add_budget_hook(lambda total, delta: loaded_hook_calls.append((total, delta)))
    loaded_state.increment_evals(2)
    assert loaded_hook_calls == [(20, 2)]


def test_dynamic_validation(run_dir, rng):
    trainset = [{"id": i, "difficulty": i + 2} for i in range(3)]
    valset_initial = [{"id": i, "difficulty": i + 2} for i in range(2)]
    seed_candidate = {"system_prompt": "weight=0"}

    class DummyAdapter:
        def __init__(self):
            self.propose_new_texts = self._propose_new_texts

        def evaluate(self, batch, candidate, capture_traces=False):
            weight = int(candidate["system_prompt"].split("=")[-1])
            outputs = [{"id": item["id"], "weight": weight} for item in batch]
            scores = [min(1.0, (weight + 1) / (item["difficulty"])) for item in batch]
            trajectories = [{"score": score} for score in scores] if capture_traces else None
            return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

        def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
            records = [{"score": score} for score in eval_batch.scores]
            return dict.fromkeys(components_to_update, records)

        def _propose_new_texts(self, candidate, reflective_dataset, components_to_update):
            weight = int(candidate["system_prompt"].split("=")[-1])
            return dict.fromkeys(components_to_update, f"weight={weight + 1}")

    adapter = DummyAdapter()

    # initially only validate on first example
    class InitValidationPolicy(EvaluationPolicy):
        def get_eval_batch(self, loader, state, target_program_idx=None):
            return [0]

        def is_evaluation_sparse(self) -> bool:
            return False

        def get_best_program(self, state: state_mod.GEPAState) -> state_mod.ProgramIdx:
            return 0

        def get_valset_score(self, program_idx: state_mod.ProgramIdx, state: state_mod.GEPAState) -> float:
            return state.get_program_average_val_subset(program_idx)[0]

    init_validation_policy = InitValidationPolicy()
    gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset_initial,
        adapter=adapter,
        reflection_lm=None,
        max_metric_calls=6,
        run_dir=run_dir,
        val_evaluation_policy=init_validation_policy,
    )

    state_phase_one = state_mod.GEPAState.load(str(run_dir))
    assert len(state_phase_one.program_candidates) >= 2
    assert 0 in state_phase_one.prog_candidate_val_subscores[-1]
    assert 1 not in state_phase_one.prog_candidate_val_subscores[-1]
    assert state_phase_one.valset_evaluations.keys() == {0, 1}

    extended_valset = valset_initial + [{"id": 2, "difficulty": 4}]

    valset_ids = set(range(len(extended_valset)))

    class BackfillValidationPolicy(EvaluationPolicy):
        def get_eval_batch(self, loader, state, target_program_idx=None) -> list[int]:
            missing_valset_ids = valset_ids.difference(state.valset_evaluations.keys())
            if missing_valset_ids:
                return sorted(list(missing_valset_ids))
            return rng.sample(valset_ids, 1)

        def get_best_program(self, state: state_mod.GEPAState) -> state_mod.ProgramIdx:
            return 0

        def is_evaluation_sparse(self) -> bool:
            return False

        def get_valset_score(self, program_idx: state_mod.ProgramIdx, state: state_mod.GEPAState) -> float:
            return state.get_program_average_val_subset(program_idx)[0]

    best_stage1_candidate_idx = init_validation_policy.get_best_program(state_phase_one)
    best_stage1_candidate = state_phase_one.program_candidates[best_stage1_candidate_idx]
    gepa.optimize(
        seed_candidate=best_stage1_candidate,
        trainset=trainset,
        valset=extended_valset,
        adapter=adapter,
        reflection_lm=None,
        max_metric_calls=10,
        run_dir=run_dir,
        val_evaluation_policy=BackfillValidationPolicy(),
    )

    resumed_state = state_mod.GEPAState.load(str(run_dir))
    assert resumed_state.valset_evaluations.keys() == valset_ids
    assert set(resumed_state.prog_candidate_val_subscores[0].keys()) == {0, 1}
    covered_ids = set().union(*[scores.keys() for scores in resumed_state.prog_candidate_val_subscores])
    assert covered_ids == {0, 1, 2}


@pytest.fixture
def legacy_run_dir(tmp_path: Path) -> Path:
    legacy_run_dir = tmp_path / "legacy_run"
    legacy_run_dir.mkdir(parents=True, exist_ok=True)
    legacy_resource_path = Path(__file__).parent / "legacy_test_state.bin"
    shutil.copy2(legacy_resource_path, legacy_run_dir / "gepa_state.bin")
    return legacy_run_dir


def test_load_legacy_state(legacy_run_dir):
    """Ensure legacy gepa_state.bin files migrate correctly when loaded."""
    state = state_mod.GEPAState.load(str(legacy_run_dir))

    assert isinstance(state.prog_candidate_val_subscores, list)
    assert all(isinstance(scores, dict) for scores in state.prog_candidate_val_subscores)
    assert state.validation_schema_version == state_mod.GEPAState._VALIDATION_SCHEMA_VERSION
    assert state.valset_evaluations.keys() == set(range(45))


# ---------------------------------------------------------------------------
# adapter_state tests
# ---------------------------------------------------------------------------


def test_adapter_state_defaults_to_empty_dict():
    """Fresh GEPAState has adapter_state = {}."""
    state = state_mod.GEPAState(
        {"prompt": "p"},
        ValsetEvaluation(outputs_by_val_id={0: "out"}, scores_by_val_id={0: 0.5}, objective_scores_by_val_id=None),
    )
    assert state.adapter_state == {}


def test_adapter_state_save_and_load(run_dir):
    """adapter_state round-trips through save/load."""
    state = state_mod.GEPAState(
        {"prompt": "p"},
        ValsetEvaluation(outputs_by_val_id={0: "out"}, scores_by_val_id={0: 0.5}, objective_scores_by_val_id=None),
    )
    state.num_full_ds_evals = 1
    state.total_num_evals = 1
    state.adapter_state = {"key": "value", "nested": {"a": [1, 2, 3]}}

    state.save(str(run_dir))
    loaded = state_mod.GEPAState.load(str(run_dir))

    assert loaded.adapter_state == {"key": "value", "nested": {"a": [1, 2, 3]}}


def test_upgrade_state_dict_adds_adapter_state():
    """Migration adds adapter_state={} when missing (v4 → v5)."""
    d = {
        "program_candidates": [{"a": "b"}],
        "prog_candidate_objective_scores": [{}],
        "objective_pareto_front": {},
        "program_at_pareto_front_objectives": {},
        "frontier_type": "instance",
        "pareto_front_cartesian": {},
        "program_at_pareto_front_cartesian": {},
        "evaluation_cache": None,
    }
    state_mod.GEPAState._upgrade_state_dict(d)
    assert d["adapter_state"] == {}
    assert d["validation_schema_version"] == state_mod.GEPAState._VALIDATION_SCHEMA_VERSION


def test_existing_adapters_lack_adapter_state_methods():
    """DefaultAdapter doesn't define get/set_adapter_state — duck typing is safe."""
    from gepa.adapters.default_adapter.default_adapter import DefaultAdapter

    assert getattr(DefaultAdapter, "get_adapter_state", None) is None
    assert getattr(DefaultAdapter, "set_adapter_state", None) is None


def test_engine_sync_noop_for_adapter_without_methods(run_dir):
    """Engine sync helpers are no-ops for adapters that lack get/set_adapter_state."""
    state = state_mod.GEPAState(
        {"prompt": "p"},
        ValsetEvaluation(outputs_by_val_id={0: "out"}, scores_by_val_id={0: 0.5}, objective_scores_by_val_id=None),
    )
    state.num_full_ds_evals = 1
    state.total_num_evals = 1

    class PlainAdapter:
        pass

    adapter = PlainAdapter()

    # Simulate engine sync helpers
    getter = getattr(adapter, "get_adapter_state", None)
    setter = getattr(adapter, "set_adapter_state", None)

    assert getter is None
    assert setter is None
    # adapter_state stays at default; no exceptions raised
    assert state.adapter_state == {}


def test_engine_sync_round_trips_adapter_state(run_dir):
    """Engine sync correctly saves and restores adapter state across save/load."""

    class StatefulAdapter:
        def __init__(self):
            self._data: dict = {}

        def get_adapter_state(self) -> dict:
            return dict(self._data)

        def set_adapter_state(self, state: dict) -> None:
            self._data = dict(state)

    # First run: adapter has data, engine syncs to state before save
    adapter = StatefulAdapter()
    adapter._data = {"prompt_abc": ("generated code", 0.95)}

    state = state_mod.GEPAState(
        {"prompt": "p"},
        ValsetEvaluation(outputs_by_val_id={0: "out"}, scores_by_val_id={0: 0.5}, objective_scores_by_val_id=None),
    )
    state.num_full_ds_evals = 1
    state.total_num_evals = 1

    # Simulate _sync_adapter_state_to_state
    getter = getattr(adapter, "get_adapter_state", None)
    assert getter is not None
    state.adapter_state = getter()

    state.save(str(run_dir))

    # Resume: load state, new adapter starts empty, engine syncs from state
    loaded = state_mod.GEPAState.load(str(run_dir))
    new_adapter = StatefulAdapter()
    assert new_adapter._data == {}

    # Simulate _sync_state_to_adapter
    setter = getattr(new_adapter, "set_adapter_state", None)
    assert setter is not None
    setter(loaded.adapter_state)

    assert new_adapter._data == {"prompt_abc": ("generated code", 0.95)}


@pytest.fixture(scope="module")
def recorder_dir() -> Path:
    """Use the cached mocked LLM aime prompt optimization"""
    RECORDER_DIR = Path(__file__).parent / "test_aime_prompt_optimization"
    RECORDER_DIR.mkdir(parents=True, exist_ok=True)
    return RECORDER_DIR


def test_e2e_resume_run(mocked_lms, run_dir):
    """E2E tests for resuming a previous run from a run_dir."""
    import gepa
    from gepa.adapters.default_adapter.default_adapter import DefaultAdapter

    # 1. Setup: Unpack fixtures and load data
    task_lm, reflection_lm = mocked_lms
    adapter = DefaultAdapter(model=task_lm)
    trainset, valset, _ = gepa.examples.aime.init_dataset()
    trainset = trainset[:10]
    valset = valset[:10]  # [3:8]
    seed_prompt = {
        "system_prompt": "You are a helpful assistant. You are given a question and you need to answer it. The answer should be given at the end of your response in exactly the format '### <final answer>'"
    }

    first_run = gepa.optimize(
        seed_candidate=seed_prompt,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        max_metric_calls=30,
        reflection_lm=reflection_lm,
        display_progress_bar=True,
        run_dir=run_dir,
    )

    # Resume from the same run_dir. Even if called with `max_metric_calls=0`,
    # the result should have `total_metric_calls` equal to the amount from the previous run.
    second_run = gepa.optimize(
        seed_candidate=seed_prompt,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        max_metric_calls=0,
        reflection_lm=reflection_lm,
        display_progress_bar=True,
        run_dir=run_dir,
    )
    assert second_run.total_metric_calls == first_run.total_metric_calls
