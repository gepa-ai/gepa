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
    valset_evaluator = MagicMock(return_value=valset_out)

    result = state_mod.initialize_gepa_state(
        run_dir=str(run_dir),
        logger=fake_logger,
        seed_candidate=[seed],
        valset_evaluator=valset_evaluator,
        track_best_outputs=False,
    )

    assert isinstance(result, state_mod.GEPAState)
    assert result.num_full_ds_evals == 1
    assert result.total_num_evals == len(valset_out.scores_by_val_id)
    fake_logger.log.assert_not_called()
    valset_evaluator.assert_called_once_with(seed)

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
    valset_evaluator = MagicMock(return_value=valset_out)

    result = state_mod.initialize_gepa_state(
        run_dir=None,
        logger=fake_logger,
        seed_candidate=[seed],
        valset_evaluator=valset_evaluator,
        track_best_outputs=False,
    )

    assert isinstance(result, state_mod.GEPAState)
    assert result.num_full_ds_evals == 1
    assert result.total_num_evals == len(valset_out.scores_by_val_id)
    fake_logger.log.assert_not_called()
    valset_evaluator.assert_called_once_with(seed)


def test_gepa_state_save_and_initialize(run_dir):
    """With a run dir that contains a saved state, the state is saved and initialized from it."""
    seed = {"model": "m"}
    valset_out = ValsetEvaluation(
        outputs_by_val_id={0: {"x": 1}, 1: {"y": 2}},
        scores_by_val_id={0: 0.3, 1: 0.7},
        objective_scores_by_val_id=None,
    )
    fake_logger = MagicMock()
    valset_evaluator = MagicMock(return_value=valset_out)

    state = state_mod.GEPAState(seed, valset_out)
    state.num_full_ds_evals = 3
    state.total_num_evals = 10
    assert state.is_consistent()

    # Ensure both regular pickle and cloudpickle save and restore equivalent state
    state.save(run_dir)
    result = state_mod.initialize_gepa_state(
        run_dir=str(run_dir),
        logger=fake_logger,
        seed_candidate=[seed],
        valset_evaluator=valset_evaluator,
        track_best_outputs=False,
    )

    assert state.__dict__ == result.__dict__

    state.save(run_dir, use_cloudpickle=True)
    result = state_mod.initialize_gepa_state(
        run_dir=str(run_dir),
        logger=fake_logger,
        seed_candidate=[seed],
        valset_evaluator=valset_evaluator,
        track_best_outputs=False,
    )

    assert state.__dict__ == result.__dict__


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


# =============================================================================
# Multi-seed tests
# =============================================================================


def test_initialize_gepa_state_multi_seed_fresh(run_dir):
    """Multiple seed candidates are evaluated and added to state correctly."""
    seed_a = {"model": "a"}
    seed_b = {"model": "b"}

    valset_out_a = ValsetEvaluation(
        outputs_by_val_id={0: "out_a0", 1: "out_a1"},
        scores_by_val_id={0: 0.1, 1: 0.2},
        objective_scores_by_val_id=None,
    )
    valset_out_b = ValsetEvaluation(
        outputs_by_val_id={0: "out_b0", 1: "out_b1"},
        scores_by_val_id={0: 0.5, 1: 0.6},
        objective_scores_by_val_id=None,
    )

    fake_logger = MagicMock()
    valset_evaluator = MagicMock(side_effect=[valset_out_a, valset_out_b])

    result = state_mod.initialize_gepa_state(
        run_dir=str(run_dir),
        logger=fake_logger,
        seed_candidate=[seed_a, seed_b],
        valset_evaluator=valset_evaluator,
        track_best_outputs=False,
    )

    assert isinstance(result, state_mod.GEPAState)
    assert result.num_seed_candidates == 2
    assert len(result.program_candidates) == 2
    assert result.program_candidates[0] == seed_a
    assert result.program_candidates[1] == seed_b

    # Both seeds evaluated
    assert result.num_full_ds_evals == 2
    expected_total = len(valset_out_a.scores_by_val_id) + len(valset_out_b.scores_by_val_id)
    assert result.total_num_evals == expected_total

    # Both seeds have val subscores
    assert result.prog_candidate_val_subscores[0] == {0: 0.1, 1: 0.2}
    assert result.prog_candidate_val_subscores[1] == {0: 0.5, 1: 0.6}

    # Pareto front updated to reflect better seed b
    assert result.pareto_front_valset[0] == 0.5
    assert result.pareto_front_valset[1] == 0.6

    # Parent tracking: both seeds have [None]
    assert result.parent_program_for_candidate[0] == [None]
    assert result.parent_program_for_candidate[1] == [None]

    assert result.is_consistent()

    # Evaluator called twice (once per seed)
    assert valset_evaluator.call_count == 2
    valset_evaluator.assert_any_call(seed_a)
    valset_evaluator.assert_any_call(seed_b)

    # Seed outputs written to appropriate directories
    base = run_dir / "generated_best_outputs_valset"
    assert (base / "task_0" / "iter_0_prog_0.json").exists()
    seed_out = run_dir / "generated_seed_outputs" / "seed_1"
    assert (seed_out / "task_0" / "iter_0_prog_0.json").exists()
    assert (seed_out / "task_1" / "iter_0_prog_0.json").exists()


def test_initialize_gepa_state_multi_seed_no_run_dir():
    """Multiple seeds work without a run directory."""
    seed_a = {"model": "a"}
    seed_b = {"model": "b"}

    valset_out_a = ValsetEvaluation(
        outputs_by_val_id={0: "out_a"},
        scores_by_val_id={0: 0.3},
        objective_scores_by_val_id=None,
    )
    valset_out_b = ValsetEvaluation(
        outputs_by_val_id={0: "out_b"},
        scores_by_val_id={0: 0.7},
        objective_scores_by_val_id=None,
    )

    fake_logger = MagicMock()
    valset_evaluator = MagicMock(side_effect=[valset_out_a, valset_out_b])

    result = state_mod.initialize_gepa_state(
        run_dir=None,
        logger=fake_logger,
        seed_candidate=[seed_a, seed_b],
        valset_evaluator=valset_evaluator,
        track_best_outputs=False,
    )

    assert result.num_seed_candidates == 2
    assert len(result.program_candidates) == 2
    assert result.total_num_evals == 2
    assert result.is_consistent()


def test_initialize_gepa_state_multi_seed_save_and_load(run_dir):
    """Multi-seed state serializes and deserializes correctly, preserving num_seed_candidates."""
    seed_a = {"model": "a"}
    seed_b = {"model": "b"}

    valset_out_a = ValsetEvaluation(
        outputs_by_val_id={0: "a0"},
        scores_by_val_id={0: 0.2},
        objective_scores_by_val_id=None,
    )
    valset_out_b = ValsetEvaluation(
        outputs_by_val_id={0: "b0"},
        scores_by_val_id={0: 0.8},
        objective_scores_by_val_id=None,
    )

    fake_logger = MagicMock()
    valset_evaluator = MagicMock(side_effect=[valset_out_a, valset_out_b])

    state = state_mod.initialize_gepa_state(
        run_dir=str(run_dir),
        logger=fake_logger,
        seed_candidate=[seed_a, seed_b],
        valset_evaluator=valset_evaluator,
        track_best_outputs=False,
    )
    state.save(str(run_dir))

    loaded = state_mod.GEPAState.load(str(run_dir))
    assert loaded.num_seed_candidates == 2
    assert len(loaded.program_candidates) == 2
    assert loaded.program_candidates[0] == seed_a
    assert loaded.program_candidates[1] == seed_b
    assert loaded.is_consistent()


def test_initialize_gepa_state_multi_seed_mismatched_keys_raises():
    """Seed candidates with different component keys raise a clear error."""
    seed_a = {"model": "a"}
    seed_b = {"instructions": "b"}  # different key

    fake_logger = MagicMock()
    valset_evaluator = MagicMock()

    with pytest.raises(ValueError, match="same component keys"):
        state_mod.initialize_gepa_state(
            run_dir=None,
            logger=fake_logger,
            seed_candidate=[seed_a, seed_b],
            valset_evaluator=valset_evaluator,
            track_best_outputs=False,
        )

    # Evaluator should not have been called at all
    valset_evaluator.assert_not_called()


def test_initialize_gepa_state_empty_seed_list_raises():
    """An empty seed candidate list raises a clear error."""
    fake_logger = MagicMock()
    valset_evaluator = MagicMock()

    with pytest.raises(ValueError, match="non-empty list"):
        state_mod.initialize_gepa_state(
            run_dir=None,
            logger=fake_logger,
            seed_candidate=[],
            valset_evaluator=valset_evaluator,
            track_best_outputs=False,
        )


def test_multi_seed_budget_tracking():
    """Budget (total_num_evals) correctly accounts for all seed evaluations."""
    seeds = [{"model": f"s{i}"} for i in range(3)]

    def make_eval(score):
        return ValsetEvaluation(
            outputs_by_val_id={0: "o", 1: "o", 2: "o"},
            scores_by_val_id={0: score, 1: score, 2: score},
            objective_scores_by_val_id=None,
        )

    valset_evaluator = MagicMock(side_effect=[make_eval(0.1), make_eval(0.5), make_eval(0.9)])
    fake_logger = MagicMock()

    result = state_mod.initialize_gepa_state(
        run_dir=None,
        logger=fake_logger,
        seed_candidate=seeds,
        valset_evaluator=valset_evaluator,
        track_best_outputs=False,
    )

    # 3 seeds x 3 val examples each = 9 total evals
    assert result.total_num_evals == 9
    assert result.num_full_ds_evals == 3
    assert result.num_seed_candidates == 3
    assert len(result.program_candidates) == 3

    # num_metric_calls_by_discovery tracks when each seed was discovered
    assert result.num_metric_calls_by_discovery[0] == 0  # first seed: 0 calls before
    assert result.num_metric_calls_by_discovery[1] == 3  # second seed: 3 calls before
    assert result.num_metric_calls_by_discovery[2] == 6  # third seed: 6 calls before

    assert result.is_consistent()
