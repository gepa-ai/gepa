import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import gepa.core.state as state_mod
import gepa


@pytest.fixture
def run_dir(tmp_path):
    os.makedirs(tmp_path / "run")
    return tmp_path / "run"


def test_initialize_gepa_state_fresh_init_writes_and_counts(run_dir):
    """With a run dir but no state, the state is initialized from scratch and the eval output is written to the run dir."""
    seed = {"model": "m"}
    valset_out = (["out0", {"k": "out1"}], [0.1, 0.2])

    fake_logger = MagicMock()
    valset_evaluator = MagicMock(return_value=valset_out)

    result = state_mod.initialize_gepa_state(
        run_dir=str(run_dir),
        logger=fake_logger,
        seed_candidate=seed,
        valset_evaluator=valset_evaluator,
        track_best_outputs=False,
    )

    assert isinstance(result, state_mod.GEPAState)
    assert result.num_full_ds_evals == 1
    assert result.total_num_evals == len(valset_out[1])
    fake_logger.log.assert_not_called()
    valset_evaluator.assert_called_once_with(seed)

    # Files written for each task with outputs (not scores)
    base = run_dir / "generated_best_outputs_valset"
    p0 = base / "task_0" / "iter_0_prog_0.json"
    p1 = base / "task_1" / "iter_0_prog_0.json"
    assert p0.exists() and p1.exists()
    assert json.loads(p0.read_text()) == 0.1
    assert json.loads(p1.read_text()) == 0.2


def test_initialize_gepa_state_no_run_dir():
    """Without a run dir, the state is initialized from scratch and not saved."""
    seed = {"model": "m"}
    valset_out = (["out"], [0.5])
    fake_logger = MagicMock()
    valset_evaluator = MagicMock(return_value=valset_out)

    result = state_mod.initialize_gepa_state(
        run_dir=None,
        logger=fake_logger,
        seed_candidate=seed,
        valset_evaluator=valset_evaluator,
        track_best_outputs=False,
    )

    assert isinstance(result, state_mod.GEPAState)
    assert result.num_full_ds_evals == 1
    assert result.total_num_evals == len(valset_out[1])
    fake_logger.log.assert_not_called()
    valset_evaluator.assert_called_once_with(seed)


def test_gepa_state_save_and_initialize(run_dir):
    """With a run dir that contains a saved state, the state is saved and initialized from it."""
    seed = {"model": "m"}
    valset_out = ([{"x": 1}, {"y": 2}], [0.3, 0.7])
    fake_logger = MagicMock()
    valset_evaluator = MagicMock(return_value=valset_out)

    state = state_mod.GEPAState(seed, valset_out)
    state.num_full_ds_evals = 3
    state.total_num_evals = 10
    assert state.is_consistent()

    state.save(run_dir)
    result = state_mod.initialize_gepa_state(
        run_dir=str(run_dir),
        logger=fake_logger,
        seed_candidate=seed,
        valset_evaluator=valset_evaluator,
        track_best_outputs=False,
    )

    assert state.__dict__ == result.__dict__


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
