import pytest

from gepa.core.result import GEPAResult
from gepa.core.state import GEPAState, ValsetEvaluation


def test_gepa_result_from_dict_upcasts_version0():
    legacy_payload = {
        "candidates": [{"system_prompt": "weight=0"}, {"system_prompt": "weight=1"}],
        "parents": [[None], [0]],
        "val_aggregate_scores": [0.15, 0.35],
        "val_subscores": [[0.1, 0.2], [0.3, 0.4]],
        "per_val_instance_best_candidates": [[0], [1]],
        "discovery_eval_counts": [0, 2],
        "best_outputs_valset": [
            [(0, {"value": 0.1})],
            [(1, {"value": 0.4})],
        ],
        "total_metric_calls": 5,
        "num_full_val_evals": 2,
        "run_dir": "/tmp/gepa",
        "seed": 42,
    }

    result = GEPAResult.from_dict(legacy_payload)

    assert result.val_subscores == [{0: 0.1, 1: 0.2}, {0: 0.3, 1: 0.4}]
    assert result.per_val_instance_best_candidates == {0: {0}, 1: {1}}
    assert result.best_outputs_valset == {
        0: [(0, {"value": 0.1})],
        1: [(1, {"value": 0.4})],
    }

    serialized = result.to_dict()
    assert serialized["validation_schema_version"] == GEPAResult._VALIDATION_SCHEMA_VERSION


def test_to_dict_omits_runtime_oa_fields():
    """eval_log / metadata / eval_server_calls are runtime conveniences; the
    frozen JSON pool snapshot stays byte-stable without them."""
    result = GEPAResult(
        candidates=[{"current_candidate": "x"}],
        parents=[[None]],
        val_aggregate_scores=[0.1],
        val_subscores=[{}],
        per_val_instance_best_candidates={},
        discovery_eval_counts=[1],
        total_metric_calls=4,
        eval_log=[{"score": 0.1}],
        metadata={"engine": "gepa"},
        eval_server_calls=7,
        _str_candidate_key="current_candidate",
    )

    serialized = result.to_dict()
    assert "eval_log" not in serialized
    assert "metadata" not in serialized
    assert "eval_server_calls" not in serialized
    assert serialized["total_metric_calls"] == 4
    assert result.total_evals == 7

    restored = GEPAResult.from_dict(serialized)
    assert restored.eval_log == []
    assert restored.metadata == {}
    assert restored.eval_server_calls is None
    assert restored.total_evals == 4  # falls back to total_metric_calls


@pytest.mark.parametrize(
    ("discovery_eval_counts", "expected"),
    [
        ([0, 5, 10], 10),  # sequential discoveries; summing would report 15
        ([0, 5, 5], 5),  # two candidates accepted from the same parallel step
        ([3], 3),
        ([], 0),
    ],
)
def test_total_evals_without_explicit_total_uses_last_discovery_checkpoint(discovery_eval_counts, expected):
    """discovery_eval_counts holds running totals, one checkpoint per candidate.
    Without total_metric_calls the largest checkpoint is a lower bound on the
    run total; adding the checkpoints counts earlier evaluations repeatedly."""
    num_candidates = len(discovery_eval_counts)
    snapshot = {
        "validation_schema_version": GEPAResult._VALIDATION_SCHEMA_VERSION,
        "candidates": [{"p": str(i)} for i in range(num_candidates)],
        "parents": [[None]] + [[i - 1] for i in range(1, num_candidates)],
        "val_aggregate_scores": [float(i) for i in range(num_candidates)],
        "val_subscores": [{0: float(i)} for i in range(num_candidates)],
        "per_val_instance_best_candidates": {0: [num_candidates - 1]} if num_candidates else {},
        "discovery_eval_counts": discovery_eval_counts,
    }

    restored = GEPAResult.from_dict(snapshot)
    assert restored.total_metric_calls is None
    assert restored.total_evals == expected

    # An explicit total still wins over the checkpoints.
    assert GEPAResult.from_dict(dict(snapshot, total_metric_calls=12)).total_evals == 12


def test_total_evals_fallback_never_exceeds_state_total():
    """Round-trip a real state the way the engine fills it (checkpoint taken
    before the candidate's own evaluations are added) and drop the explicit
    total: the fallback must stay a lower bound on the true count."""
    state = GEPAState(
        {"prompt": "seed"},
        ValsetEvaluation(
            outputs_by_val_id={0: "seed output"},
            scores_by_val_id={0: 0.0},
            objective_scores_by_val_id=None,
        ),
    )
    state.total_num_evals = 1  # initialize_gepa_state counts the seed evaluation
    for step, evals_for_candidate in enumerate([5, 5, 2], start=1):
        checkpoint = state.total_num_evals
        state.increment_evals(evals_for_candidate)
        state.update_state_with_new_program(
            [step - 1],
            {"prompt": f"candidate {step}"},
            ValsetEvaluation(
                outputs_by_val_id={0: f"output {step}"},
                scores_by_val_id={0: float(step)},
                objective_scores_by_val_id=None,
            ),
            run_dir=None,
            num_metric_calls_by_discovery_of_new_program=checkpoint,
            iteration_id=f"step-{step}",
        )

    serialized = GEPAResult.from_state(state).to_dict()
    assert serialized["total_metric_calls"] == state.total_num_evals
    serialized.pop("total_metric_calls")

    restored = GEPAResult.from_dict(serialized)
    assert restored.total_evals == max(state.num_metric_calls_by_discovery)
    assert restored.total_evals <= state.total_num_evals


@pytest.mark.parametrize("new_score", [0.5, 1.0])
def test_from_state_keeps_best_outputs_snapshot_when_state_advances(new_score):
    state = GEPAState(
        {"prompt": "seed"},
        ValsetEvaluation(
            outputs_by_val_id={0: "seed output"},
            scores_by_val_id={0: 0.5},
            objective_scores_by_val_id=None,
        ),
        track_best_outputs=True,
    )
    result = GEPAResult.from_state(state)

    state.update_state_with_new_program(
        [0],
        {"prompt": "next"},
        ValsetEvaluation(
            outputs_by_val_id={0: "next output"},
            scores_by_val_id={0: new_score},
            objective_scores_by_val_id=None,
        ),
        run_dir=None,
        num_metric_calls_by_discovery_of_new_program=1,
        iteration_id="next",
    )

    assert result.candidates == [{"prompt": "seed"}]
    assert result.best_outputs_valset == {0: [(0, "seed output")]}
    expected = [(1, "next output")]
    if new_score == 0.5:
        expected.insert(0, (0, "seed output"))
    assert state.best_outputs_valset == {0: expected}


@pytest.mark.parametrize("track_best_outputs", [False, True])
def test_from_state_preserves_disabled_and_empty_best_outputs(track_best_outputs):
    state = GEPAState(
        {"prompt": "seed"},
        ValsetEvaluation(
            outputs_by_val_id={},
            scores_by_val_id={0: 0.5},
            objective_scores_by_val_id=None,
        ),
        track_best_outputs=track_best_outputs,
    )

    result = GEPAResult.from_state(state)

    assert result.best_outputs_valset == ({} if track_best_outputs else None)


@pytest.mark.parametrize("mutate_result", [False, True])
@pytest.mark.parametrize("field", ["candidates", "parents"])
def test_from_state_detaches_candidate_and_parent_rows(mutate_result, field):
    state = GEPAState(
        {"prompt": "seed"},
        ValsetEvaluation(
            outputs_by_val_id={0: "seed output"},
            scores_by_val_id={0: 0.5},
            objective_scores_by_val_id=None,
        ),
    )
    state.update_state_with_new_program(
        [0],
        {"prompt": "next"},
        ValsetEvaluation(
            outputs_by_val_id={0: "next output"},
            scores_by_val_id={0: 0.75},
            objective_scores_by_val_id=None,
        ),
        run_dir=None,
        num_metric_calls_by_discovery_of_new_program=1,
        iteration_id="next",
    )
    result = GEPAResult.from_state(state)

    if field == "candidates":
        source = result.candidates if mutate_result else state.program_candidates
        source[0]["prompt"] = "edited seed"
        source[1]["prompt"] = "edited next"
        untouched = state.program_candidates if mutate_result else result.candidates
        assert untouched == [{"prompt": "seed"}, {"prompt": "next"}]
    else:
        source = result.parents if mutate_result else state.parent_program_for_candidate
        source[0].append(1)
        source[1][0] = None
        untouched = state.parent_program_for_candidate if mutate_result else result.parents
        assert untouched == [[None], [0]]


def test_string_candidate_snapshot_keeps_original_best_text():
    state = GEPAState(
        {"current_candidate": "seed"},
        ValsetEvaluation(
            outputs_by_val_id={},
            scores_by_val_id={0: 0.5},
            objective_scores_by_val_id=None,
        ),
    )
    result = GEPAResult.from_state(state, str_candidate_key="current_candidate")

    state.program_candidates[0]["current_candidate"] = "edited"

    assert result.best_candidate == "seed"
    assert result.to_dict()["candidates"] == [{"current_candidate": "seed"}]
