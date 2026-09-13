from gepa.core.result import GEPAResult


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
