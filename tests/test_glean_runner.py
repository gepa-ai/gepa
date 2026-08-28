import json
from datetime import date

import pytest

from glean_gepa.runner import _load_seed_candidate, _parse_args, _select_recent_train_and_val_versions


def test_load_seed_candidate_accepts_writing_code_only(tmp_path):
    path = tmp_path / "seed.json"
    path.write_text(json.dumps({"WRITING_CODE": "code instructions"}))

    assert _load_seed_candidate(path) == {"WRITING_CODE": "code instructions"}


def test_load_seed_candidate_rejects_legacy_modules(tmp_path):
    path = tmp_path / "seed.json"
    path.write_text(json.dumps({"GLOBAL_ROLE": "role", "WRITING_CODE": "code instructions"}))

    with pytest.raises(SystemExit, match="must contain only 'WRITING_CODE'"):
        _load_seed_candidate(path)


def test_load_seed_candidate_rejects_non_string_writing_code(tmp_path):
    path = tmp_path / "seed.json"
    path.write_text(json.dumps({"WRITING_CODE": ["code instructions"]}))

    with pytest.raises(SystemExit, match="WRITING_CODE must be a string"):
        _load_seed_candidate(path)


def test_parse_args_accepts_all_reflection_samples_and_hamming_k():
    args = _parse_args(
        [
            "--seed_candidate",
            "seed.json",
            "--reflection_samples",
            "all",
            "--reflection_hamming_distance_k",
            "10",
        ]
    )

    assert args.reflection_samples is None
    assert args.reflection_hamming_distance_k == 10


@pytest.mark.parametrize("value", ["0", "-1", "not-a-number"])
def test_parse_args_rejects_invalid_reflection_sample_count(value):
    with pytest.raises(SystemExit):
        _parse_args(["--seed_candidate", "seed.json", "--reflection_samples", value])


def test_recent_versions_are_split_into_incremental_train_and_held_out_val():
    train_versions, val_versions = _select_recent_train_and_val_versions(
        [{"version": "20260813"}, {"version": "20260820"}, {"version": "20260827"}],
        today=date(2026, 8, 27),
        lookback_days=14,
        valset_size=2,
    )

    assert train_versions == ["20260813"]
    assert val_versions == ["20260820", "20260827"]


def test_recent_versions_fall_back_to_one_val_version_when_only_two_are_available():
    train_versions, val_versions = _select_recent_train_and_val_versions(
        [{"version": "20260820"}, {"version": "20260827"}],
        today=date(2026, 8, 27),
        lookback_days=14,
        valset_size=2,
    )

    assert train_versions == ["20260820"]
    assert val_versions == ["20260827"]
