import json

import pytest

from glean_gepa.runner import _load_seed_candidate


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
