# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for the opt-in admission manifest (per-decision JSONL audit ledger)."""

import json
import os

import pytest

from gepa.core.state import _candidate_hash
from gepa.logging.admission_manifest import (
    SCHEMA_VERSION,
    AdmissionManifestWriter,
    build_decision_record,
    components_changed,
    read_manifest,
    verify_record,
)

# ---------------------------------------------------------------------------
# Unit: components_changed (candidate-vs-parent diff, not metadata)
# ---------------------------------------------------------------------------


def test_components_changed_single_parent():
    assert components_changed({"a": "x", "b": "Y2"}, [{"a": "x", "b": "y"}]) == ["b"]


def test_components_changed_new_component():
    assert components_changed({"a": "x", "b": "new"}, [{"a": "x"}]) == ["b"]


def test_components_changed_two_parents_merge():
    # a taken from p1, b taken from p2 -> each differs from one parent.
    assert components_changed({"a": "1", "b": "2"}, [{"a": "1", "b": "1"}, {"a": "2", "b": "2"}]) == ["a", "b"]


def test_components_changed_no_parents():
    assert components_changed({"a": "x", "b": "y"}, []) == ["a", "b"]


# ---------------------------------------------------------------------------
# Unit: build_decision_record
# ---------------------------------------------------------------------------


def test_build_decision_record_schema_and_hashes():
    parent, cand = {"sys": "old"}, {"sys": "new"}
    rec = build_decision_record(
        iteration=3,
        proposal_kind="reflective_mutation",
        decision="accepted",
        decision_reason="r",
        candidate=cand,
        parent_ids=[0],
        parents=[parent],
        subsample_ids=["t1", "t2"],
        subsample_score_before=1.0,
        subsample_score_after=2.0,
        metric_calls_before_decision=10,
    )
    assert rec["schema_version"] == SCHEMA_VERSION
    assert rec["candidate_hash"] == _candidate_hash(cand)
    assert rec["parent_candidate_hashes"] == [_candidate_hash(parent)]
    assert rec["candidate_idx"] is None
    assert rec["validation"] is None
    assert rec["cache_hits"] is None and rec["cache_misses"] is None
    assert rec["components_changed"] == ["sys"]
    assert rec["decision_basis"] == "subsample_acceptance"
    assert rec["subsample_ids"] == ["t1", "t2"]


# ---------------------------------------------------------------------------
# Unit: writer + read_manifest (incl. truncated final line)
# ---------------------------------------------------------------------------


def test_writer_roundtrip_and_truncated_final_line(tmp_path):
    path = str(tmp_path / "admission_manifest.jsonl")
    w = AdmissionManifestWriter(path)
    w.append_record({"i": 1, "decision": "accepted"})
    w.append_record({"i": 2, "decision": "rejected"})
    # Simulate a crash mid-append: an unterminated, invalid final line.
    with open(path, "a") as f:
        f.write('{"i": 3, "decision": "acce')
    records = read_manifest(path)
    assert [r["i"] for r in records] == [1, 2]


def test_read_manifest_missing_file(tmp_path):
    assert read_manifest(str(tmp_path / "nope.jsonl")) == []


def test_read_manifest_raises_on_nonfinal_corruption(tmp_path):
    path = str(tmp_path / "m.jsonl")
    with open(path, "w") as f:
        f.write("not json\n")
        f.write('{"i": 2}\n')
    with pytest.raises(json.JSONDecodeError):
        read_manifest(path)


# ---------------------------------------------------------------------------
# Unit: verify_record replay (accept / reject / merge + tamper detection)
# ---------------------------------------------------------------------------


def _rec(decision, before, after, cand, parents, kind="reflective_mutation"):
    return build_decision_record(
        iteration=1,
        proposal_kind=kind,
        decision=decision,
        decision_reason="r",
        candidate=cand,
        parent_ids=list(range(len(parents))),
        parents=parents,
        subsample_ids=None,
        subsample_score_before=before,
        subsample_score_after=after,
        metric_calls_before_decision=0,
    )


def test_verify_record_accept_reflective():
    parent, cand = {"a": "x"}, {"a": "y"}
    assert verify_record(_rec("accepted", 1.0, 2.0, cand, [parent]), cand, [parent]) is True


def test_verify_record_reject_reflective():
    parent, cand = {"a": "x"}, {"a": "y"}
    assert verify_record(_rec("rejected", 2.0, 1.0, cand, [parent]), cand, [parent]) is True


def test_verify_record_merge_accept_uses_ge_gate():
    p1, p2, cand = {"a": "1"}, {"a": "2"}, {"a": "3"}
    # merge gate is new_sum >= baseline; equal scores still accept.
    assert verify_record(_rec("accepted", 1.0, 1.0, cand, [p1, p2], kind="merge"), cand, [p1, p2]) is True


def test_verify_record_detects_tampered_hash():
    parent, cand = {"a": "x"}, {"a": "y"}
    rec = _rec("accepted", 1.0, 2.0, cand, [parent])
    rec["candidate_hash"] = "deadbeef"
    assert verify_record(rec, cand, [parent]) is False


def test_verify_record_detects_inconsistent_decision():
    parent, cand = {"a": "x"}, {"a": "y"}
    # after < before but labelled accepted -> not re-derivable -> False.
    assert verify_record(_rec("accepted", 2.0, 1.0, cand, [parent]), cand, [parent]) is False


# ---------------------------------------------------------------------------
# Integration: a real optimize_anything run writes a replayable manifest
# ---------------------------------------------------------------------------


def _content_evaluator(candidate):
    """Seed ('x') scores 0; the reflection's 'improved' candidate scores 1."""
    return (1.0 if "improved" in str(candidate) else 0.0), {}


def _improve_lm(prompt):
    return "```\nimproved\n```"


def _run(tmp_path, **engine_kwargs):
    from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything

    return optimize_anything(
        seed_candidate="x",
        evaluator=_content_evaluator,
        config=GEPAConfig(
            engine=EngineConfig(max_metric_calls=40, run_dir=str(tmp_path), **engine_kwargs),
            reflection=ReflectionConfig(reflection_lm=_improve_lm),
        ),
    )


def test_manifest_absent_when_flag_off(tmp_path):
    _run(tmp_path)  # write_admission_manifest defaults to False
    assert not os.path.exists(os.path.join(str(tmp_path), "admission_manifest.jsonl"))


def test_requires_run_dir():
    from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything

    with pytest.raises(ValueError, match="requires run_dir"):
        optimize_anything(
            seed_candidate="x",
            evaluator=_content_evaluator,
            config=GEPAConfig(
                engine=EngineConfig(max_metric_calls=5, run_dir=None, write_admission_manifest=True),
                reflection=ReflectionConfig(reflection_lm=_improve_lm),
            ),
        )


def test_manifest_written_and_replayable(tmp_path):
    _run(tmp_path, write_admission_manifest=True)
    path = os.path.join(str(tmp_path), "admission_manifest.jsonl")
    assert os.path.exists(path)

    records = read_manifest(path)
    assert len(records) >= 1

    candidates = json.load(open(os.path.join(str(tmp_path), "candidates.json")))
    accepted = [r for r in records if r["decision"] == "accepted"]
    rejected = [r for r in records if r["decision"] == "rejected"]
    # The seed 'x' -> 'improved' improvement should be admitted at least once.
    assert len(accepted) >= 1

    for r in records:
        assert r["schema_version"] == SCHEMA_VERSION
        assert r["decision_basis"] == "subsample_acceptance"
        assert len(r["parent_candidate_ids"]) == len(r["parent_candidate_hashes"])

    for r in accepted:
        assert r["candidate_idx"] is not None
        assert r["validation"] is not None
        assert "val_average_score" in r["validation"]
        cand = candidates[r["candidate_idx"]]
        parents = [candidates[i] for i in r["parent_candidate_ids"]]
        assert verify_record(r, cand, parents) is True

    for r in rejected:
        assert r["candidate_idx"] is None
        assert r["validation"] is None
        parents = [candidates[i] for i in r["parent_candidate_ids"]]
        assert r["parent_candidate_hashes"] == [_candidate_hash(p) for p in parents]
