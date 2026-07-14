# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for the opt-in admission manifest (per-decision JSONL audit ledger)."""

import json
import os

import pytest

from gepa.core.state import _candidate_hash
from gepa.logging.admission_manifest import (
    IMPROVEMENT_OR_EQUAL,
    MERGE_GATE,
    SCHEMA_VERSION,
    STRICT_IMPROVEMENT,
    AdmissionManifestWriter,
    acceptance_operator,
    build_decision_record,
    components_changed,
    criterion_name,
    read_manifest,
    verify_record,
)
from gepa.strategies.acceptance import ImprovementOrEqualAcceptance, StrictImprovementAcceptance

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
        acceptance_criterion=IMPROVEMENT_OR_EQUAL,
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
    # The criterion and its operator are recorded so replay is self-contained.
    assert rec["acceptance_criterion"] == IMPROVEMENT_OR_EQUAL
    assert rec["acceptance_operator"] == ">="


# ---------------------------------------------------------------------------
# Unit: criterion_name / acceptance_operator
# ---------------------------------------------------------------------------


def test_criterion_name_builtins_use_public_string_names():
    # Instance or class, the built-ins map to the public optimize() string names.
    assert criterion_name(StrictImprovementAcceptance()) == STRICT_IMPROVEMENT
    assert criterion_name(ImprovementOrEqualAcceptance) == IMPROVEMENT_OR_EQUAL


def test_criterion_name_custom_falls_back_to_class_name():
    class MyCriterion:
        pass

    assert criterion_name(MyCriterion()) == "MyCriterion"


def test_acceptance_operator_mapping():
    assert acceptance_operator(STRICT_IMPROVEMENT) == ">"
    assert acceptance_operator(IMPROVEMENT_OR_EQUAL) == ">="
    assert acceptance_operator(MERGE_GATE) == ">="
    assert acceptance_operator("MyCriterion") is None
    assert acceptance_operator(None) is None


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


def _rec(decision, before, after, cand, parents, kind="reflective_mutation", acceptance=None):
    if acceptance is None:
        acceptance = MERGE_GATE if kind == "merge" else STRICT_IMPROVEMENT
    return build_decision_record(
        iteration=1,
        proposal_kind=kind,
        decision=decision,
        decision_reason="r",
        acceptance_criterion=acceptance,
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


def test_verify_record_equal_score_accept_under_improvement_or_equal():
    # The maintainer's bug: an equal-score acceptance under improvement_or_equal
    # must replay True from the record alone, without the caller re-supplying the
    # strategy. Under the default strict-improvement operator it would be False.
    parent, cand = {"a": "x"}, {"a": "y"}
    rec = _rec("accepted", 1.0, 1.0, cand, [parent], acceptance=IMPROVEMENT_OR_EQUAL)
    assert rec["acceptance_operator"] == ">="
    assert verify_record(rec, cand, [parent]) is True  # no acceptance= override needed
    # Same record under strict improvement would (correctly) be inconsistent.
    strict = _rec("accepted", 1.0, 1.0, cand, [parent], acceptance=STRICT_IMPROVEMENT)
    assert verify_record(strict, cand, [parent]) is False


def test_verify_record_uses_recorded_operator_over_caller_override():
    # A stale/incorrect caller override must not flip a record that stores its
    # own operator -- the record is authoritative.
    parent, cand = {"a": "x"}, {"a": "y"}
    rec = _rec("accepted", 1.0, 1.0, cand, [parent], acceptance=IMPROVEMENT_OR_EQUAL)
    assert verify_record(rec, cand, [parent], acceptance=STRICT_IMPROVEMENT) is True


def test_verify_record_custom_criterion_score_decision_out_of_scope():
    # A custom criterion records a null operator: hashes/components are still
    # verified, but the score decision is not re-derivable, so a "surprising"
    # accept is not flagged as inconsistent.
    parent, cand = {"a": "x"}, {"a": "y"}
    rec = _rec("accepted", 2.0, 1.0, cand, [parent], acceptance="MyCriterion")
    assert rec["acceptance_operator"] is None
    assert verify_record(rec, cand, [parent]) is True
    # ...but tampering with identity is still caught.
    rec["candidate_hash"] = "deadbeef"
    assert verify_record(rec, cand, [parent]) is False


def test_verify_record_legacy_without_recorded_operator():
    # Backward-compat: a record written before the operator was recorded falls
    # back to the caller override (then to strict/merge defaults).
    parent, cand = {"a": "x"}, {"a": "y"}
    rec = _rec("accepted", 1.0, 1.0, cand, [parent], acceptance=IMPROVEMENT_OR_EQUAL)
    rec.pop("acceptance_operator")
    rec.pop("acceptance_criterion")
    assert verify_record(rec, cand, [parent], acceptance=IMPROVEMENT_OR_EQUAL) is True
    assert verify_record(rec, cand, [parent]) is False  # defaults to strict


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
        # The engine records the criterion that made each decision (default is
        # strict improvement for reflective, the merge gate for merges).
        expected = MERGE_GATE if r["proposal_kind"] == "merge" else STRICT_IMPROVEMENT
        assert r["acceptance_criterion"] == expected
        assert r["acceptance_operator"] == acceptance_operator(expected)

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


def test_manifest_records_improvement_or_equal_criterion(tmp_path):
    # A run configured with improvement_or_equal records the >= operator, and its
    # records replay from local files alone (no acceptance= override needed).
    _run(tmp_path, write_admission_manifest=True, acceptance_criterion="improvement_or_equal")
    path = os.path.join(str(tmp_path), "admission_manifest.jsonl")
    records = read_manifest(path)
    candidates = json.load(open(os.path.join(str(tmp_path), "candidates.json")))
    reflective = [r for r in records if r["proposal_kind"] != "merge"]
    assert reflective, "expected at least one reflective decision"
    for r in reflective:
        assert r["acceptance_criterion"] == IMPROVEMENT_OR_EQUAL
        assert r["acceptance_operator"] == ">="
    for r in records:
        if r["decision"] == "accepted":
            cand = candidates[r["candidate_idx"]]
            parents = [candidates[i] for i in r["parent_candidate_ids"]]
            assert verify_record(r, cand, parents) is True
