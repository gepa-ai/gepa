import pytest

from gepa.core.adapter import EvaluationBatch
from gepa.core.data_loader import ensure_loader
from gepa.evidence.config import SequentialEvidenceConfig
from gepa.evidence.controller import EvidenceController
from gepa.proposer.base import CandidateProposal


class Adapter:
    propose_new_texts = None
    def __init__(self, fail_at=None):
        self.calls = []
        self.fail_at = fail_at
    def evaluate(self, batch, candidate, capture_traces=False):
        self.calls.append((candidate["p"], tuple(batch)))
        if self.fail_at == len(self.calls):
            raise RuntimeError("informative missingness")
        return EvaluationBatch(list(batch), [float(candidate["p"] == "child")] * len(batch))


def controller(tmp_path, adapter=None, **kwargs):
    return EvidenceController(
        SequentialEvidenceConfig("v1", kwargs.pop("comparison_cap", 300), **kwargs),
        adapter or Adapter(), ensure_loader(["a", "b"]), ensure_loader(["v"]),
        max_metric_calls=1000, run_dir=str(tmp_path), seed=0,
    )


def proposal():
    return CandidateProposal({"p": "child"}, [0], [0, 1], [0, 0], [1, 1])


def test_comparison_collects_fresh_balanced_complete_rounds(tmp_path):
    raw = Adapter()
    control = controller(tmp_path, raw)
    decision = control.compare(proposal(), {"p": "parent"})
    assert decision.status == "supported_improvement"
    assert decision.lower > 0
    assert decision.completed_rounds >= 2
    assert len(raw.calls) == 2 * decision.completed_rounds
    assert all(ids == ("a", "b") for _, ids in raw.calls)
    assert decision.metric_calls == 4 * decision.completed_rounds == control.guard.used
    assert decision.support == "bounded_independent"
    assert len({o.identity.key for o in decision.observations.values()}) == 4 * decision.completed_rounds


def test_informative_failure_is_terminal_without_selective_retry(tmp_path):
    raw = Adapter(fail_at=2)
    control = controller(tmp_path, raw)
    decision = control.compare(proposal(), {"p": "parent"})
    assert decision.status == "inconclusive"
    assert decision.completed_rounds == 0
    assert control.guard.used == 4
    assert len(raw.calls) == 2
    assert "missingness" in decision.reason
    assert not decision.eligible


def test_budget_stops_before_unbalanced_round(tmp_path):
    control = controller(tmp_path, comparison_cap=7)
    decision = control.compare(proposal(), {"p": "parent"})
    assert decision.status == "budget_exhausted"
    assert decision.completed_rounds == 1
    assert control.guard.used == 4


def test_replayed_decision_does_not_reexecute_or_relearn(tmp_path):
    raw = Adapter()
    control = controller(tmp_path, raw)
    p = proposal()
    first = control.compare(p, {"p": "parent"})
    n = len(raw.calls)
    assert control.compare(p, {"p": "parent"}).to_dict() == first.to_dict()
    assert len(raw.calls) == n
    restored = controller(tmp_path, Adapter())
    assert restored.guard.used == control.guard.used
    assert restored.report()["comparisons"][0]["completed_rounds"] == first.completed_rounds


def test_context_and_example_content_mismatch_reject_resume(tmp_path):
    controller(tmp_path).compare(proposal(), {"p": "parent"})
    with pytest.raises(ValueError, match="context"):
        controller(tmp_path, effect_margin=0.1)
    with pytest.raises(ValueError, match="context"):
        EvidenceController(SequentialEvidenceConfig("v1", 300), Adapter(), ensure_loader(["changed", "b"]),
                           ensure_loader(["v"]), max_metric_calls=1000, run_dir=str(tmp_path), seed=0)


def test_heuristic_never_claims_statistical_support(tmp_path):
    control = controller(tmp_path, comparison_cap=8, assumption="heuristic", unresolved="explore")
    decision = control.compare(proposal(), {"p": "parent"})
    assert decision.status == "heuristic"
    assert decision.support == "none"
    assert not decision.eligible


def test_duplicate_or_changed_strata_fail_before_execution(tmp_path):
    control = controller(tmp_path)
    p = proposal()
    p.subsample_indices = [0, 0]
    with pytest.raises(ValueError, match="unique"):
        control.compare(p, {"p": "parent"})
    assert control.guard.used == 0
