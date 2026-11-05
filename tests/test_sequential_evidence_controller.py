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
        adapter or Adapter(),
        ensure_loader(["a", "b"]),
        ensure_loader(["v"]),
        max_metric_calls=1000,
        run_dir=str(tmp_path),
        seed=0,
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
        EvidenceController(
            SequentialEvidenceConfig("v1", 300),
            Adapter(),
            ensure_loader(["changed", "b"]),
            ensure_loader(["v"]),
            max_metric_calls=1000,
            run_dir=str(tmp_path),
            seed=0,
        )


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


@pytest.mark.parametrize("bad_score", [float("nan"), float("inf"), -0.01, 1.01])
def test_invalid_replication_score_is_terminal_missingness(tmp_path, bad_score):
    class Invalid(Adapter):
        def evaluate(self, batch, candidate, capture_traces=False):
            result = super().evaluate(batch, candidate, capture_traces)
            if len(self.calls) == 2:
                result.scores[0] = bad_score
            return result

    raw = Invalid()
    decision = controller(tmp_path, raw).compare(proposal(), {"p": "parent"})
    assert decision.status == "inconclusive" and decision.completed_rounds == 0
    assert len(raw.calls) == 2


def test_supported_regression_minimum_rounds_and_tie_exhaustion(tmp_path):
    class Reverse(Adapter):
        def evaluate(self, batch, candidate, capture_traces=False):
            result = super().evaluate(batch, candidate, capture_traces)
            result.scores = [1 - score for score in result.scores]
            return result

    regression = controller(tmp_path / "regression", Reverse()).compare(proposal(), {"p": "parent"})
    assert regression.status == "supported_regression" and regression.upper < 0
    short = controller(tmp_path / "minimum", min_rounds=100, comparison_cap=16).compare(proposal(), {"p": "parent"})
    assert short.status == "budget_exhausted" and short.completed_rounds == 4
    p = proposal()
    p.candidate = {"p": "parent"}
    tie = controller(tmp_path / "tie", comparison_cap=16).compare(p, {"p": "parent"})
    assert tie.status == "budget_exhausted" and tie.lower <= 0 <= tie.upper


def test_extreme_finite_score_range_does_not_overflow_persistence(tmp_path):
    class Huge(Adapter):
        def evaluate(self, batch, candidate, capture_traces=False):
            result = super().evaluate(batch, candidate, capture_traces)
            result.scores = [
                (0.95 if score else 0.05) * 1e300 + ((-1) ** (len(self.calls) // 2)) * 4e298 for score in result.scores
            ]
            return result

    control = controller(tmp_path, Huge(), score_min=0, score_max=1e300)
    decision = control.compare(proposal(), {"p": "parent"})
    assert decision.status == "supported_improvement"
    # Nonconstant raw scores make a variance of order 1e600; interval state
    # must use normalized bounded observations instead of overflowing m2.
    assert all(m.m2 < float("inf") for m in decision.moments_child)


def test_replication_context_gives_unique_trials_and_shared_seeds_without_pairing_claim(tmp_path):
    class ContextAdapter(Adapter):
        def __init__(self):
            super().__init__()
            self.contexts = []

        def evaluate_replication(self, batch, candidate, *, trial_context, capture_traces):
            self.contexts.append(trial_context)
            return self.evaluate(batch, candidate, capture_traces)

    raw = ContextAdapter()
    record = controller(tmp_path, raw).compare(proposal(), {"p": "parent"})
    contexts = raw.contexts
    assert record.summary()["replication_context_supplied"]
    assert not record.summary()["paired_randomness_benefit_claimed"]
    seeds = []
    for parent, child in zip(contexts[::2], contexts[1::2], strict=True):
        assert parent.trial_id == child.trial_id
        assert parent.paired_seed == child.paired_seed
        assert (parent.side, child.side) == ("parent", "child")
        seeds.append(parent.paired_seed)
    assert len(set(seeds)) == len(seeds)


def test_unsupported_identifier_rejected_before_scoring(tmp_path):
    class Loader:
        def all_ids(self):
            return [("tuple", 1)]

        def fetch(self, ids):
            return ["example"] * len(ids)

        def __len__(self):
            return 1

    raw = Adapter()
    with pytest.raises(ValueError, match="IDs"):
        EvidenceController(
            SequentialEvidenceConfig("v1", 300),
            raw,
            Loader(),
            Loader(),
            max_metric_calls=100,
            run_dir=str(tmp_path),
            seed=0,
        )
    assert raw.calls == []
