# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import pytest

import gepa
from gepa.core.adapter import EvaluationBatch
from gepa.core.state import GEPAState, ValsetEvaluation
from gepa.proposer.base import CandidateProposal, SubsampleEvaluation
from gepa.strategies.acceptance import (
    AcceptanceCriterion,
    ImprovementOrEqualAcceptance,
    NonRegressionAcceptance,
    StrictImprovementAcceptance,
)
from gepa.strategies.proposal_sampling import SameParentSampling, SingleMutationSampling
from gepa.strategies.proposal_selection import AllImprovements, BestImprovement, TopKImprovements


@pytest.fixture
def mock_state():
    """Minimal GEPAState for acceptance criterion tests."""
    seed_candidate = {"instructions": "test"}
    base_eval = ValsetEvaluation(
        outputs_by_val_id={0: "out"},
        scores_by_val_id={0: 0.5},
        objective_scores_by_val_id=None,
    )
    return GEPAState(seed_candidate, base_eval, track_best_outputs=False)


def _make_proposal(
    scores_before: list[float],
    scores_after: list[float],
    objectives_before: list[dict[str, float]] | None = None,
    objectives_after: list[dict[str, float]] | None = None,
) -> CandidateProposal:
    return CandidateProposal(
        candidate={"instructions": "test"},
        parent_program_ids=[0],
        subsample_scores_before=scores_before,
        subsample_scores_after=scores_after,
        eval_before=SubsampleEvaluation(
            scores=scores_before,
            outputs=["out"] * len(scores_before),
            objective_scores=objectives_before,
        ),
        eval_after=SubsampleEvaluation(
            scores=scores_after,
            outputs=["out"] * len(scores_after),
            objective_scores=objectives_after,
        ),
        tag="reflective_mutation",
    )


class TestStrictImprovementAcceptance:
    def test_accepts_strict_improvement(self, mock_state):
        criterion = StrictImprovementAcceptance()
        proposal = _make_proposal([0.5, 0.3], [0.6, 0.4])
        assert criterion.should_accept(proposal, mock_state) is True

    def test_rejects_equal_scores(self, mock_state):
        criterion = StrictImprovementAcceptance()
        proposal = _make_proposal([0.5, 0.3], [0.5, 0.3])
        assert criterion.should_accept(proposal, mock_state) is False

    def test_rejects_worse_scores(self, mock_state):
        criterion = StrictImprovementAcceptance()
        proposal = _make_proposal([0.5, 0.3], [0.4, 0.2])
        assert criterion.should_accept(proposal, mock_state) is False

    def test_handles_none_scores(self, mock_state):
        criterion = StrictImprovementAcceptance()
        proposal = CandidateProposal(
            candidate={"instructions": "test"},
            parent_program_ids=[0],
            subsample_scores_before=None,
            subsample_scores_after=None,
        )
        assert criterion.should_accept(proposal, mock_state) is False

    def test_accepts_marginal_improvement(self, mock_state):
        criterion = StrictImprovementAcceptance()
        proposal = _make_proposal([0.5, 0.3], [0.5, 0.3001])
        assert criterion.should_accept(proposal, mock_state) is True


class TestImprovementOrEqualAcceptance:
    def test_accepts_improvement(self, mock_state):
        criterion = ImprovementOrEqualAcceptance()
        proposal = _make_proposal([0.5, 0.3], [0.6, 0.4])
        assert criterion.should_accept(proposal, mock_state) is True

    def test_accepts_equal_scores(self, mock_state):
        criterion = ImprovementOrEqualAcceptance()
        proposal = _make_proposal([0.5, 0.3], [0.5, 0.3])
        assert criterion.should_accept(proposal, mock_state) is True

    def test_rejects_worse_scores(self, mock_state):
        criterion = ImprovementOrEqualAcceptance()
        proposal = _make_proposal([0.5, 0.3], [0.4, 0.2])
        assert criterion.should_accept(proposal, mock_state) is False


class TestProtocolConformance:
    def test_strict_improvement_is_acceptance_criterion(self):
        assert isinstance(StrictImprovementAcceptance(), AcceptanceCriterion)

    def test_improvement_or_equal_is_acceptance_criterion(self):
        assert isinstance(ImprovementOrEqualAcceptance(), AcceptanceCriterion)

    def test_custom_criterion_using_objective_scores(self, mock_state):
        """A custom criterion can use eval_before/eval_after for multi-objective decisions."""

        class ObjectiveImprovementAcceptance:
            def __init__(self, objective: str):
                self.objective = objective

            def should_accept(self, proposal: CandidateProposal, state: GEPAState) -> bool:
                if proposal.eval_before is None or proposal.eval_after is None:
                    return False
                if proposal.eval_before.objective_scores is None or proposal.eval_after.objective_scores is None:
                    return False
                old_obj = sum(s.get(self.objective, 0.0) for s in proposal.eval_before.objective_scores)
                new_obj = sum(s.get(self.objective, 0.0) for s in proposal.eval_after.objective_scores)
                return new_obj > old_obj

        criterion = ObjectiveImprovementAcceptance("accuracy")
        assert isinstance(criterion, AcceptanceCriterion)

        proposal = CandidateProposal(
            candidate={"instructions": "test"},
            parent_program_ids=[0],
            subsample_scores_before=[0.5],
            subsample_scores_after=[0.5],
            eval_before=SubsampleEvaluation(
                scores=[0.5],
                outputs=["old"],
                objective_scores=[{"accuracy": 0.4, "speed": 0.9}],
            ),
            eval_after=SubsampleEvaluation(
                scores=[0.5],
                outputs=["new"],
                objective_scores=[{"accuracy": 0.6, "speed": 0.7}],
            ),
        )
        # Aggregate score is the same, but accuracy objective improved
        assert criterion.should_accept(proposal, mock_state) is True

    def test_custom_criterion_using_outputs(self, mock_state):
        """A custom criterion can inspect outputs, not just scores."""

        class RejectEmptyOutputs:
            def should_accept(self, proposal: CandidateProposal, state: GEPAState) -> bool:
                if proposal.eval_after is None:
                    return False
                return all(output != "" for output in proposal.eval_after.outputs)

        criterion = RejectEmptyOutputs()
        assert isinstance(criterion, AcceptanceCriterion)

        proposal = CandidateProposal(
            candidate={"instructions": "test"},
            parent_program_ids=[0],
            subsample_scores_before=[0.5],
            subsample_scores_after=[1.0],
            eval_before=SubsampleEvaluation(scores=[0.5], outputs=["ok"]),
            eval_after=SubsampleEvaluation(scores=[1.0], outputs=[""]),
        )
        assert criterion.should_accept(proposal, mock_state) is False

    def test_custom_criterion_using_trajectories(self, mock_state):
        """A custom criterion can inspect trajectories."""

        class RejectIfNoTrajectories:
            def should_accept(self, proposal: CandidateProposal, state: GEPAState) -> bool:
                if proposal.eval_before is None or not proposal.eval_before.trajectories:
                    return False
                return sum(proposal.subsample_scores_after or []) > sum(proposal.subsample_scores_before or [])

        criterion = RejectIfNoTrajectories()

        proposal = CandidateProposal(
            candidate={"instructions": "test"},
            parent_program_ids=[0],
            subsample_scores_before=[0.5],
            subsample_scores_after=[0.6],
            eval_before=SubsampleEvaluation(scores=[0.5], outputs=["ok"], trajectories=["trace1"]),
            eval_after=SubsampleEvaluation(scores=[0.6], outputs=["ok"]),
        )
        assert criterion.should_accept(proposal, mock_state) is True

        proposal_no_traces = CandidateProposal(
            candidate={"instructions": "test"},
            parent_program_ids=[0],
            subsample_scores_before=[0.5],
            subsample_scores_after=[0.6],
            eval_before=SubsampleEvaluation(scores=[0.5], outputs=["ok"], trajectories=None),
            eval_after=SubsampleEvaluation(scores=[0.6], outputs=["ok"]),
        )
        assert criterion.should_accept(proposal_no_traces, mock_state) is False

    def test_custom_criterion_using_state(self, mock_state):
        """A custom criterion can use GEPAState for context-dependent decisions."""

        class AcceptOnlyEarlyIterations:
            """Accept improvements only in the first N iterations, then require strict improvement."""

            def __init__(self, lenient_until: int):
                self.lenient_until = lenient_until

            def should_accept(self, proposal: CandidateProposal, state: GEPAState) -> bool:
                old_sum = sum(proposal.subsample_scores_before or [])
                new_sum = sum(proposal.subsample_scores_after or [])
                if state.i < self.lenient_until:
                    return new_sum >= old_sum  # allow equal in early iterations
                return new_sum > old_sum  # strict after

        criterion = AcceptOnlyEarlyIterations(lenient_until=5)
        assert isinstance(criterion, AcceptanceCriterion)

        proposal_equal = _make_proposal([0.5], [0.5])

        # Early iteration (i=0): equal scores accepted
        mock_state.i = 0
        assert criterion.should_accept(proposal_equal, mock_state) is True

        # Late iteration (i=10): equal scores rejected
        mock_state.i = 10
        assert criterion.should_accept(proposal_equal, mock_state) is False


class TestSubsampleEvaluation:
    def test_defaults(self):
        e = SubsampleEvaluation(scores=[0.5])
        assert e.scores == [0.5]
        assert e.outputs == []
        assert e.objective_scores is None
        assert e.trajectories is None

    def test_full_construction(self):
        e = SubsampleEvaluation(
            scores=[0.5, 0.7],
            outputs=["a", "b"],
            objective_scores=[{"acc": 0.5}, {"acc": 0.7}],
            trajectories=["t1", "t2"],
        )
        assert e.scores == [0.5, 0.7]
        assert e.outputs == ["a", "b"]
        assert e.objective_scores == [{"acc": 0.5}, {"acc": 0.7}]
        assert e.trajectories == ["t1", "t2"]


class RecordingAcceptance:
    def __init__(self, accepted=True):
        self.accepted = accepted
        self.calls = []

    def should_accept(self, proposal, state):
        self.calls.append((proposal, state))
        return self.accepted


@pytest.fixture
def base_criterion():
    return RecordingAcceptance()


@pytest.fixture
def protected_proposal():
    return _make_proposal(
        [0.5, 0.2],
        [0.5, 0.8],
        [{"schema_valid": 1.0}, {}],
        [{"schema_valid": 1.0}, {}],
    )


class TestNonRegressionAcceptance:
    @pytest.mark.parametrize(
        "before,after,accepted",
        [
            pytest.param([{"schema_valid": 1.0}], [{"schema_valid": 1.0}], True, id="preserved-pass"),
            pytest.param([{"schema_valid": 1.0}], [{"schema_valid": 0.0}], False, id="regression"),
            pytest.param([{"schema_valid": 1.0}], [{}], False, id="missing-passing-check"),
            pytest.param([{"schema_valid": 0.0}], [{"schema_valid": 0.0}], True, id="existing-failure"),
            pytest.param([{"schema_valid": 0.8}], [{"schema_valid": 0.2}], True, id="only-passes-protected"),
            pytest.param([{"schema_valid": 0.0}], [{"schema_valid": 1.0}], True, id="new-pass"),
            pytest.param([{}], [{"schema_valid": 0.0}], True, id="not-previously-protected"),
            pytest.param([{"other": 1.0}], [{"other": 0.0}], True, id="unprotected-objective"),
            pytest.param(
                [{"schema_valid": 1.0}, {"schema_valid": 0.0}],
                [{"schema_valid": 0.0}, {"schema_valid": 1.0}],
                False,
                id="equal-total-hides-regression",
            ),
            pytest.param(
                [{"schema_valid": 1.0}, {}],
                [{"schema_valid": 1.0}, {}],
                True,
                id="sparse-objectives",
            ),
        ],
    )
    def test_protected_passes(self, mock_state, base_criterion, before, after, accepted):
        proposal = _make_proposal([0.5] * len(before), [0.8] * len(after), before, after)
        criterion = NonRegressionAcceptance(base_criterion, protected=("schema_valid",))
        assert criterion.should_accept(proposal, mock_state) is accepted
        assert base_criterion.calls == ([(proposal, mock_state)] if accepted else [])

    def test_vetoes_even_when_aggregate_improves(self, mock_state):
        proposal = _make_proposal(
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [{"schema_valid": 1.0}, {}, {}],
            [{"schema_valid": 0.0}, {}, {}],
        )
        base = StrictImprovementAcceptance()
        assert base.should_accept(proposal, mock_state) is True
        criterion = NonRegressionAcceptance(base, protected=("schema_valid",))
        assert criterion.should_accept(proposal, mock_state) is False
        assert criterion.reject_reason(proposal, mock_state) == (
            "Protected objective 'schema_valid' regressed at minibatch index 0"
        )

    @pytest.mark.parametrize("failed_check", ["schema_valid", "required_field"])
    def test_all_named_checks_are_protected(self, mock_state, base_criterion, failed_check):
        before = {"schema_valid": 1.0, "required_field": 1.0}
        after = {**before, failed_check: 0.0}
        proposal = _make_proposal([0.5], [1.0], [before], [after])
        criterion = NonRegressionAcceptance(base_criterion, protected=("schema_valid", "required_field"))
        assert criterion.should_accept(proposal, mock_state) is False
        assert failed_check in criterion.reject_reason(proposal, mock_state)
        assert base_criterion.calls == []

    @pytest.mark.parametrize("accepted", [True, False])
    def test_delegates_clean_proposals(self, mock_state, base_criterion, protected_proposal, accepted):
        base_criterion.accepted = accepted
        criterion = NonRegressionAcceptance(base_criterion, protected=("schema_valid",))
        assert isinstance(criterion, AcceptanceCriterion)
        assert criterion.should_accept(protected_proposal, mock_state) is accepted
        assert base_criterion.calls == [(protected_proposal, mock_state)]

    @pytest.mark.parametrize(
        "base,accepted", [(StrictImprovementAcceptance(), False), (ImprovementOrEqualAcceptance(), True)]
    )
    def test_preserves_base_tie_policy(self, mock_state, base, accepted):
        proposal = _make_proposal([0.5], [0.5], [{"schema_valid": 1.0}], [{"schema_valid": 1.0}])
        criterion = NonRegressionAcceptance(base, protected=("schema_valid",))
        assert criterion.should_accept(proposal, mock_state) is accepted

    @pytest.mark.parametrize(
        "attribute,value",
        [
            ("eval_before", None),
            ("eval_after", None),
            ("subsample_scores_before", []),
            ("subsample_scores_after", [0.8]),
            ("subsample_indices", [0]),
        ],
    )
    def test_rejects_incomplete_proposals(self, mock_state, base_criterion, protected_proposal, attribute, value):
        setattr(protected_proposal, attribute, value)
        criterion = NonRegressionAcceptance(base_criterion, protected=("schema_valid",))
        assert criterion.should_accept(protected_proposal, mock_state) is False
        assert "could not be verified" in criterion.reject_reason(protected_proposal, mock_state)
        assert base_criterion.calls == []

    @pytest.mark.parametrize("side", ["eval_before", "eval_after"])
    @pytest.mark.parametrize(
        "attribute,value",
        [
            ("objective_scores", None),
            ("objective_scores", []),
            ("objective_scores", [{"schema_valid": 1.0}]),
            ("objective_scores", [None, {}]),
            ("objective_scores", [[], {}]),
            ("scores", [0.5]),
        ],
    )
    def test_rejects_malformed_evaluations(
        self, mock_state, base_criterion, protected_proposal, side, attribute, value
    ):
        setattr(getattr(protected_proposal, side), attribute, value)
        criterion = NonRegressionAcceptance(base_criterion, protected=("schema_valid",))
        assert criterion.should_accept(protected_proposal, mock_state) is False
        assert "could not be verified" in criterion.reject_reason(protected_proposal, mock_state)
        assert base_criterion.calls == []

    @pytest.mark.parametrize("side", ["eval_before", "eval_after"])
    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), None, "1.0"])
    def test_rejects_invalid_protected_scores(self, mock_state, base_criterion, protected_proposal, side, value):
        getattr(protected_proposal, side).objective_scores[0]["schema_valid"] = value
        criterion = NonRegressionAcceptance(base_criterion, protected=("schema_valid",))
        assert criterion.should_accept(protected_proposal, mock_state) is False
        assert "invalid score for 'schema_valid'" in criterion.reject_reason(protected_proposal, mock_state)
        assert base_criterion.calls == []

    def test_rejects_empty_evaluations(self, mock_state, base_criterion):
        proposal = _make_proposal([], [], [], [])
        criterion = NonRegressionAcceptance(base_criterion, protected=("schema_valid",))
        assert criterion.should_accept(proposal, mock_state) is False
        assert base_criterion.calls == []

    @pytest.mark.parametrize("protected", [(), [], "schema_valid", b"schema_valid", ("",), (None,)])
    def test_requires_objective_names(self, base_criterion, protected):
        with pytest.raises(ValueError, match="nonempty sequence of nonempty objective names"):
            NonRegressionAcceptance(base_criterion, protected=protected)

    def test_copies_objective_names(self, mock_state, base_criterion, protected_proposal):
        protected = ["schema_valid"]
        criterion = NonRegressionAcceptance(base_criterion, protected=protected)
        protected.clear()
        protected_proposal.eval_after.objective_scores[0]["schema_valid"] = 0.0
        assert criterion.should_accept(protected_proposal, mock_state) is False

    @pytest.mark.parametrize("custom_reason", [True, False])
    def test_rejection_logging_does_not_repeat_base_decision(self, mock_state, protected_proposal, custom_reason):
        class StatefulBase(RecordingAcceptance):
            def should_accept(self, proposal, state):
                super().should_accept(proposal, state)
                return len(self.calls) > 1

        class StatefulBaseWithReason(StatefulBase):
            def reject_reason(self, proposal, state):
                return "base declined this proposal"

        base = StatefulBaseWithReason() if custom_reason else StatefulBase()
        criterion = NonRegressionAcceptance(base, protected=("schema_valid",))
        assert criterion.should_accept(protected_proposal, mock_state) is False
        reason = criterion.reject_reason(protected_proposal, mock_state)
        expected = "base declined this proposal" if custom_reason else "base acceptance criterion (StatefulBase)"
        assert expected in reason
        assert base.calls == [(protected_proposal, mock_state)]


@pytest.fixture
def tradeoff_adapter():
    class TradeoffAdapter:
        def __init__(self):
            self.metric_calls = 0

        def evaluate(self, batch, candidate, capture_traces=False):
            self.metric_calls += len(batch)
            updated = candidate["instructions"] == "updated"
            scores = [float(not updated) if row == 0 else float(updated) for row in batch]
            return EvaluationBatch(
                outputs=["output"] * len(batch),
                scores=scores,
                objective_scores=[
                    {"schema_valid": score} if row == 0 else {} for row, score in zip(batch, scores, strict=True)
                ],
                trajectories=[{"feedback": "Improve ordinary cases while preserving schema validity."}] * len(batch)
                if capture_traces
                else None,
            )

        def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
            return dict.fromkeys(components_to_update, eval_batch.trajectories)

        def propose_new_texts(self, candidate, reflective_dataset, components_to_update):
            return dict.fromkeys(components_to_update, "updated")

    return TradeoffAdapter()


@pytest.mark.parametrize("guarded", [False, True])
@pytest.mark.parametrize(
    "selection", [AllImprovements, BestImprovement, lambda: TopKImprovements(k=1)], ids=["all", "best", "top-k"]
)
@pytest.mark.parametrize(
    "sampling", [SingleMutationSampling, lambda: SameParentSampling(n=2)], ids=["single", "multiple"]
)
def test_non_regression_acceptance_in_optimize(tmp_path, tradeoff_adapter, guarded, selection, sampling):
    class Rejections:
        def __init__(self):
            self.reasons = []

        def on_candidate_rejected(self, event):
            self.reasons.append(event["reason"])

    log = Rejections()
    base = StrictImprovementAcceptance()
    criterion = NonRegressionAcceptance(base, protected=("schema_valid",)) if guarded else base
    result = gepa.optimize(
        seed_candidate={"instructions": "seed"},
        trainset=[0, 1, 2],
        valset=[0, 1, 2],
        adapter=tradeoff_adapter,
        acceptance_criterion=criterion,
        sampling_strategy=sampling(),
        selection_strategy=selection(),
        reflection_minibatch_size=3,
        max_metric_calls=12,
        callbacks=[log],
        run_dir=str(tmp_path),
        display_progress_bar=False,
        seed=0,
    )
    candidates = [candidate["instructions"] for candidate in result.candidates]
    assert candidates == (["seed"] if guarded else ["seed", "updated"])
    assert result.total_metric_calls == tradeoff_adapter.metric_calls
    if guarded:
        assert log.reasons
        assert all("Protected objective 'schema_valid' regressed" in reason for reason in log.reasons)
