# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import pytest

from gepa.core.state import GEPAState, ValsetEvaluation
from gepa.proposer.base import CandidateProposal, SubsampleEvaluation
from gepa.strategies.acceptance import (
    AcceptanceCriterion,
    ImprovementOrEqualAcceptance,
    StrictImprovementAcceptance,
)


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


def _make_proposal(scores_before: list[float], scores_after: list[float]) -> CandidateProposal:
    return CandidateProposal(
        candidate={"instructions": "test"},
        parent_program_ids=[0],
        subsample_scores_before=scores_before,
        subsample_scores_after=scores_after,
        eval_before=SubsampleEvaluation(
            scores=scores_before,
            outputs=["out"] * len(scores_before),
        ),
        eval_after=SubsampleEvaluation(
            scores=scores_after,
            outputs=["out"] * len(scores_after),
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
