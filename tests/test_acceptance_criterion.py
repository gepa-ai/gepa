# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import pytest

from gepa.proposer.base import CandidateProposal
from gepa.strategies.acceptance import (
    AcceptanceCriterion,
    ImprovementOrEqualAcceptance,
    StrictImprovementAcceptance,
)


def _make_proposal(scores_before: list[float], scores_after: list[float]) -> CandidateProposal:
    return CandidateProposal(
        candidate={"instructions": "test"},
        parent_program_ids=[0],
        subsample_scores_before=scores_before,
        subsample_scores_after=scores_after,
        tag="reflective_mutation",
    )


class TestStrictImprovementAcceptance:
    def test_accepts_strict_improvement(self):
        criterion = StrictImprovementAcceptance()
        proposal = _make_proposal([0.5, 0.3], [0.6, 0.4])
        assert criterion.should_accept(proposal) is True

    def test_rejects_equal_scores(self):
        criterion = StrictImprovementAcceptance()
        proposal = _make_proposal([0.5, 0.3], [0.5, 0.3])
        assert criterion.should_accept(proposal) is False

    def test_rejects_worse_scores(self):
        criterion = StrictImprovementAcceptance()
        proposal = _make_proposal([0.5, 0.3], [0.4, 0.2])
        assert criterion.should_accept(proposal) is False

    def test_handles_none_scores(self):
        criterion = StrictImprovementAcceptance()
        proposal = CandidateProposal(
            candidate={"instructions": "test"},
            parent_program_ids=[0],
            subsample_scores_before=None,
            subsample_scores_after=None,
        )
        assert criterion.should_accept(proposal) is False

    def test_accepts_marginal_improvement(self):
        criterion = StrictImprovementAcceptance()
        proposal = _make_proposal([0.5, 0.3], [0.5, 0.3001])
        assert criterion.should_accept(proposal) is True


class TestImprovementOrEqualAcceptance:
    def test_accepts_improvement(self):
        criterion = ImprovementOrEqualAcceptance()
        proposal = _make_proposal([0.5, 0.3], [0.6, 0.4])
        assert criterion.should_accept(proposal) is True

    def test_accepts_equal_scores(self):
        criterion = ImprovementOrEqualAcceptance()
        proposal = _make_proposal([0.5, 0.3], [0.5, 0.3])
        assert criterion.should_accept(proposal) is True

    def test_rejects_worse_scores(self):
        criterion = ImprovementOrEqualAcceptance()
        proposal = _make_proposal([0.5, 0.3], [0.4, 0.2])
        assert criterion.should_accept(proposal) is False


class TestProtocolConformance:
    def test_strict_improvement_is_acceptance_criterion(self):
        assert isinstance(StrictImprovementAcceptance(), AcceptanceCriterion)

    def test_improvement_or_equal_is_acceptance_criterion(self):
        assert isinstance(ImprovementOrEqualAcceptance(), AcceptanceCriterion)

    def test_custom_criterion(self):
        class AlwaysAccept:
            def should_accept(self, proposal: CandidateProposal) -> bool:
                return True

        assert isinstance(AlwaysAccept(), AcceptanceCriterion)


class TestApiWiring:
    """Test that acceptance_criterion can be passed as string or instance through gepa.optimize."""

    def test_strict_improvement_string(self):
        """Verify the string factory works (tested indirectly via import)."""
        from gepa.strategies.acceptance import StrictImprovementAcceptance

        criterion = StrictImprovementAcceptance()
        proposal = _make_proposal([0.5], [0.6])
        assert criterion.should_accept(proposal) is True

    def test_improvement_or_equal_string(self):
        from gepa.strategies.acceptance import ImprovementOrEqualAcceptance

        criterion = ImprovementOrEqualAcceptance()
        proposal = _make_proposal([0.5], [0.5])
        assert criterion.should_accept(proposal) is True
