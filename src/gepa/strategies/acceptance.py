# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from typing import Protocol, runtime_checkable

from gepa.proposer.base import CandidateProposal


@runtime_checkable
class AcceptanceCriterion(Protocol):
    """Decides whether a proposed candidate should be accepted based on subsample evaluation results.

    The :pymethod:`should_accept` method receives the full ``CandidateProposal``, which
    contains the per-example scores before and after the mutation (``subsample_scores_before``
    and ``subsample_scores_after``), the candidate text, parent indices, and free-form
    metadata.  Implementations may use any subset of this information to make the
    acceptance decision.
    """

    def should_accept(self, proposal: CandidateProposal) -> bool:
        """Return ``True`` if the proposed candidate should be accepted.

        Args:
            proposal: The full proposal including old/new scores, candidate, and metadata.
        """
        ...


class StrictImprovementAcceptance:
    """Accept only if the sum of new subsample scores is strictly greater than the old sum.

    This is the default acceptance criterion used by GEPA.
    """

    def should_accept(self, proposal: CandidateProposal) -> bool:
        old_sum = sum(proposal.subsample_scores_before or [])
        new_sum = sum(proposal.subsample_scores_after or [])
        return new_sum > old_sum


class ImprovementOrEqualAcceptance:
    """Accept if the sum of new subsample scores is greater than or equal to the old sum.

    Useful when you want to allow lateral moves that don't improve the score but
    may explore different regions of the solution space.
    """

    def should_accept(self, proposal: CandidateProposal) -> bool:
        old_sum = sum(proposal.subsample_scores_before or [])
        new_sum = sum(proposal.subsample_scores_after or [])
        return new_sum >= old_sum
