# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from typing import Protocol, runtime_checkable

from gepa.core.state import GEPAState
from gepa.proposer.base import CandidateProposal


@runtime_checkable
class AcceptanceCriterion(Protocol):
    """Decides whether a proposed candidate should be accepted based on subsample evaluation results.

    The ``should_accept`` method receives:

    - ``proposal``: the full ``CandidateProposal`` containing:

      - ``eval_before`` / ``eval_after``: ``SubsampleEvaluation`` objects with
        per-example ``scores``, ``outputs``, ``objective_scores``, and ``trajectories``.
      - ``subsample_scores_before`` / ``subsample_scores_after``: shorthand score lists.
      - ``candidate``: the proposed candidate text.
      - ``parent_program_ids``: indices of parent candidates.
      - ``metadata``: free-form dict with LM prompts and raw outputs.

    - ``state``: the full ``GEPAState``, giving access to all existing candidates,
      validation scores, the Pareto frontier, iteration count, etc.
    """

    def should_accept(self, proposal: CandidateProposal, state: GEPAState) -> bool:
        """Return ``True`` if the proposed candidate should be accepted.

        Args:
            proposal: The full proposal including evaluation data, candidate, and metadata.
            state: The current optimization state.
        """
        ...


class StrictImprovementAcceptance:
    """Accept only if the sum of new subsample scores is strictly greater than the old sum.

    This is the default acceptance criterion used by GEPA.
    """

    def should_accept(self, proposal: CandidateProposal, state: GEPAState) -> bool:
        old_sum = sum(proposal.subsample_scores_before or [])
        new_sum = sum(proposal.subsample_scores_after or [])
        return new_sum > old_sum


class ImprovementOrEqualAcceptance:
    """Accept if the sum of new subsample scores is greater than or equal to the old sum.

    Useful when you want to allow lateral moves that don't improve the score but
    may explore different regions of the solution space.
    """

    def should_accept(self, proposal: CandidateProposal, state: GEPAState) -> bool:
        old_sum = sum(proposal.subsample_scores_before or [])
        new_sum = sum(proposal.subsample_scores_after or [])
        return new_sum >= old_sum
