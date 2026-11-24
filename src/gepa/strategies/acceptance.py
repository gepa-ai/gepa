# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import math
from typing import Protocol, runtime_checkable

from gepa.core.state import GEPAState
from gepa.proposer.base import CandidateProposal

# Default tolerances for :func:`math.isclose` when comparing sums of float scores.
DEFAULT_REL_TOL = 1e-9
DEFAULT_ABS_TOL = 1e-9


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

    Implementations may additionally define an optional
    ``reject_reason(proposal, state) -> str`` method; when present, the engine
    uses it to produce an informative log message for rejected proposals
    instead of the generic fallback.

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

    Sums within ``rel_tol`` / ``abs_tol`` (see :func:`math.isclose`) count as ties, so float
    rounding in the sums cannot turn a tie into an improvement. Set both to ``0`` for exact
    comparison.
    """

    def __init__(self, rel_tol: float = DEFAULT_REL_TOL, abs_tol: float = DEFAULT_ABS_TOL):
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol

    def should_accept(self, proposal: CandidateProposal, state: GEPAState) -> bool:
        old_sum = sum(proposal.subsample_scores_before or [])
        new_sum = sum(proposal.subsample_scores_after or [])
        return new_sum > old_sum and not math.isclose(new_sum, old_sum, rel_tol=self.rel_tol, abs_tol=self.abs_tol)


class ImprovementOrEqualAcceptance:
    """Accept if the sum of new subsample scores is greater than or equal to the old sum.

    Useful when you want to allow lateral moves that don't improve the score but
    may explore different regions of the solution space.

    Sums within ``rel_tol`` / ``abs_tol`` (see :func:`math.isclose`) count as ties, so float
    rounding in the sums cannot turn a tie into a rejection. Set both to ``0`` for exact
    comparison.
    """

    def __init__(self, rel_tol: float = DEFAULT_REL_TOL, abs_tol: float = DEFAULT_ABS_TOL):
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol

    def should_accept(self, proposal: CandidateProposal, state: GEPAState) -> bool:
        old_sum = sum(proposal.subsample_scores_before or [])
        new_sum = sum(proposal.subsample_scores_after or [])
        return new_sum >= old_sum or math.isclose(new_sum, old_sum, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
