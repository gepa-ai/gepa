# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from collections.abc import Sequence
from math import isfinite
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


class NonRegressionAcceptance:
    """Reject regressions on protected, passing minibatch objectives before applying a base criterion.

    Args:
        base: Criterion to apply after the protected checks pass, such as
            ``StrictImprovementAcceptance()`` or ``ImprovementOrEqualAcceptance()``.
        protected: Nonempty sequence of objective names from
            ``SubsampleEvaluation.objective_scores``. A score of exactly ``1.0``
            means the check passed. Omit a key on rows where it does not apply.

    A candidate is rejected if a protected objective scored ``1.0`` for the
    parent and no longer scores ``1.0`` for the candidate on the same row.
    Missing evaluations, inconsistent lengths, malformed objective rows, and
    nonfinite or nonnumeric protected scores also cause rejection.

    Only the evaluated minibatch is checked, against the proposal's parent.
    This criterion performs no additional evaluations. It does not protect
    unsampled examples or enforce a fixed reference across the search.
    Merge proposals use a separate acceptance path. Custom selection strategies
    must honor the supplied criterion for these checks to take effect.

    Raises:
        ValueError: If ``protected`` is empty, a bare string, or contains an
            empty or nonstring name.
    """

    def __init__(self, base: AcceptanceCriterion, protected: Sequence[str]):
        if (
            isinstance(protected, str | bytes)
            or not protected
            or any(not isinstance(name, str) or not name for name in protected)
        ):
            raise ValueError("protected must be a nonempty sequence of nonempty objective names")
        self.base = base
        self.protected = tuple(protected)

    def should_accept(self, proposal: CandidateProposal, state: GEPAState) -> bool:
        if self._protected_rejection(proposal) is not None:
            return False
        return self.base.should_accept(proposal, state)

    def reject_reason(self, proposal: CandidateProposal, state: GEPAState) -> str:
        """Describe a rejection without evaluating the base criterion again."""
        reason = self._protected_rejection(proposal)
        if reason is not None:
            return reason
        base_reason = getattr(self.base, "reject_reason", None)
        if callable(base_reason):
            return str(base_reason(proposal, state))
        return f"Candidate rejected by base acceptance criterion ({type(self.base).__name__})"

    def _protected_rejection(self, proposal: CandidateProposal) -> str | None:
        before_eval, after_eval = proposal.eval_before, proposal.eval_after
        if before_eval is None or after_eval is None:
            return "Protected checks could not be verified: missing subsample evaluation"
        before, after = before_eval.objective_scores, after_eval.objective_scores
        if before is None or after is None:
            return "Protected checks could not be verified: missing objective_scores"

        size = len(before)
        if not size or len(after) != size:
            return "Protected checks could not be verified: empty or mismatched objective_scores"
        if len(before_eval.scores) != size or len(after_eval.scores) != size:
            return "Protected checks could not be verified: objective_scores do not align with evaluation scores"
        for scores in (proposal.subsample_scores_before, proposal.subsample_scores_after):
            if scores is not None and len(scores) != size:
                return "Protected checks could not be verified: objective_scores do not align with subsample scores"
        if proposal.subsample_indices is not None and len(proposal.subsample_indices) != size:
            return "Protected checks could not be verified: objective_scores do not align with subsample indices"

        for index, (old, new) in enumerate(zip(before, after, strict=True)):
            if not isinstance(old, dict) or not isinstance(new, dict):
                return f"Protected checks could not be verified: invalid objective row at minibatch index {index}"
            for name in self.protected:
                for row in (old, new):
                    if name not in row:
                        continue
                    try:
                        finite = isfinite(row[name])
                    except (TypeError, ValueError, OverflowError):
                        finite = False
                    if not finite:
                        return (
                            f"Protected checks could not be verified: invalid score for {name!r} "
                            f"at minibatch index {index}"
                        )
                if old.get(name) == 1.0 and new.get(name) != 1.0:
                    return f"Protected objective {name!r} regressed at minibatch index {index}"
        return None
