# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Selection strategies for filtering proposals after evaluation."""

from typing import Protocol

from gepa.core.state import GEPAState
from gepa.proposer.base import CandidateProposal
from gepa.strategies.acceptance import AcceptanceCriterion


class SelectionStrategy(Protocol):
    """Filters proposals after subsample evaluation.

    The default (``AllImprovements``) accepts every proposal that passes
    the acceptance criterion, matching GEPA's original behavior.
    
    Contract: ``select`` must return proposals FROM the given list (the same
    objects, not copies) — the engine ignores foreign objects, deduplicates
    identical candidates, and reports everything unselected as rejected. The
    acceptance criterion passed in is memoized per batch: consulting it any
    number of times costs exactly one underlying judgement per proposal.
    """

    def select(
        self,
        proposals: list[CandidateProposal],
        state: GEPAState,
        acceptance_criterion: AcceptanceCriterion,
    ) -> list[CandidateProposal]: ...


class AllImprovements(SelectionStrategy):
    """Accept all proposals that pass the acceptance criterion."""

    def select(
        self,
        proposals: list[CandidateProposal],
        state: GEPAState,
        acceptance_criterion: AcceptanceCriterion,
    ) -> list[CandidateProposal]:
        return [p for p in proposals if acceptance_criterion.should_accept(p, state)]


class BestImprovement(SelectionStrategy):
    """Accept only the single best-improving proposal."""

    def select(
        self,
        proposals: list[CandidateProposal],
        state: GEPAState,
        acceptance_criterion: AcceptanceCriterion,
    ) -> list[CandidateProposal]:
        best: CandidateProposal | None = None
        best_imp = float("-inf")
        for p in proposals:
            if not acceptance_criterion.should_accept(p, state):
                continue
            imp = sum(p.subsample_scores_after or []) - sum(p.subsample_scores_before or [])
            if imp > best_imp:
                best, best_imp = p, imp
        return [best] if best else []


class TopKImprovements(SelectionStrategy):
    """Accept top-K proposals by improvement margin."""

    def __init__(self, k: int):
        self.k = k

    def select(
        self,
        proposals: list[CandidateProposal],
        state: GEPAState,
        acceptance_criterion: AcceptanceCriterion,
    ) -> list[CandidateProposal]:
        passing = [
            (sum(p.subsample_scores_after or []) - sum(p.subsample_scores_before or []), p)
            for p in proposals
            if acceptance_criterion.should_accept(p, state)
        ]
        passing.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in passing[: self.k]]
