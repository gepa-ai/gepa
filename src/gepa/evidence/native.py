"""Conservative supported-path boundary for enforceable metric reservations."""

from __future__ import annotations

from typing import Any

from gepa.evidence.config import positive_integer
from gepa.proposer.reflective_mutation.reflective_mutation import ReflectiveMutationProposer
from gepa.strategies.batch_sampler import EpochShuffledBatchSampler
from gepa.strategies.candidate_selector import (
    CurrentBestCandidateSelector,
    EpsilonGreedyCandidateSelector,
    ParetoCandidateSelector,
    TopKParetoCandidateSelector,
)
from gepa.strategies.component_selector import AllReflectionComponentSelector, RoundRobinReflectionComponentSelector
from gepa.strategies.eval_policy import FullEvaluationPolicy
from gepa.strategies.proposal_sampling import (
    IndependentSampling,
    PxNSampling,
    SameParentSampling,
    SingleMutationSampling,
)


def validate_native_support(proposer: Any, merge: Any, policy: Any, max_metric_calls: Any) -> None:
    positive_integer(max_metric_calls, "evidence max_metric_calls")
    if merge is not None:
        raise ValueError("sequential evidence does not support merge proposals in this release")
    if policy is not None and type(policy) is not FullEvaluationPolicy:
        raise ValueError("sequential evidence requires the exact native FullEvaluationPolicy")
    if type(proposer) is not ReflectiveMutationProposer or proposer.custom_candidate_proposer is not None:
        raise ValueError(
            "sequential evidence requires the audited native reflective proposer, without custom proposers"
        )
    supported = (
        (proposer.sampling_strategy, (SingleMutationSampling, SameParentSampling, IndependentSampling, PxNSampling)),
        (proposer.batch_sampler, (EpochShuffledBatchSampler,)),
        (
            proposer.candidate_selector,
            (
                CurrentBestCandidateSelector,
                EpsilonGreedyCandidateSelector,
                ParetoCandidateSelector,
                TopKParetoCandidateSelector,
            ),
        ),
        (proposer.module_selector, (AllReflectionComponentSelector, RoundRobinReflectionComponentSelector)),
    )
    for strategy, allowed in supported:
        if type(strategy) not in allowed:
            raise ValueError(
                f"sequential evidence cannot guard unsupported custom proposer strategy {type(strategy).__name__}"
            )
