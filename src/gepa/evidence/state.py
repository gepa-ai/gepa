"""Persistent comparison state; only fully balanced rounds enter inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gepa.evidence.config import SequentialEvidenceConfig, finite_number
from gepa.evidence.observations import Observation, ObservationId, candidate_identity
from gepa.evidence.statistics import RunningMoments, alpha_at, intersect_interval, mean_interval, weighted_difference

TERMINAL = {
    "supported_improvement",
    "supported_regression",
    "budget_exhausted",
    "inconclusive",
    "heuristic",
    "validation_deferred",
    "promoted",
    "exploration",
    "not_selected",
    "criterion_rejected",
}


@dataclass
class ComparisonState:
    comparison_id: str
    index: int
    parent: dict[str, str]
    child: dict[str, str]
    parent_ids: list[int]
    example_ids: list[Any]
    example_keys: list[str]
    weights: tuple[float, ...]
    support: str
    status: str = "running"
    reason: str = "collecting balanced fresh executions"
    completed_rounds: int = 0
    metric_calls: int = 0
    lower: float = 0.0
    upper: float = 0.0
    observations: dict[str, Observation] = field(default_factory=dict)
    moments_parent: list[RunningMoments] = field(default_factory=list)
    moments_child: list[RunningMoments] = field(default_factory=list)
    intervals_parent: list[tuple[float, float]] = field(default_factory=list)
    intervals_child: list[tuple[float, float]] = field(default_factory=list)
    selected: bool = False
    selection_label: str | None = None
    validation_candidate_idx: int | None = None
    diagnostics_before: list[float] | None = None
    diagnostics_after: list[float] | None = None
    iteration_id: str | None = None

    @property
    def eligible(self) -> bool:
        return self.status == "supported_improvement" and self.support != "none"

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL

    def initialize_intervals(self, config: SequentialEvidenceConfig) -> None:
        count = len(self.example_keys)
        self.moments_parent = [RunningMoments() for _ in range(count)]
        self.moments_child = [RunningMoments() for _ in range(count)]
        self.intervals_parent = [(config.score_min, config.score_max) for _ in range(count)]
        self.intervals_child = [(config.score_min, config.score_max) for _ in range(count)]
        self.lower, self.upper = config.score_min - config.score_max, config.score_max - config.score_min

    def reserve_round(self, config: SequentialEvidenceConfig) -> tuple[list[ObservationId], list[ObservationId]]:
        if self.terminal:
            raise ValueError("terminal comparisons cannot schedule new observations")
        trial = f"{self.comparison_id}:{self.completed_rounds + 1}"
        sides = []
        for side, candidate in (("parent", self.parent), ("child", self.child)):
            identities = [
                ObservationId(config.evaluator_fingerprint, candidate_identity(candidate), "train", key, trial, side)
                for key in self.example_keys
            ]
            for identity in identities:
                if identity.key in self.observations:
                    raise ValueError("duplicate trial must be recovered, never dispatched again")
                self.observations[identity.key] = Observation(identity, "reserved")
            sides.append(identities)
        return sides[0], sides[1]

    def complete_side(
        self, identities: list[ObservationId], scores: list[float], config: SequentialEvidenceConfig
    ) -> None:
        if len(identities) != len(scores):
            raise ValueError("missing observation score")
        validated = [config.validate_score(score) for score in scores]
        for identity, score in zip(identities, validated, strict=True):
            previous = self.observations[identity.key]
            completed = Observation(identity, "completed", score)
            if previous.status == "completed":
                if previous != completed:
                    raise ValueError("conflicting completed observation replay")
                continue
            if previous.status != "reserved":
                raise ValueError("failed observations cannot be selectively completed")
            self.observations[identity.key] = completed

    def fail_round(self, identities: list[ObservationId], reason: str) -> None:
        for identity in identities:
            previous = self.observations[identity.key]
            if previous.status == "reserved":
                self.observations[identity.key] = Observation(identity, "failed", failure=reason)
        self.status = "inconclusive"
        self.reason = reason

    def finish_round(
        self, parent_ids: list[ObservationId], child_ids: list[ObservationId], config: SequentialEvidenceConfig
    ) -> None:
        if self.terminal:
            raise ValueError("incomplete or terminal comparisons cannot accumulate statistical evidence")
        for identities in (parent_ids, child_ids):
            if any(self.observations[key.key].status != "completed" for key in identities):
                raise ValueError("balanced complete rounds are required")
        for identities, moments, intervals in (
            (parent_ids, self.moments_parent, self.intervals_parent),
            (child_ids, self.moments_child, self.intervals_child),
        ):
            for stratum, identity in enumerate(identities):
                score = self.observations[identity.key].score
                assert score is not None
                moments[stratum].add(score)
                allocation = alpha_at(config.alpha, self.index, stratum, moments[stratum].count, len(self.weights))
                interval = mean_interval(moments[stratum], config.score_min, config.score_max, allocation)
                intervals[stratum] = intersect_interval(intervals[stratum], interval)
        self.completed_rounds += 1
        self.lower, self.upper = weighted_difference(self.intervals_parent, self.intervals_child, self.weights)
        if self.completed_rounds < config.min_rounds or not config.confidence_supported:
            return
        if self.lower > config.effect_margin:
            self.status = "supported_improvement"
            self.reason = "simultaneous lower improvement bound exceeds effect margin"
        elif self.upper < -config.effect_margin:
            self.status = "supported_regression"
            self.reason = "simultaneous upper improvement bound is below negative effect margin"

    def summary(self) -> dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "index": self.index,
            "parent_identity": candidate_identity(self.parent),
            "child_identity": candidate_identity(self.child),
            "parent_ids": self.parent_ids,
            "example_ids": self.example_ids,
            "weights": list(self.weights),
            "status": self.status,
            "reason": self.reason,
            "support": self.support,
            "completed_rounds": self.completed_rounds,
            "metric_calls": self.metric_calls,
            "lower": self.lower,
            "upper": self.upper,
            "selected": self.selected,
            "selection_label": self.selection_label,
            "validation_candidate_idx": self.validation_candidate_idx,
            "observation_counts": {
                status: sum(o.status == status for o in self.observations.values())
                for status in ("reserved", "completed", "failed", "abandoned")
            },
            "paired_randomness": False,
            "validation_confidence_calibrated": False,
        }

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.__dict__)
        data["weights"] = list(self.weights)
        data["observations"] = [o.to_dict() for o in self.observations.values()]
        data["moments_parent"] = [m.to_dict() for m in self.moments_parent]
        data["moments_child"] = [m.to_dict() for m in self.moments_child]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any], config: SequentialEvidenceConfig) -> ComparisonState:
        fields = dict(data)
        observations = [Observation.from_dict(o) for o in fields.pop("observations")]
        fields["weights"] = tuple(fields["weights"])
        for name in ("moments_parent", "moments_child"):
            fields[name] = [RunningMoments(**item) for item in fields[name]]
        for name in ("intervals_parent", "intervals_child"):
            fields[name] = [tuple(item) for item in fields[name]]
        result = cls(**fields)
        result.observations = {o.identity.key: o for o in observations}
        if len(result.observations) != len(observations):
            raise ValueError("duplicate saved observation identities")
        if result.status not in TERMINAL | {"running"}:
            raise ValueError("unknown saved comparison status")
        count = len(result.example_keys)
        if count == 0 or len(set(result.example_keys)) != count:
            raise ValueError("saved comparison requires unique strata")
        if len(result.example_ids) != count or any(
            len(xs) != count
            for xs in (
                result.weights,
                result.moments_parent,
                result.moments_child,
                result.intervals_parent,
                result.intervals_child,
            )
        ):
            raise ValueError("corrupt saved stratum coverage")
        if result.weights != config.weights_for(count):
            raise ValueError("saved stratum weights differ from fixed configuration")
        for observation in observations:
            if observation.identity.evaluator != config.evaluator_fingerprint:
                raise ValueError("saved evaluator fingerprint mismatch")
            if observation.identity.example not in result.example_keys:
                raise ValueError("saved observation has unknown example identity")
            if observation.status == "completed":
                config.validate_score(observation.score)
        if result.completed_rounds < 0 or result.metric_calls < 0:
            raise ValueError("corrupt saved comparison counts")
        for moments in (*result.moments_parent, *result.moments_child):
            if moments.count != result.completed_rounds or moments.m2 < 0:
                raise ValueError("saved moments disagree with completed rounds")
            finite_number(moments.mean, "saved mean")
            finite_number(moments.m2, "saved m2")
        finite_number(result.lower, "saved lower bound")
        finite_number(result.upper, "saved upper bound")
        if result.lower > result.upper:
            raise ValueError("corrupt saved interval")
        if result.status == "running":
            # No selective completion after an interrupted evaluator call,
            # including a fully returned side whose peer outcome is unknown.
            result.status = "inconclusive"
            result.reason = "interrupted comparison has unknown completion; no selective retries"
            for key, observation in result.observations.items():
                if observation.status == "reserved":
                    result.observations[key] = Observation(observation.identity, "abandoned", failure=result.reason)
        return result
