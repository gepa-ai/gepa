"""Content identities and replay-safe records for actual evaluator executions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from gepa.evidence.config import digest

Side = Literal["parent", "child"]
ObservationStatus = Literal["reserved", "completed", "failed", "abandoned"]


def candidate_identity(candidate: dict[str, str]) -> str:
    if not candidate or any(not isinstance(k, str) or not isinstance(v, str) for k, v in candidate.items()):
        raise ValueError("evidence candidates must be nonempty string-to-string mappings")
    return digest(candidate)


def example_identity(example_id: Any) -> str:
    """Preserve JSON type identity, including integer versus string IDs."""
    return digest([type(example_id).__name__, example_id])


@dataclass(frozen=True)
class ObservationId:
    evaluator: str
    candidate: str
    split: str
    example: str
    trial: str
    side: Side

    def __post_init__(self) -> None:
        if self.side not in ("parent", "child"):
            raise ValueError("unknown evidence side")
        if any(not isinstance(v, str) or not v for v in asdict(self).values()):
            raise ValueError("observation identity fields must be nonempty strings")

    @property
    def key(self) -> str:
        return digest(asdict(self))


@dataclass(frozen=True)
class Observation:
    identity: ObservationId
    status: ObservationStatus
    score: float | None = None
    metric_calls: int = 0
    failure: str | None = None

    def __post_init__(self) -> None:
        if self.status not in ("reserved", "completed", "failed", "abandoned"):
            raise ValueError("unknown observation status")
        if isinstance(self.metric_calls, bool) or not isinstance(self.metric_calls, int) or self.metric_calls < 0:
            raise ValueError("invalid observation cost")
        if self.status == "completed" and self.score is None:
            raise ValueError("completed observations require a score")
        if self.status != "completed" and self.score is not None:
            raise ValueError("incomplete observations cannot contain evidence scores")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Observation:
        data = dict(value)
        data["identity"] = ObservationId(**data["identity"])
        return cls(**data)


@dataclass(frozen=True)
class TrialContext:
    """Optional adapter replication context; equal seeds require adapter support.

    Implement ``evaluate_replication(batch, candidate, *, trial_context,
    capture_traces)`` to consume it. Without this method ordinary evaluate is
    called freshly; no common-random-number benefit is assumed by the bound.
    """

    comparison_id: str
    round_index: int
    trial_id: str
    side: Side
    example_ids: tuple[str, ...]
    paired_seed: int
