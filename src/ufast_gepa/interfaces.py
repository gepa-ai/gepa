"""
Core data contracts shared across uFast-GEPA modules.

These dataclasses mirror the artifacts produced and consumed by the
orchestrator loop, allowing the rest of the system to remain loosely coupled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence


@dataclass(frozen=True)
class Candidate:
    """Represents an optimizer candidate (e.g., a prompt string)."""

    text: str
    meta: Dict[str, Any] = field(default_factory=dict)

    def with_meta(self, **updates: Any) -> "Candidate":
        """Return a new candidate with additional metadata merged in."""
        merged = dict(self.meta)
        merged.update(updates)
        return Candidate(text=self.text, meta=merged)

    @property
    def fingerprint(self) -> str:
        """Stable identifier derived from the candidate text and temperature."""
        import hashlib

        # Include temperature in fingerprint so same prompt with different temps
        # are treated as distinct candidates for caching and evaluation
        temp = self.meta.get("temperature", "default")
        key = f"{self.text}::temp={temp}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


@dataclass
class EvalResult:
    """
    Captures evaluation metrics, traces, and coverage for a candidate.

    All objectives are maximized; callers can negate costs upstream.
    """

    objectives: Dict[str, float]
    traces: List[Dict[str, Any]]
    n_examples: int
    shard_fraction: float | None = None
    example_ids: Sequence[str] | None = None

    def objective(self, key: str, default: float | None = None) -> float | None:
        """Convenience accessor for a specific objective value."""
        if default is None:
            return self.objectives[key]
        return self.objectives.get(key, default)

    def merge(self, other: "EvalResult") -> "EvalResult":
        """Combine two evaluation results by summing objectives and traces."""
        combined = dict(self.objectives)
        for key, value in other.objectives.items():
            combined[key] = combined.get(key, 0.0) + value
        traces = list(self.traces)
        traces.extend(other.traces)
        example_ids: List[str] = []
        if self.example_ids:
            example_ids.extend(self.example_ids)
        if other.example_ids:
            example_ids.extend(other.example_ids)
        total_examples = self.n_examples + other.n_examples
        averaged = {k: v / max(total_examples, 1) for k, v in combined.items()}
        return EvalResult(
            objectives=averaged,
            traces=traces,
            n_examples=total_examples,
            shard_fraction=self.shard_fraction,
            example_ids=example_ids,
        )


TraceIterable = Iterable[Dict[str, Any]]


class AsyncEvaluatorProtocol:
    """Structural protocol for async evaluation implementations."""

    async def eval_on_shard(
        self,
        candidate: Candidate,
        example_ids: Sequence[str],
        concurrency: int,
    ) -> EvalResult:  # pragma: no cover - interface definition only
        raise NotImplementedError
