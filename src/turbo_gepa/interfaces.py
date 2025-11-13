"""
Core data contracts shared across TurboGEPA modules.

These dataclasses mirror the artifacts produced and consumed by the
orchestrator loop, allowing the rest of the system to remain loosely coupled.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class Candidate:
    """Represents an optimizer candidate (e.g., a prompt string)."""

    text: str
    meta: dict[str, Any] = field(default_factory=dict)

    def with_meta(self, **updates: Any) -> Candidate:
        """Return a new candidate with additional metadata merged in."""
        merged = dict(self.meta)
        merged.update(updates)
        return Candidate(text=self.text, meta=merged)

    @property
    def fingerprint(self) -> str:
        """Stable identifier derived from canonicalised prompt + metadata."""
        import xxhash

        def _normalize(value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, str):
                return " ".join(value.split())
            if isinstance(value, Candidate):  # pragma: no cover - defensive
                return value.fingerprint
            if isinstance(value, Mapping):
                return {k: _normalize(value[k]) for k in sorted(value)}
            if isinstance(value, (list, tuple)):
                return [_normalize(v) for v in value]
            if isinstance(value, set):
                normalised = [_normalize(v) for v in value]
                return sorted(normalised, key=lambda x: repr(x))
            return value

        # Exclude result-related metadata from fingerprint (these change as candidate is evaluated)
        RESULT_KEYS = {"quality", "quality_shard_fraction", "parent_objectives"}
        identity_meta = {k: v for k, v in self.meta.items() if k not in RESULT_KEYS}

        canonical = {
            "text": _normalize(self.text),
            "meta": _normalize({k: identity_meta[k] for k in sorted(identity_meta)}),
        }

        try:
            payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
        except TypeError:
            # Fallback: stringify non-serialisable objects deterministically
            fallback = {
                "text": canonical["text"],
                "meta": {k: repr(v) for k, v in canonical["meta"].items()}
                if isinstance(canonical["meta"], dict)
                else repr(canonical["meta"]),
            }
            payload = json.dumps(fallback, sort_keys=True, separators=(",", ":")).encode("utf-8")

        return xxhash.xxh3_64_hexdigest(payload)


@dataclass
class EvalResult:
    """
    Captures evaluation metrics, traces, and coverage for a candidate.

    All objectives are maximized; callers can negate costs upstream.
    """

    objectives: dict[str, float]
    traces: list[dict[str, Any]]
    n_examples: int
    shard_fraction: float | None = None
    example_ids: Sequence[str] | None = None
    # Optional: fraction of examples completed on this shard (for live partials)
    coverage_fraction: float | None = None

    def objective(self, key: str, default: float | None = None) -> float | None:
        """Convenience accessor for a specific objective value."""
        if default is None:
            return self.objectives[key]
        return self.objectives.get(key, default)

    def merge(self, other: EvalResult) -> EvalResult:
        """Combine two evaluation results by summing objectives and traces."""
        combined: dict[str, float] = {}
        for key, value in self.objectives.items():
            combined[key] = value * self.n_examples
        for key, value in other.objectives.items():
            combined[key] = combined.get(key, 0.0) + value * other.n_examples
        traces = list(self.traces)
        traces.extend(other.traces)
        example_ids: list[str] = []
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
            coverage_fraction=self.coverage_fraction,
        )


TraceIterable = Iterable[dict[str, Any]]


class AsyncEvaluatorProtocol:
    """Structural protocol for async evaluation implementations."""

    async def eval_on_shard(
        self,
        candidate: Candidate,
        example_ids: Sequence[str],
        concurrency: int,
        *,
        shard_fraction: float | None = None,
        show_progress: bool = False,
        early_stop_fraction: float = 0.9,
        is_final_shard: bool = False,
    ) -> EvalResult:  # pragma: no cover - interface definition only
        raise NotImplementedError
