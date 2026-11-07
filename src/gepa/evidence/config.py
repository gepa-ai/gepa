"""Explicit bounded-evaluator contract for sequential candidate comparisons."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal


def positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def canonical_json(value: Any) -> str:
    """Serialize identities without repr(), object addresses or nonfinite numbers."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SequentialEvidenceConfig:
    """Caller declarations, rather than empirically checked noise assumptions.

    ``bounded_independent`` requires each stratum/side to have a fixed mean
    and independent bounded repetitions, conditionally on proposal history.
    ``bounded_martingale`` permits dependent repetitions with that same fixed
    conditional mean. ``heuristic`` spends the same budget but makes no
    confidence-supported acceptance claim. Costs are scorer attempts, separate
    from reflection LM spending. An adapter call must perform no more than
    ``max_calls_per_example * len(batch)`` attempts, including internal retries.
    """

    evaluator_fingerprint: str
    max_comparison_calls: int
    score_min: float = 0.0
    score_max: float = 1.0
    min_rounds: int = 2
    alpha: float = 0.05
    effect_margin: float = 0.0
    max_calls_per_example: int = 1
    assumption: Literal["bounded_independent", "bounded_martingale", "heuristic"] = "bounded_independent"
    unresolved: Literal["reject", "explore"] = "reject"
    stratum_weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.evaluator_fingerprint, str) or not self.evaluator_fingerprint.strip():
            raise ValueError("evaluator_fingerprint must identify evaluator/model/environment/data versions")
        positive_integer(self.max_comparison_calls, "max_comparison_calls")
        positive_integer(self.min_rounds, "min_rounds")
        positive_integer(self.max_calls_per_example, "max_calls_per_example")
        lower = finite_number(self.score_min, "score_min")
        upper = finite_number(self.score_max, "score_max")
        if lower >= upper or not math.isfinite(upper - lower):
            raise ValueError("score_min must be strictly below score_max with finite range")
        if not 0 < finite_number(self.alpha, "alpha") < 1:
            raise ValueError("alpha must be strictly between zero and one")
        if finite_number(self.effect_margin, "effect_margin") < 0:
            raise ValueError("effect_margin must be nonnegative")
        if self.assumption not in ("bounded_independent", "bounded_martingale", "heuristic"):
            raise ValueError("unsupported evidence assumption")
        if self.unresolved not in ("reject", "explore"):
            raise ValueError("unresolved must be reject or explore")
        if self.stratum_weights is not None:
            if not isinstance(self.stratum_weights, tuple) or not self.stratum_weights:
                raise ValueError("stratum_weights must be a nonempty immutable tuple")
            if any(finite_number(w, "weight") <= 0 for w in self.stratum_weights):
                raise ValueError("stratum weights must be positive")
            if not math.isclose(math.fsum(self.stratum_weights), 1.0, rel_tol=0, abs_tol=1e-12):
                raise ValueError("stratum weights must sum to one")

    @property
    def fingerprint(self) -> str:
        return digest(self.to_dict())

    @property
    def confidence_supported(self) -> bool:
        return self.assumption != "heuristic"

    def weights_for(self, count: int) -> tuple[float, ...]:
        positive_integer(count, "stratum count")
        if self.stratum_weights is None:
            return tuple(1.0 / count for _ in range(count))
        if len(self.stratum_weights) != count:
            raise ValueError("stratum_weights must match the frozen proposal positions")
        return self.stratum_weights

    def validate_score(self, value: Any) -> float:
        score = finite_number(value, "score")
        if not self.score_min <= score <= self.score_max:
            raise ValueError(f"score {score} is outside declared bounds [{self.score_min}, {self.score_max}]")
        return score

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SequentialEvidenceConfig:
        data = dict(value)
        if data.get("stratum_weights") is not None:
            data["stratum_weights"] = tuple(data["stratum_weights"])
        return cls(**data)
