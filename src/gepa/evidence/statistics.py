"""Conservative simultaneous Hoeffding intervals, without distribution fitting.

For each comparison c>=1, stratum s in [0,S), side and sample count n>=1,
allocate delta = alpha / (2 S c(c+1) n(n+1)). The two factors telescope,
so a union bound over all these intervals spends at most alpha. Each interval
uses P(|mean-mu| >= r) <= 2 exp(-2 n r^2 / width^2). Intersections remain
simultaneously valid. See Hoeffding (1963), doi:10.1080/01621459.1963.10500830.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from gepa.evidence.config import finite_number, positive_integer


@dataclass
class RunningMoments:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, value: float) -> None:
        value = finite_number(value, "observation")
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    @property
    def variance(self) -> float:
        return max(0.0, self.m2 / (self.count - 1)) if self.count > 1 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def alpha_at(alpha: float, comparison: int, stratum: int, count: int, strata: int) -> float:
    positive_integer(comparison, "comparison")
    positive_integer(count, "count")
    positive_integer(strata, "strata")
    if not 0 < alpha < 1 or not 0 <= stratum < strata:
        raise ValueError("invalid alpha or stratum")
    return alpha / (2 * strata * comparison * (comparison + 1) * count * (count + 1))


def log_alpha_at(alpha: float, comparison: int, stratum: int, count: int, strata: int) -> float:
    # Validate using the same public allocation oracle; compute its logarithm
    # directly so valid subnormal error budgets never become log(0).
    alpha_at(alpha, comparison, stratum, count, strata)
    return (
        math.log(alpha)
        - math.log(2)
        - math.log(strata)
        - math.log(comparison)
        - math.log(comparison + 1)
        - math.log(count)
        - math.log(count + 1)
    )


def mean_interval(moments: RunningMoments, lower: float, upper: float, delta: float) -> tuple[float, float]:
    if lower >= upper or not 0 < delta < 1:
        raise ValueError("invalid bounds or error allocation")
    return mean_interval_log(moments, lower, upper, math.log(delta))


def mean_interval_log(moments: RunningMoments, lower: float, upper: float, log_delta: float) -> tuple[float, float]:
    if lower >= upper or not math.isfinite(log_delta) or log_delta >= 0:
        raise ValueError("invalid bounds or log error allocation")
    if moments.count == 0:
        return lower, upper
    # Subtract logs rather than forming 2/delta, which can overflow at tiny alpha.
    radius = (upper - lower) * math.sqrt((math.log(2) - log_delta) / (2 * moments.count))
    return max(lower, moments.mean - radius), min(upper, moments.mean + radius)


def intersect_interval(old: tuple[float, float], new: tuple[float, float]) -> tuple[float, float]:
    lower, upper = max(old[0], new[0]), min(old[1], new[1])
    if lower > upper:
        raise ValueError("inconsistent simultaneous intervals; evidence is inconclusive")
    return lower, upper


def weighted_difference(
    parent: Sequence[tuple[float, float]], child: Sequence[tuple[float, float]], weights: Sequence[float]
) -> tuple[float, float]:
    if not parent or len(parent) != len(child) or len(parent) != len(weights):
        raise ValueError("both sides must have all fixed-weight strata")
    if any(not math.isfinite(w) or w <= 0 for w in weights) or not math.isclose(math.fsum(weights), 1, abs_tol=1e-12):
        raise ValueError("weights must be positive and sum to one")
    for bounds in (*parent, *child):
        if not all(math.isfinite(x) for x in bounds) or bounds[0] > bounds[1]:
            raise ValueError("invalid stratum interval")
    return (
        math.fsum(w * (c[0] - p[1]) for p, c, w in zip(parent, child, weights, strict=True)),
        math.fsum(w * (c[1] - p[0]) for p, c, w in zip(parent, child, weights, strict=True)),
    )
