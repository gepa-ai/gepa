"""
Central configuration knobs for TurboGEPA.

Defaults are intentionally conservative so the system can run on a laptop
without further tuning. Users can override values by passing keyword args
into orchestrator entrypoints or by subclassing ``Config``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


def adaptive_shards(
    dataset_size: int,
    *,
    strategy: str = "balanced",
    min_first_shard_examples: int = 15,
) -> tuple[float, ...]:
    """
    Automatically select optimal shard configuration based on dataset size.

    The function follows the "Rule of 15": the first shard should evaluate at
    least ~15 examples for statistical reliability (±20-22% confidence interval).

    Parameters:
        dataset_size: Number of examples in the validation/training set
        strategy: Optimization strategy - "conservative", "balanced", or "aggressive"
        min_first_shard_examples: Minimum examples for first shard (default: 15)

    Returns:
        Tuple of shard fractions (e.g., (0.05, 0.20, 1.0))

    Strategy descriptions:
        - "conservative": Prioritizes quality over speed, higher confidence at each stage
        - "balanced": Good exploration/exploitation tradeoff (recommended default)
        - "aggressive": Prioritizes speed over quality, tests more candidates

    Examples:
        >>> adaptive_shards(50)  # Small dataset
        (0.30, 1.0)
        >>> adaptive_shards(500)  # Medium dataset
        (0.05, 0.20, 1.0)
        >>> adaptive_shards(5000, strategy="aggressive")  # Large dataset, aggressive
        (0.02, 0.08, 0.25, 1.0)
    """
    if dataset_size <= 0:
        return (1.0,)

    # Calculate target first shard percentage to get min_first_shard_examples
    target_first_pct = min_first_shard_examples / dataset_size

    if strategy == "conservative":
        # Conservative: Higher confidence, fewer candidates tested
        if dataset_size < 50:
            return (0.50, 1.0)
        elif dataset_size < 100:
            return (0.30, 1.0)
        elif dataset_size < 500:
            return (0.15, 0.50, 1.0)
        elif dataset_size < 2000:
            return (0.10, 0.30, 1.0)
        else:
            return (0.05, 0.20, 0.50, 1.0)

    elif strategy == "aggressive":
        # Aggressive: Lower confidence, more candidates tested
        if dataset_size < 50:
            return (0.30, 1.0)  # Still need reasonable signal
        elif dataset_size < 100:
            return (0.20, 1.0)
        elif dataset_size < 500:
            return (0.05, 0.15, 1.0)
        elif dataset_size < 2000:
            return (0.03, 0.12, 1.0)
        else:
            # 4-rung for large datasets
            return (0.02, 0.08, 0.25, 1.0)

    else:  # balanced (default)
        # Balanced: Good tradeoff for most cases
        if dataset_size < 50:
            # Very small: 2-rung, conservative first stage
            first_shard = max(0.30, min(0.50, target_first_pct))
            return (first_shard, 1.0)
        elif dataset_size < 100:
            # Small: 2-rung, moderate first stage
            first_shard = max(0.20, min(0.30, target_first_pct))
            return (first_shard, 1.0)
        elif dataset_size < 500:
            # Medium: 3-rung
            first_shard = max(0.10, min(0.20, target_first_pct))
            return (first_shard, 0.30, 1.0)
        elif dataset_size < 2000:
            # Large: 3-rung, standard ASHA
            first_shard = max(0.05, min(0.10, target_first_pct))
            return (first_shard, 0.20, 1.0)
        else:
            # Very large: 3 or 4-rung depending on size
            if dataset_size < 5000:
                return (0.05, 0.20, 1.0)
            else:
                return (0.03, 0.12, 0.40, 1.0)


@dataclass(slots=True)
class Config:
    """Runtime parameters controlling concurrency, shard sizes, and promotion."""

    eval_concurrency: int = 64
    compression_concurrency: int = 16  # Separate limit for finalization compression
    n_islands: int = 4
    shards: Sequence[float] = field(default_factory=lambda: (0.05, 0.2, 1.0))
    eps_improve: float = 0.01
    cohort_quantile: float = 0.6
    qd_bins_length: int = 8
    qd_bins_bullets: int = 6
    qd_flags: Sequence[str] = field(default_factory=lambda: ("cot", "format", "fewshot"))
    amortized_rate: float = 0.8
    reflection_batch_size: int = 6
    merge_period: int = 3
    merge_uplift_min: float = 0.01
    max_tokens: int = 2048
    prune_delta: float = 0.005
    migration_period: int = 2
    migration_k: int = 3
    cache_path: str = ".turbo_gepa/cache"
    log_path: str = ".turbo_gepa/logs"
    batch_size: int = 8
    queue_limit: int = 128
    promote_objective: str = "quality"
    compression_objective: str = "quality"
    compression_shard_fraction: float = 0.2
    log_summary_interval: int = 10
    max_mutations_per_round: int = 16
    task_lm_temperature: float | None = 1.0
    reflection_lm_temperature: float | None = 1.0


DEFAULT_CONFIG = Config()
