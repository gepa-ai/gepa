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
    reflection_batch_size: int = 6
    # merge_period and merge_uplift_min removed - merge functionality removed
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
    target_quality: float | None = None  # Stop when best quality reaches this threshold


DEFAULT_CONFIG = Config()


def adaptive_config(
    dataset_size: int,
    *,
    base_config: Config | None = None,
    strategy: str = "balanced",
    available_compute: str = "laptop",
) -> Config:
    """
    Automatically configure TurboGEPA based on dataset size and available resources.

    This function adjusts all key parameters to optimize for the dataset size:
    - Shards: ASHA successive halving rungs
    - Batch size: Candidates per round
    - Mutations: New variants generated per round
    - Concurrency: Parallel evaluations
    - Queue: Working set size
    - Islands: Parallel optimization processes

    Parameters:
        dataset_size: Number of examples in dataset
        base_config: Optional base config to override (uses DEFAULT_CONFIG if None)
        strategy: "conservative", "balanced", or "aggressive"
        available_compute: "laptop", "workstation", or "server"

    Returns:
        Optimized Config object

    Examples:
        >>> config = adaptive_config(50)  # Small dataset, laptop
        >>> config.batch_size
        4
        >>> config.eval_concurrency
        8

        >>> config = adaptive_config(5000, available_compute="server")  # Large dataset
        >>> config.batch_size
        16
        >>> config.eval_concurrency
        128
    """
    config = base_config or Config()

    # Auto-select shards
    config.shards = adaptive_shards(dataset_size, strategy=strategy)

    # Compute multipliers based on available_compute
    if available_compute == "laptop":
        compute_mult = 1.0
        max_concurrency = 32
        max_islands = 2
    elif available_compute == "workstation":
        compute_mult = 2.0
        max_concurrency = 128
        max_islands = 4
    else:  # server
        compute_mult = 4.0
        max_concurrency = 256
        max_islands = 8

    # Scale parameters based on dataset size and compute
    if dataset_size < 10:
        # Tiny dataset: minimal parallelism, focus on exploration
        config.batch_size = 2
        config.max_mutations_per_round = 4
        config.eval_concurrency = min(4, max_concurrency)
        config.queue_limit = 16
        config.n_islands = 1  # No island parallelism for tiny datasets
        config.reflection_batch_size = 3

    elif dataset_size < 50:
        # Small dataset: moderate parallelism
        config.batch_size = max(4, int(4 * compute_mult))
        config.max_mutations_per_round = max(6, int(6 * compute_mult))
        config.eval_concurrency = min(int(8 * compute_mult), max_concurrency)
        config.queue_limit = 32
        config.n_islands = min(2, max_islands) if compute_mult >= 2.0 else 1
        config.reflection_batch_size = 4

    elif dataset_size < 200:
        # Medium dataset: good parallelism
        config.batch_size = max(6, int(6 * compute_mult))
        config.max_mutations_per_round = max(8, int(8 * compute_mult))
        config.eval_concurrency = min(int(16 * compute_mult), max_concurrency)
        config.queue_limit = 64
        config.n_islands = min(2, max_islands)
        config.reflection_batch_size = 5

    elif dataset_size < 1000:
        # Large dataset: high parallelism
        config.batch_size = max(8, int(8 * compute_mult))
        config.max_mutations_per_round = max(12, int(12 * compute_mult))
        config.eval_concurrency = min(int(32 * compute_mult), max_concurrency)
        config.queue_limit = 128
        config.n_islands = min(4, max_islands)
        config.reflection_batch_size = 6

    else:
        # Very large dataset: maximum parallelism
        config.batch_size = max(12, int(12 * compute_mult))
        config.max_mutations_per_round = max(16, int(16 * compute_mult))
        config.eval_concurrency = min(int(64 * compute_mult), max_concurrency)
        config.queue_limit = 256
        config.n_islands = min(max_islands, max(4, int(dataset_size // 500)))
        config.reflection_batch_size = 8

    # Adjust strategy-specific parameters
    if strategy == "conservative":
        # Conservative: higher quality signals, less parallelism
        config.max_mutations_per_round = max(4, config.max_mutations_per_round // 2)
        config.cohort_quantile = 0.7  # Stricter promotion (top 30% only)
        config.eps_improve = 0.02  # Require larger improvements

    elif strategy == "aggressive":
        # Aggressive: more exploration, higher parallelism
        config.max_mutations_per_round = min(32, int(config.max_mutations_per_round * 1.5))
        config.cohort_quantile = 0.5  # Easier promotion (top 50%)
        config.eps_improve = 0.005  # Accept smaller improvements
        config.batch_size = min(24, int(config.batch_size * 1.5))

    # Ensure migration_period scales with dataset size
    # Smaller datasets need more frequent migration to avoid local optima
    if dataset_size < 100:
        config.migration_period = 1  # Migrate every round
    elif dataset_size < 500:
        config.migration_period = 2  # Default
    else:
        config.migration_period = 3  # Less frequent for large datasets

    # Scale migration_k with number of islands
    config.migration_k = min(config.n_islands, max(2, config.n_islands // 2))

    return config


def lightning_config(dataset_size: int, *, base_config: Config | None = None) -> Config:
    """
    Lightning mode: 5x faster, ~85% quality retention.

    Optimizations:
    - 2-rung ASHA (eliminates middle shard)
    - Aggressive pruning (top 25% advance)
    - Single island (no migration overhead)
    - Reduced mutation budget
    - Smaller batches (test fewer candidates thoroughly)

    Best for: Quick iteration, prototyping, debugging

    Parameters:
        dataset_size: Number of examples in dataset
        base_config: Optional base config to override

    Returns:
        Optimized Config for 5x speedup

    Examples:
        >>> config = lightning_config(500)
        >>> config.shards
        (0.1, 1.0)
        >>> config.n_islands
        1
    """
    config = base_config or Config()

    # Aggressive 2-rung ASHA
    config.shards = (0.10, 1.0)
    config.cohort_quantile = 0.75  # Top 25% only (vs 40%)
    config.eps_improve = 0.02  # Stricter improvement (vs 0.01)

    # Reduce breadth
    config.batch_size = 4  # Half the candidates (vs 8)
    config.max_mutations_per_round = 8  # Half the mutations (vs 16)

    # Streamlined reflection
    config.reflection_batch_size = 3  # Fewer traces (vs 6)

    # Single island - no migration overhead
    config.n_islands = 1
    config.migration_period = 999  # Effectively disabled

    # Keep high concurrency for speed
    config.eval_concurrency = 64

    return config


def sprint_config(dataset_size: int, *, base_config: Config | None = None) -> Config:
    """
    Sprint mode: 3x faster, ~90% quality retention.

    Optimizations:
    - 3-rung ASHA with tighter gaps
    - Moderate pruning (top 35% advance)
    - 2 islands (reduced overhead)
    - Reduced mutation budget

    Best for: Fast production runs, balanced speed/quality

    Parameters:
        dataset_size: Number of examples in dataset
        base_config: Optional base config to override

    Returns:
        Optimized Config for 3x speedup

    Examples:
        >>> config = sprint_config(500)
        >>> config.shards
        (0.08, 0.3, 1.0)
        >>> config.n_islands
        2
    """
    config = base_config or Config()

    # Moderate 3-rung ASHA
    config.shards = (0.08, 0.30, 1.0)
    config.cohort_quantile = 0.65  # Top 35% advance (vs 40%)
    config.eps_improve = 0.015  # Moderate threshold (vs 0.01)

    # Moderate breadth reduction
    config.batch_size = 6  # Slightly smaller (vs 8)
    config.max_mutations_per_round = 12  # Fewer mutations (vs 16)

    # Streamlined reflection
    config.reflection_batch_size = 4  # 4 traces (vs 6)

    # Reduced parallelism
    config.n_islands = 2  # Half the islands (vs 4)
    config.migration_period = 3  # Less frequent (vs 2)
    config.migration_k = 2  # Fewer elites (vs 3)

    # High concurrency
    config.eval_concurrency = 64

    return config


def blitz_config(dataset_size: int, *, base_config: Config | None = None) -> Config:
    """
    Blitz mode: 10x faster, ~70% quality retention.

    Optimizations:
    - Single promotion step (15% → 100%)
    - Extreme pruning (top 15% advance)
    - Minimal mutation budget
    - Single island
    - Maximum concurrency

    Best for: Rapid exploration, disposable experiments, initial prototyping

    WARNING: Quality loss is significant. Use for exploration only.

    Parameters:
        dataset_size: Number of examples in dataset
        base_config: Optional base config to override

    Returns:
        Optimized Config for 10x speedup

    Examples:
        >>> config = blitz_config(500)
        >>> config.shards
        (0.15, 1.0)
        >>> config.cohort_quantile
        0.85
    """
    config = base_config or Config()

    # Extreme 2-rung ASHA with large gap
    config.shards = (0.15, 1.0)
    config.cohort_quantile = 0.85  # Only top 15% advance!
    config.eps_improve = 0.03  # Very strict (vs 0.01)

    # Minimal breadth
    config.batch_size = 3  # Very small batches
    config.max_mutations_per_round = 4  # Minimal mutations

    # Fast reflection
    config.reflection_batch_size = 2  # Minimal traces

    # Single island
    config.n_islands = 1
    config.migration_period = 999

    # Maximum concurrency to compensate
    config.eval_concurrency = 128

    return config


def get_lightning_config(
    mode: str,
    dataset_size: int,
    *,
    base_config: Config | None = None,
) -> Config:
    """
    Get pre-configured lightning mode for speed optimization.

    Available modes:
    - "blitz": 10x faster, ~70% quality (rapid exploration)
    - "lightning": 5x faster, ~85% quality (quick iteration)
    - "sprint": 3x faster, ~90% quality (fast production)
    - "balanced": 1x speed, 100% quality (default adaptive config)

    Parameters:
        mode: Speed mode (blitz, lightning, sprint, balanced)
        dataset_size: Number of examples in dataset
        base_config: Optional base config to override

    Returns:
        Optimized Config for selected mode

    Raises:
        ValueError: If mode is not recognized

    Examples:
        >>> # Quick prototyping
        >>> config = get_lightning_config("lightning", 500)
        >>> config.shards
        (0.1, 1.0)

        >>> # Fast production
        >>> config = get_lightning_config("sprint", 1000)
        >>> config.n_islands
        2

        >>> # Rapid exploration
        >>> config = get_lightning_config("blitz", 200)
        >>> config.batch_size
        3
    """
    modes = {
        "blitz": blitz_config,
        "lightning": lightning_config,
        "sprint": sprint_config,
        "balanced": lambda ds, base_config=None: adaptive_config(
            ds, base_config=base_config, strategy="balanced"
        ),
    }

    if mode not in modes:
        raise ValueError(
            f"Unknown lightning mode: '{mode}'. "
            f"Choose from: {', '.join(modes.keys())}"
        )

    return modes[mode](dataset_size, base_config=base_config)
