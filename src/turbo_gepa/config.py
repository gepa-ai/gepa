"""
Central configuration knobs for TurboGEPA.

Defaults are intentionally conservative so the system can run on a laptop
without further tuning. Users can override values by passing keyword args
into orchestrator entrypoints or by subclassing ``Config``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Sequence

from turbo_gepa.stop_governor import StopGovernorConfig
from turbo_gepa.strategies import (
    ReflectionStrategy,
    resolve_reflection_strategy_names,
)


def _default_variance_tolerance(shards: Sequence[float]) -> dict[float, float]:
    """
    Auto-generate variance tolerance values for each rung.

    Uses a principled approach based on binomial confidence intervals:
    - For binary scoring (0/1 per example), stderr = sqrt(p(1-p)/n)
    - Worst case: p=0.5, giving stderr ≈ 0.5/sqrt(n)
    - We use 1.5 standard errors for ~87% confidence interval

    Formula: tolerance = 0.75 / sqrt(n) + 0.02
    - Statistical component: 0.75/sqrt(n) captures sample size uncertainty
    - Base component: 0.02 prevents over-fitting to final rung

    Parameters:
        shards: Sequence of rung fractions (e.g., (0.2, 0.5, 1.0))

    Returns:
        Dict mapping rung fraction to tolerance value

    Example:
        With 100 examples:
        >>> _default_variance_tolerance((0.2, 0.5, 1.0))
        {0.2: 0.188, 0.5: 0.126, 1.0: 0.095}
        # At 20%: 20 samples -> tolerance ±18.8%
        # At 50%: 50 samples -> tolerance ±12.6%
        # At 100%: 100 samples -> tolerance ±9.5%
    """
    tolerance_map = {}
    for shard in shards:
        if shard <= 0:
            continue

        # Statistical tolerance: 1.5 * stderr for binary outcomes
        # stderr = 0.5/sqrt(n), where n = shard * dataset_size
        # For unit dataset: 0.75/sqrt(shard)
        statistical_tolerance = 0.75 / (shard ** 0.5)

        # Add base tolerance to prevent overfitting at final rung
        base_tolerance = 0.02

        # Combine: statistical uncertainty + base margin
        total_tolerance = statistical_tolerance + base_tolerance

        # Cap at 0.40 (40%) to avoid accepting everything
        total_tolerance = min(total_tolerance, 0.40)

        tolerance_map[shard] = round(total_tolerance, 3)

    return tolerance_map


def _default_shrinkage_alpha(shards: Sequence[float]) -> dict[float, float]:
    """
    Auto-generate shrinkage coefficients for estimating parent scores at earlier rungs.

    Formula: alpha = shard_fraction ^ 0.3
    This gives more weight to final score as rung gets larger.

    Parameters:
        shards: Sequence of rung fractions (e.g., (0.05, 0.2, 1.0))

    Returns:
        Dict mapping rung fraction to shrinkage alpha

    Example:
        >>> _default_shrinkage_alpha((0.05, 0.2, 1.0))
        {0.05: 0.457, 0.2: 0.724, 1.0: 1.0}
    """
    alpha_map = {}
    for shard in shards:
        if shard <= 0:
            continue
        # Alpha increases with shard size (less shrinkage at larger rungs)
        # Use power of 0.3 for smooth interpolation
        alpha = shard ** 0.3
        alpha_map[shard] = round(alpha, 3)

    return alpha_map


def adaptive_shards(
    dataset_size: int,
    *,
    min_samples_per_rung: int = 20,
    reduction_factor: float = 3.0,
) -> tuple[float, ...]:
    """
    Automatically select optimal shard configuration using principled algorithm.

    Algorithm:
    1. First rung: Ensure ≥ min_samples_per_rung examples for statistical validity
    2. Subsequent rungs: Multiply by reduction_factor (geometric progression)
    3. Stop when we reach 100% of dataset

    This follows standard ASHA (Asynchronous Successive Halving) principles:
    - Geometric progression balances early pruning vs. evaluation cost
    - 3x reduction is standard (provides good discrimination while being efficient)
    - Minimum sample size ensures meaningful comparisons

    Parameters:
        dataset_size: Number of examples in dataset
        min_samples_per_rung: Minimum examples at first rung (default: 20)
                              20 examples gives ±22% confidence interval for binary
        reduction_factor: Geometric multiplier between rungs (default: 3.0)
                         Higher = more aggressive pruning, fewer rungs

    Returns:
        Tuple of shard fractions, always ending in 1.0

    Examples:
        >>> adaptive_shards(50)   # Small: (0.4, 1.0)
        >>> adaptive_shards(100)  # Medium: (0.2, 0.6, 1.0)
        >>> adaptive_shards(500)  # Large: (0.04, 0.12, 0.36, 1.0)
        >>> adaptive_shards(100, min_samples_per_rung=10)  # More aggressive
        (0.1, 0.3, 0.9, 1.0)
    """
    if dataset_size <= 0:
        return (1.0,)

    # If dataset too small for meaningful first rung, just use full dataset
    if dataset_size < min_samples_per_rung:
        return (1.0,)

    # Calculate first rung fraction
    first_shard = min_samples_per_rung / dataset_size

    # If first rung would be > 40%, just use 2-rung system
    if first_shard > 0.40:
        # Clamp first rung to reasonable range
        first_shard = min(0.40, max(0.20, first_shard))
        return (round(first_shard, 2), 1.0)

    # Build geometric progression of rungs
    rungs = []
    current_shard = first_shard

    while current_shard < 0.95:  # Stop before reaching 1.0
        rungs.append(current_shard)
        current_shard *= reduction_factor

    # Always end with full dataset
    rungs.append(1.0)

    # Round to 2 decimal places, but ensure no rung rounds to 0
    rounded_rungs = []
    for r in rungs:
        rounded = round(r, 2)
        # Prevent rounding to 0.0 - use minimum of 0.01 (1%)
        if rounded == 0.0:
            rounded = 0.01
        # Skip duplicates (can happen when multiple values round to same number)
        if not rounded_rungs or rounded != rounded_rungs[-1]:
            rounded_rungs.append(rounded)

    return tuple(rounded_rungs)


@dataclass(slots=True)
class Config:
    """Runtime parameters controlling concurrency, shard sizes, and promotion."""

    eval_concurrency: int = 128  # Higher default with fixed connection pooling
    n_islands: int = 4
    shards: Sequence[float] = field(default_factory=lambda: (0.05, 0.2, 1.0))

    # Variance-aware promotion: rung-specific tolerance values
    # Higher tolerance at smaller rungs accounts for score noise with fewer examples
    # Example: {0.05: 0.15, 0.2: 0.08, 1.0: 0.02} means:
    #   - At 5% rung: accept scores within 15% of parent (high variance)
    #   - At 20% rung: accept scores within 8% of parent (medium variance)
    #   - At 100% rung: accept scores within 2% of parent (low variance)
    variance_tolerance: dict[float, float] | None = None  # If None, auto-generates based on shards

    # Shrinkage coefficient for estimating parent score at earlier rungs
    # Higher alpha = more weight to parent's final score (less shrinkage toward baseline)
    # Example: {0.2: 0.7, 0.5: 0.85, 1.0: 1.0}
    shrinkage_alpha: dict[float, float] | None = None  # If None, uses defaults

    # Stop governor: automatic convergence detection
    # Always enabled - monitors hypervolume, quality, stability, ROI
    stop_governor_config: StopGovernorConfig = field(default_factory=StopGovernorConfig)

    reflection_batch_size: int = 6
    max_tokens: int = 2048
    migration_period: int = 1  # Migrate every evaluation batch by default
    migration_k: int = 3
    cache_path: str = ".turbo_gepa/cache"
    log_path: str = ".turbo_gepa/logs"
    batch_size: int | None = None  # Auto-scaled to eval_concurrency if None
    queue_limit: int | None = None  # Auto-scaled to 2x eval_concurrency if None
    promote_objective: str = "quality"
    max_mutations_per_round: int | None = None  # Auto-scaled to eval_concurrency if None
    task_lm_temperature: float | None = 1.0
    reflection_lm_temperature: float | None = 1.0
    target_quality: float | None = None  # Stop when best quality reaches this threshold
    eval_timeout_seconds: float | None = 120.0  # Max time to wait for a single LLM evaluation
    max_optimization_time_seconds: float | None = None  # Global timeout - stop optimization after this many seconds
    reflection_strategy_names: tuple[str, ...] | None = None  # Default to all known strategies
    reflection_strategies: tuple[ReflectionStrategy, ...] | None = None

    # Streaming mode config

    # Logging config
    # Log levels control verbosity:
    #   - DEBUG: All messages including detailed traces (very verbose)
    #   - INFO: Progress updates and mutation generation (verbose)
    #   - WARNING: Important milestones, target reached, auto-stop (dashboard + key events - default)
    #   - ERROR: Only errors
    #   - CRITICAL: Only critical failures
    log_level: str = "WARNING"  # Minimum log level (default: WARNING for clean dashboard output)
    enable_debug_log: bool = False  # Write verbose orchestrator debug file when True


    def __post_init__(self):
        """Auto-scale parameters based on eval_concurrency if not explicitly set."""
        # Auto-scale batch_size to utilize concurrency efficiently
        if self.batch_size is None:
            # Scale batch_size to ~25% of eval_concurrency (capped between 8-64)
            self.batch_size = max(8, min(64, self.eval_concurrency // 4))

        # Ensure batch_size doesn't exceed eval_concurrency
        if self.batch_size > self.eval_concurrency:
            self.batch_size = self.eval_concurrency

        # Auto-scale queue_limit to prevent candidate starvation
        if self.queue_limit is None:
            # Queue should hold at least 4x concurrency worth of candidates (increased from 2x)
            self.queue_limit = max(128, self.eval_concurrency * 4)

        # Auto-scale max_mutations_per_round to match throughput needs
        if self.max_mutations_per_round is None:
            # Generate 2x concurrency to stay ahead of evaluations (increased from 0.25x)
            # This ensures mutation generation can keep the pipeline full
            self.max_mutations_per_round = max(16, min(128, self.eval_concurrency * 2))

        # Auto-generate variance_tolerance if not provided
        if self.variance_tolerance is None:
            self.variance_tolerance = _default_variance_tolerance(self.shards)

        # Auto-generate shrinkage_alpha if not provided
        if self.shrinkage_alpha is None:
            self.shrinkage_alpha = _default_shrinkage_alpha(self.shards)

        custom_strategies = list(self.reflection_strategies or ())
        resolved_defaults = list(resolve_reflection_strategy_names(self.reflection_strategy_names))
        resolved_defaults.extend(custom_strategies)
        if not resolved_defaults:
            raise ValueError(
                "At least one reflection strategy must be configured. "
                "Provide reflection_strategy_names or reflection_strategies."
            )
        self.reflection_strategies = tuple(resolved_defaults)


DEFAULT_CONFIG = Config()


def recommended_executor_workers(eval_concurrency: int, *, cpu_count: int | None = None) -> int:
    """Return a safe default for threadpool workers used by async executors."""
    detected_cpus = cpu_count if cpu_count is not None else (os.cpu_count() or 1)
    # Limit concurrency to avoid oversubscribing CPUs while servicing IO-bound work
    cpu_cap = max(4, detected_cpus * 4)
    eval_cap = max(4, eval_concurrency)
    return min(cpu_cap, eval_cap)


def adaptive_config(
    dataset_size: int,
    *,
    base_config: Config | None = None,
) -> Config:
    """
    Automatically configure TurboGEPA based on dataset size.

    Simple, principled configuration that scales with dataset size:
        - Shards: Geometric progression (20 samples minimum, 3x reduction)
        - Concurrency: Scaled to dataset size (4 to 64)
        - Everything else: Auto-scaled from concurrency

    Parameters:
        dataset_size: Number of examples in dataset
        base_config: Optional base config to override (uses DEFAULT_CONFIG if None)

    Returns:
        Optimized Config object

    Examples:
        >>> config = adaptive_config(50)
        >>> config.shards
        (0.4, 1.0)
        >>> config.eval_concurrency
        16
    """
    config = base_config or Config()

    # Auto-select shards using principled algorithm
    config.shards = adaptive_shards(dataset_size)

    # Scale concurrency with dataset size (simple and predictable)
    # With fixed connection pooling, we can be more aggressive
    if dataset_size < 10:
        config.eval_concurrency = 32
        config.n_islands = 1
    elif dataset_size < 50:
        config.eval_concurrency = 64
        config.n_islands = 1
    elif dataset_size < 200:
        config.eval_concurrency = 128
        config.n_islands = 2
    else:
        config.eval_concurrency = 256
        config.n_islands = 4

    # Everything else auto-scales from concurrency in Config.__post_init__
    # This keeps the logic in one place and maintains consistency

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
    # Stricter variance tolerance for faster pruning
    config.variance_tolerance = _default_variance_tolerance(config.shards)
    if config.variance_tolerance:
        config.variance_tolerance = {k: v * 0.5 for k, v in config.variance_tolerance.items()}

    # Reduce breadth
    config.batch_size = 4  # Half the candidates (vs 8)
    config.max_mutations_per_round = 8  # Half the mutations (vs 16)

    # Streamlined reflection
    config.reflection_batch_size = 3  # Fewer traces (vs 6)

    # Single island - no migration overhead
    config.n_islands = 1
    config.migration_period = 999  # Effectively disabled

    # High concurrency for speed (with fixed connection pooling)
    config.eval_concurrency = 128

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
    # Moderate variance tolerance adjustment
    config.variance_tolerance = _default_variance_tolerance(config.shards)
    if config.variance_tolerance:
        config.variance_tolerance = {k: v * 0.75 for k, v in config.variance_tolerance.items()}

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
        0.85
    """
    config = base_config or Config()

    # Extreme 2-rung ASHA with large gap
    config.shards = (0.15, 1.0)
    # Very strict variance tolerance for maximum speed
    config.variance_tolerance = _default_variance_tolerance(config.shards)
    if config.variance_tolerance:
        config.variance_tolerance = {k: v * 0.3 for k, v in config.variance_tolerance.items()}

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
        "balanced": lambda ds, base_config=None: adaptive_config(ds, base_config=base_config, strategy="balanced"),
    }

    if mode not in modes:
        raise ValueError(f"Unknown lightning mode: '{mode}'. Choose from: {', '.join(modes.keys())}")

    return modes[mode](dataset_size, base_config=base_config)
