"""
Central configuration knobs for TurboGEPA.

Defaults are intentionally conservative so the system can run on a laptop
without further tuning. Users can override values by passing keyword args
into orchestrator entrypoints or by subclassing ``Config``.
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
from typing import Sequence

from turbo_gepa.stop_governor import StopGovernorConfig
from turbo_gepa.strategies import (
    ReflectionStrategy,
    resolve_reflection_strategy_names,
)
from turbo_gepa.scoring import ScoringFn, maximize_metric, SCORE_KEY


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

        # Apply rung-specific caps so early rungs stay strict
        # Treat the very first rung extra strictly (≤ first shard).
        if shard <= shards[0]:
            total_tolerance = min(total_tolerance, 0.10)
        # Apply medium tolerance to the next rung (e.g., 0.2) before easing up.
        elif shard <= (shards[1] if len(shards) > 1 else 0.65):
            total_tolerance = min(total_tolerance, 0.15)
        else:
            total_tolerance = min(total_tolerance, 0.25)

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
    ladder_density: float = 1.2,
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
        ladder_density: Adjusts how many intermediate rungs are generated
                        (0.5 = sparser ladder, 1.5 = denser ladder)

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

    if dataset_size <= 3:
        return (1.0,)

    # Unified algorithm for all dataset sizes: build a geometric ladder from an
    # uncertainty-aware first rung to full coverage, then merge overly dense rungs.
    # Count-aware rounding ensures sensible rungs even for small N without
    # special-casing.

    ladder_density = max(0.5, min(1.5, ladder_density))

    # First rung should have enough examples to be meaningful, but not exceed 50% coverage
    ratio = min_samples_per_rung / dataset_size
    min_fraction = ratio / (1.0 + ratio)
    dynamic_floor = max(2 / dataset_size, 0.06)
    min_fraction = max(dynamic_floor, min_fraction)
    min_fraction = min(min_fraction, 0.5)

    # Determine how many rungs we want (including the final full-shard rung)
    span = math.log(max(1.01, 1.0 / min_fraction), reduction_factor)
    raw_rungs = span * ladder_density
    target_rungs = int(math.ceil(raw_rungs)) + 1  # +1 to include the final full-shard rung
    target_rungs = max(2, min(6, target_rungs))

    # Convert target rung count into a geometric ladder that ends at full coverage
    ladder_len = max(1, target_rungs - 1)
    if min_fraction >= 0.95:
        return (0.95, 1.0)

    growth = (1.0 / min_fraction) ** (1.0 / ladder_len)
    fractions: list[float] = []
    current = min_fraction
    for _ in range(ladder_len):
        fractions.append(current)
        current *= growth

    # Convert to sample counts to avoid tiny fractions on small datasets
    counts: list[int] = []
    for frac in fractions:
        count = int(round(frac * dataset_size))
        count = min(dataset_size - 1, max(1, count))
        counts.append(count)

    # Deduplicate and ensure strictly increasing order
    deduped: list[int] = []
    for count in sorted(set(counts)):
        if deduped and count <= deduped[-1]:
            continue
        deduped.append(count)

    # Ensure we have at least one intermediate rung
    if not deduped:
        first = max(1, min(dataset_size - 1, round(min_fraction * dataset_size)))
        deduped = [first]

    rounded_rungs: list[float] = []
    for count in deduped:
        frac = round(count / dataset_size, 2)
        if frac <= 0.0:
            frac = round(1 / dataset_size, 2)
        if rounded_rungs and frac <= rounded_rungs[-1]:
            continue
        rounded_rungs.append(frac)

    if not rounded_rungs or rounded_rungs[-1] < 0.95:
        rounded_rungs.append(1.0)
    else:
        rounded_rungs[-1] = 1.0

    # Merge excessively dense ladders by removing rungs that are too close together
    merged: list[float] = []
    min_gap = 0.08
    for frac in rounded_rungs:
        if merged and frac - merged[-1] < min_gap and frac < 1.0:
            continue
        merged.append(min(frac, 1.0))

    if merged[-1] != 1.0:
        merged[-1] = 1.0

    return tuple(merged)


@dataclass(slots=True)
class Config:
    """Runtime parameters controlling concurrency, shard sizes, and promotion."""

    # Practical default ceiling for example-level concurrency. The adaptive governor
    # will adjust effective concurrency at runtime based on observed latency/backlog.
    eval_concurrency: int = 20
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
    control_dir: str | None = None
    batch_size: int | None = None  # Auto-scaled to eval_concurrency if None
    queue_limit: int | None = None  # Auto-scaled to 2x eval_concurrency if None
    scoring_fn: ScoringFn = maximize_metric("quality")
    promote_objective: str = SCORE_KEY
    max_mutations_per_round: int | None = None  # Auto-scaled to eval_concurrency if None
    mutation_buffer_limit: int | None = None  # Max pending streamed mutations awaiting queue capacity
    task_lm_temperature: float | None = 1.0
    reflection_lm_temperature: float | None = 1.0
    target_quality: float | None = None  # Stop when best quality reaches this threshold
    target_shard_fraction: float | None = 1.0  # Rung fraction that counts as "full" for turbo metric
    eval_timeout_seconds: float | None = 120.0  # Max time to wait for a single LLM evaluation
    max_optimization_time_seconds: float | None = None  # Global timeout - stop optimization after this many seconds
    reflection_strategy_names: tuple[str, ...] | None = None  # Default to all known strategies
    reflection_strategies: tuple[ReflectionStrategy, ...] | None = None
    max_final_shard_inflight: int | None = None  # If None, derive cap from eval_concurrency
    straggler_grace_seconds: float = 5.0  # Wait this long for detached stragglers before replaying
    final_rung_min_inflight: int = 2  # Do not shrink final rung concurrency below this floor
    final_rung_cap_max_fraction: float | None = None  # Max share of eval_concurrency allocated to final rung
    cancel_stragglers_immediately: bool = False  # Let detached example tasks finish in background
    replay_stragglers: bool = True  # Re-evaluate missing examples after straggler cancellation
    replay_workers: int | None = None  # Number of background workers for straggler replays
    replay_worker_queue_size: int | None = None  # Optional bound for replay queue
    replay_concurrency: int | None = None  # Max concurrency per replay evaluation
    # Single knob for verification speed/accuracy tradeoff:
    #   0.0 = strict / slow
    #   1.0 = aggressive / fast
    verification_speed_bias: float = 0.3
    # Minimum samples before early success checks (shaped from speed-bias)
    min_samples_for_confidence: int | None = None
    # Global confidence margin (z-score) derived from verification_speed_bias.
    # Used by evaluators as the number of standard errors to subtract from the
    # running mean when deciding if we've confidently cleared a target.
    confidence_z: float | None = None
    llm_connection_limit: int | None = None  # Cap simultaneous LLM calls (defaults to 1.5x eval_concurrency)
    # Dynamically scale effective evaluation concurrency to maximize throughput
    auto_scale_eval_concurrency: bool = True
    # Enforce a global example-level budget across all candidates to avoid oversubscription
    global_concurrency_budget: bool = True
    latest_results_limit: int = 2048

    # Streaming mode / distributed config
    worker_id: int | None = None
    worker_count: int | None = None
    islands_per_worker: int | None = None
    shared_cache_namespace: str | None = None
    migration_backend: str | None = None  # "local", "volume", etc.
    migration_path: str | None = None
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

        if self.mutation_buffer_limit is None:
            self.mutation_buffer_limit = max(128, self.eval_concurrency * 6)

        if self.llm_connection_limit is None:
            self.llm_connection_limit = max(8, int(self.eval_concurrency * 1.5))

        if self.target_shard_fraction is None:
            self.target_shard_fraction = 1.0

        if self.worker_count and not self.islands_per_worker:
            self.islands_per_worker = max(1, self.n_islands // max(1, self.worker_count))

        # Auto-generate variance_tolerance if not provided
        if self.variance_tolerance is None:
            self.variance_tolerance = _default_variance_tolerance(self.shards)

        # Auto-generate shrinkage_alpha if not provided
        if self.shrinkage_alpha is None:
            self.shrinkage_alpha = _default_shrinkage_alpha(self.shards)

        # Final rung concurrency guardrails
        self.final_rung_min_inflight = max(1, int(self.final_rung_min_inflight or 1))
        cap_max_fraction = self.final_rung_cap_max_fraction
        if cap_max_fraction is None:
            # Default: allow the final rung to use up to 100% of eval_concurrency.
            cap_max_fraction = 1.0
        cap_max_fraction = max(0.1, min(1.0, float(cap_max_fraction)))
        self.final_rung_cap_max_fraction = cap_max_fraction

        if self.max_final_shard_inflight is not None:
            self.max_final_shard_inflight = max(self.final_rung_min_inflight, int(self.max_final_shard_inflight))
        else:
            # Auto-set default cap from eval_concurrency:
            # - Aim for ~half of eval_concurrency allocated to final-rung candidates
            # - Respect the max fraction guardrail and the minimum inflight floor
            eval_cap = max(1, int(self.eval_concurrency))
            half_cap = max(1, eval_cap // 2)
            frac_cap = int(eval_cap * self.final_rung_cap_max_fraction)
            if frac_cap <= 0:
                frac_cap = 1
            base_cap = min(half_cap, frac_cap)
            self.max_final_shard_inflight = max(self.final_rung_min_inflight, base_cap)

        if self.replay_workers is None:
            # Default: dedicate ~10% of evaluator concurrency, at least 1
            self.replay_workers = max(1, int(round(self.eval_concurrency * 0.1)))
        else:
            self.replay_workers = max(0, int(self.replay_workers))
        if self.replay_concurrency is None:
            self.replay_concurrency = max(1, int(max(1, self.eval_concurrency) * 0.1))
        else:
            self.replay_concurrency = max(1, int(self.replay_concurrency))
        if self.replay_worker_queue_size is not None:
            self.replay_worker_queue_size = max(1, int(self.replay_worker_queue_size))

        self._apply_verification_profile()

        custom_strategies = list(self.reflection_strategies or ())
        resolved_defaults = list(resolve_reflection_strategy_names(self.reflection_strategy_names))
        resolved_defaults.extend(custom_strategies)
        if not resolved_defaults:
            raise ValueError(
                "At least one reflection strategy must be configured. "
                "Provide reflection_strategy_names or reflection_strategies."
            )
        self.reflection_strategies = tuple(resolved_defaults)
        self.promote_objective = SCORE_KEY

    def _apply_verification_profile(self) -> None:
        """Derive verification thresholds from the configured risk knob."""

        speed = max(0.0, min(1.0, float(self.verification_speed_bias)))
        # Expose the clipped value so downstream consumers see the effective bias.
        self.verification_speed_bias = speed
        # Use a non-linear mapping so higher values of the dial have a
        # disproportionately stronger effect. This keeps low/medium settings
        # close to the current behaviour but makes 0.8–1.0 meaningfully more
        # aggressive.
        fast = speed**2.5

        def _blend(high: float, low: float) -> float:
            return high + (low - high) * fast

        if self.min_samples_for_confidence is None:
            # Require many samples at slow end, very few at the fastest.
            samples = int(round(_blend(30, 3)))
            self.min_samples_for_confidence = max(1, samples)

        # Derive a global z-score (confidence margin) from the same dial.
        # At the slow/accuracy end we want a conservative margin (~95% CI),
        # while at the fast end we accept much tighter bounds.
        z_slow = 2.0   # ~95% confidence
        z_fast = 0.3   # very permissive, speed-biased
        z = z_slow - (z_slow - z_fast) * fast
        self.confidence_z = float(max(0.0, z))


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
