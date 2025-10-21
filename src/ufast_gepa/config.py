"""
Central configuration knobs for uFast-GEPA.

Defaults are intentionally conservative so the system can run on a laptop
without further tuning. Users can override values by passing keyword args
into orchestrator entrypoints or by subclassing ``Config``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


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
    cache_path: str = ".ufast_gepa/cache"
    log_path: str = ".ufast_gepa/logs"
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
