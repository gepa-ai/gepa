"""
TurboGEPA package initialization.

This package provides a high-throughput, GEPA-inspired optimization engine
designed for async, multi-island evaluation (single-process, asyncio-based)
and diversity-aware search.
"""

from .adapters import DefaultAdapter, DefaultDataInst

# Optional imports - only available if dependencies are installed
try:
    from .adapters import DSpyAdapter, ScoreWithFeedback
except ImportError:
    # dspy not installed
    pass
from .archive import Archive
from .cache import DiskCache
from .config import (
    DEFAULT_CONFIG,
    Config,
    adaptive_config,
    adaptive_shards,
    blitz_config,
    get_lightning_config,
    lightning_config,
    sprint_config,
)
from .evaluator import AsyncEvaluator
from .interfaces import Candidate, EvalResult
from .mutator import MutationConfig, Mutator
from .orchestrator import Orchestrator
from .sampler import InstanceSampler

# High-level API
# High-level optimize() API removed to avoid half-baked rule-based path
