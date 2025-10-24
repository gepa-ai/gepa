"""
TurboGEPA package initialization.

This package provides a high-throughput, GEPA-inspired optimization engine
designed for async, multi-process evaluation and diversity-aware search.
"""

from .adapters import DefaultAdapter, DefaultDataInst  # noqa: F401

# Optional imports - only available if dependencies are installed
try:
    from .adapters import DSpyAdapter, ScoreWithFeedback  # noqa: F401
except ImportError:
    # dspy not installed
    pass
from .archive import Archive  # noqa: F401
from .cache import DiskCache  # noqa: F401
from .config import (  # noqa: F401
    Config,
    DEFAULT_CONFIG,
    adaptive_config,
    adaptive_shards,
    blitz_config,
    get_lightning_config,
    lightning_config,
    sprint_config,
)
from .evaluator import AsyncEvaluator  # noqa: F401
from .interfaces import Candidate, EvalResult  # noqa: F401
from .mutator import MutationConfig, Mutator  # noqa: F401
from .orchestrator import Orchestrator  # noqa: F401
from .sampler import InstanceSampler  # noqa: F401
from .token_controller import TokenCostController  # noqa: F401

# High-level API
from .optimize import optimize  # noqa: F401
