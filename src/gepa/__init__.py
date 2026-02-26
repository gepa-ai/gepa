# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from . import (
    optimize_anything,  # expose submodule; use `from gepa.optimize_anything import optimize_anything` for the function
)
from .adapters import default_adapter
from .api import optimize
from .core.adapter import EvaluationBatch, GEPAAdapter
from .core.result import GEPAResult
from .examples import aime
from .image import Image
from .optimize_anything import OptimizeAnythingResult
from .utils.stop_condition import (
    CompositeStopper,
    FileStopper,
    MaxMetricCallsStopper,
    NoImprovementStopper,
    ScoreThresholdStopper,
    SignalStopper,
    StopperProtocol,
    TimeoutStopCondition,
)
