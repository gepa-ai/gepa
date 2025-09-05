# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from .adapters import default_adapter
from .api import optimize
from .core.adapter import EvaluationBatch, GEPAAdapter
from .core.result import GEPAResult
from .examples import aime
from .utils.stopping import (
    create_composite_stopper,
    create_file_stopper,
    create_iteration_stopper,
    create_score_threshold_stopper,
    create_timeout_stopper,
)
