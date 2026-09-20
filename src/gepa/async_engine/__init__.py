# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Stage-decoupled asynchronous execution of GEPA's reflective optimization loop.

This package is a separate code path. It reuses GEPA's building blocks and
leaves :mod:`gepa.core.engine` and the synchronous proposers untouched.

    from gepa.async_engine import AsyncEngineConfig, StageConfig, optimize

    result = optimize(
        seed_candidate={"system_prompt": "..."},
        trainset=train,
        valset=val,
        adapter=my_adapter,
        reflection_lm="openai/gpt-5-mini",
        max_metric_calls=2000,
        async_config=AsyncEngineConfig(
            rollout=StageConfig(workers=2),
            propose=StageConfig(workers=6),
            screen=StageConfig(workers=2),
            validate=StageConfig(workers=1),
            staleness_policy="guarded",
            max_staleness=4,
        ),
    )
    print(result.metadata["async"]["stages"])
"""

from gepa.async_engine.api import optimize
from gepa.async_engine.config import AsyncEngineConfig, StageConfig
from gepa.async_engine.engine import AsyncStageEngine

__all__ = ["AsyncEngineConfig", "AsyncStageEngine", "StageConfig", "optimize"]
