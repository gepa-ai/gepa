"""
Local adapter factory for TurboGEPA AIME runs.

This exposes an ``adapter_factory`` compatible with
``turbo_gepa.distributed.run_worker_from_factory`` /
``run_local_multiworker`` so we can launch multiple workers locally
without involving Modal.
"""

from __future__ import annotations

import os
from typing import Sequence, Tuple

from examples.aime_benchmark_v2 import _load_aime_subset, _quality_reward
from turbo_gepa.adapters import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config, adaptive_config


def adapter_factory() -> Tuple[DefaultAdapter, Sequence[str]]:
    """
    Build a DefaultAdapter for the AIME benchmark.

    Configuration is driven by environment variables so callers can tweak
    behaviour without changing code:

    - AIME_DATASET_SIZE (int, default 30)
    - AIME_EVAL_CONCURRENCY (int, default 20)
    - AIME_TARGET_QUALITY (float, default 0.733)
    - AIME_MAX_RUNTIME (int seconds, optional, default 240)
    - AIME_VERIFICATION_SPEED_BIAS (float 0–1, default 0.8)
    - AIME_N_ISLANDS (int, default 4)
    """

    dataset_size = int(os.getenv("AIME_DATASET_SIZE", "30"))
    eval_concurrency = int(os.getenv("AIME_EVAL_CONCURRENCY", "20"))
    target_quality = float(os.getenv("AIME_TARGET_QUALITY", "0.733"))
    max_runtime_env = os.getenv("AIME_MAX_RUNTIME")
    verification_speed_bias = float(os.getenv("AIME_VERIFICATION_SPEED_BIAS", "0.8"))
    n_islands = int(os.getenv("AIME_N_ISLANDS", "4"))

    trainset, _ = _load_aime_subset(dataset_size)
    dataset = [
        DefaultDataInst(
            input=example["input"],
            answer=example["answer"],
            additional_context=example.get("additional_context"),
            id=f"aime_{idx}",
        )
        for idx, example in enumerate(trainset)
    ]

    # Start from adaptive defaults, then override the few knobs we care about.
    config: Config = adaptive_config(len(dataset))
    config.n_islands = n_islands
    config.eval_concurrency = max(1, eval_concurrency)
    config.batch_size = min(len(dataset), config.eval_concurrency)
    config.target_quality = target_quality
    config.max_optimization_time_seconds = int(max_runtime_env) if max_runtime_env else 240
    config.verification_speed_bias = max(0.0, min(1.0, verification_speed_bias))

    adapter = DefaultAdapter(
        dataset=dataset,
        task_lm=os.getenv("TASK_LM", "openrouter/openai/gpt-oss-20b:nitro"),
        reflection_lm=os.getenv("REFLECTION_LM", "openrouter/x-ai/grok-4-fast"),
        config=config,
        auto_config=False,
        scoring_fn=_quality_reward,
    )
    # Return a small seed set; run_worker_from_factory will fall back to this
    # when no explicit seeds are supplied.
    seeds: Sequence[str] = [
        "You are a helpful assistant.",
        "You are an expert in solving challenging math competition problems, such as those from AIME.",
    ]
    return adapter, seeds

