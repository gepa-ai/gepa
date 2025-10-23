"""
Simple optimize() API for TurboGEPA that matches GEPA's interface.
"""

import asyncio
import os
import re
from typing import Any, Dict, List

from .archive import Archive
from .cache import DiskCache
from .config import Config
from .evaluator import AsyncEvaluator
from .interfaces import Candidate
from .logging_utils import build_logger
from .mutator import MutationConfig, Mutator
from .orchestrator import Orchestrator
from .sampler import InstanceSampler


def optimize(
    seed_candidate: Dict[str, str],
    trainset: List[Dict[str, Any]],
    valset: List[Dict[str, Any]] | None = None,
    task_lm: str = "openai/gpt-4o-mini",
    max_metric_calls: int = 150,
    reflection_lm: str | None = None,
    eval_concurrency: int = 16,
    shards: tuple[float, float] = (0.3, 1.0),
    cache_path: str = ".turbo_gepa/cache",
    log_path: str = ".turbo_gepa/logs",
    display_progress: bool = False,
) -> Any:
    """
    Simple optimization API matching GEPA's interface.

    Args:
        seed_candidate: Dict with 'system_prompt' key
        trainset: List of examples with 'input' and 'answer' keys
        valset: Optional validation set (not used yet)
        task_lm: Model to optimize (e.g., "openai/gpt-4o-mini")
        max_metric_calls: Budget for evaluations
        reflection_lm: Optional LLM for reflection (e.g., "openai/gpt-4o")
        eval_concurrency: Number of concurrent LLM calls
        shards: Successive halving shards (default aggressive)
        cache_path: Path for caching evaluations
        log_path: Path for logging
        display_progress: Display progress to stdout (default False)

    Returns:
        Result object with best_candidate, best_score, etc.
    """

    # Create dataset mapping
    example_map = {f"example-{idx}": ex for idx, ex in enumerate(trainset)}

    async def task_runner(candidate, example_id):
        """Async task runner for AIME - matches GEPA's DefaultAdapter logic."""
        example = example_map[example_id]

        try:
            from litellm import acompletion

            response = await acompletion(
                model=task_lm,  # Use full model name (e.g., "openrouter/x-ai/grok-4-fast")
                messages=[
                    {"role": "system", "content": candidate.text},
                    {"role": "user", "content": example["input"]},
                ],
            )

            model_output = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
        except Exception as e:
            print(f"Warning: LLM call failed ({e})")
            model_output = ""
            tokens_used = 100

        # Check correctness using GEPA's logic: simple substring match
        # This matches DefaultAdapter line 71: data["answer"] in assistant_response
        quality = 1.0 if example["answer"] in model_output else 0.0

        return {
            "quality": quality,
            "neg_cost": -float(tokens_used),
            "tokens": float(tokens_used),
        }

    async def _run_optimization():
        # Setup config
        config = Config(
            eval_concurrency=eval_concurrency,
            shards=shards,
            cache_path=cache_path,
            log_path=log_path,
            max_mutations_per_round=8,
            eps_improve=0.01,
            cohort_quantile=0.7,
        )

        # Note: reflection_lm is accepted for API compatibility but not yet implemented
        # in the simple optimize() API. Use the DSpyAdapter for full LLM-based reflection.
        # The simple API uses rule-based mutations only for now.
        reflection_runner = None

        # Setup components
        sampler = InstanceSampler(list(example_map.keys()), seed=42)
        cache = DiskCache(config.cache_path)
        evaluator = AsyncEvaluator(cache=cache, task_runner=task_runner)
        archive = Archive(
            bins_length=config.qd_bins_length,
            bins_bullets=config.qd_bins_bullets,
            flags=config.qd_flags,
        )
        mutator = Mutator(
            MutationConfig(
                reflection_batch_size=6,
                max_mutations=8,
                max_tokens=2048,
            ),
            reflection_runner=reflection_runner,
            seed=42,
        )
        logger = build_logger(config.log_path, "optimize", display_progress=display_progress)

        # Calculate max_rounds
        max_rounds = max_metric_calls // (config.batch_size * 3)

        orchestrator = Orchestrator(
            config=config,
            evaluator=evaluator,
            archive=archive,
            sampler=sampler,
            mutator=mutator,
            cache=cache,
            logger=logger,
            max_rounds=max_rounds,
        )

        # Create seed
        seed = Candidate(text=seed_candidate["system_prompt"])
        await orchestrator.run(
            seeds=[seed],
            max_rounds=max_rounds,
            max_evaluations=max_metric_calls,
        )
        await orchestrator.finalize()

        # Get best result
        pareto = archive.pareto_entries()
        if not pareto:
            raise RuntimeError("No candidates in Pareto frontier")

        best = max(pareto, key=lambda e: e.result.objectives.get("quality", 0))

        # Return GEPA-compatible result
        class Result:
            def __init__(self, candidate_text, score):
                self.best_candidate = {"system_prompt": candidate_text}
                self.best_score = score

        return Result(best.candidate.text, best.result.objectives.get("quality", 0))

    # Run async optimization
    return asyncio.run(_run_optimization())
