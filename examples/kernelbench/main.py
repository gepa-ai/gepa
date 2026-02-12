#!/usr/bin/env python3
"""KernelBench optimization with GEPA using RefinerConfig."""

import os
import time

import dspy

from examples.kernelbench.eval import (
    compute_score,
    format_error,
    load_dataset,
    load_or_measure_baselines,
    run_kernel,
)
from examples.kernelbench.prompts import (
    BACKGROUND,
    KERNEL_GEN_PROMPT,
    OBJECTIVE,
    KernelGenSig,
)
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    RefinerConfig,
    optimize_anything,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

MAX_METRIC_CALLS = 2000


def main():
    log_dir = f"outputs/kernelbench/{time.strftime('%y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)

    dataset = load_dataset()
    baselines = load_or_measure_baselines(dataset)

    lm = dspy.LM("openai/gpt-5.1", temperature=1.0, max_tokens=32000)
    predictor = dspy.Predict(KernelGenSig)

    seed = {"kernel_gen_prompt": KERNEL_GEN_PROMPT}

    def fitness_fn(candidate, example):
        """Evaluate a kernel generation prompt on a single problem."""
        baseline = baselines[example.problem_id]
        code, cuda_docs, eval_result = run_kernel(
            candidate["kernel_gen_prompt"], example.ref_arch, lm, predictor
        )
        score = compute_score(eval_result, baseline)
        runtime = eval_result.get("PerformanceStatsMean")
        side_info =  {
            "Score": score,
            "problem_id": example.problem_id,
            "level": example.level,
            "baseline_ms": baseline,
            "Code": code,
            "cuda_docs": cuda_docs,
            "runtime_ms": runtime,
            "speedup": baseline / runtime if runtime else None,
            "is_correct": eval_result.get("CorrectnessSucceeded", False),
            "error_feedback": format_error(eval_result) if eval_result.get("ErrorType") else None,
        }
        return score, side_info

    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=log_dir,
            max_metric_calls=MAX_METRIC_CALLS,
            cache_evaluation=True,
            track_best_outputs=True,
        ),
        refiner=RefinerConfig(),
    )

    optimize_anything(
        seed_candidate=seed,
        evaluator=fitness_fn,
        dataset=dataset,
        config=config,
        objective=OBJECTIVE,
        background=BACKGROUND,
    )


if __name__ == "__main__":
    main()
