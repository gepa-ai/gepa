#!/usr/bin/env python3
"""
KernelBench optimization with GEPA using RefinerConfig.

Simplified version that uses adapter-based caching and automatic refinement.
"""

import argparse
import os
import time

import dspy

from examples.kernelbench.agentic_rag import agentic_retrieve
from examples.kernelbench.eval import (
    compute_score,
    execute_kernel,
    extract_code,
    format_error,
    get_free_gpus,
    init_gpu_manager,
    load_dataset,
    load_or_measure_baselines,
)
from examples.kernelbench.prompts import (
    BACKGROUND,
    KERNEL_GEN_PROMPT,
    KernelGenSig,
)
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    RefinerConfig,
    optimize_anything,
)

LLM = "openai/gpt-5"
TIMEOUT = 360


def get_stages(r: dict) -> dict:
    """Extract 6 evaluation stages from result dict."""
    return {
        "CompilationSucceeded": r.get("CompilationSucceeded", False),
        "ModelInitializeSucceeded": r.get("ModelInitializeSucceeded", False),
        "NoRuntimeErrorDuringCorrectnessCheck": r.get("NoRuntimeErrorDuringCorrectnessCheck", False),
        "NoOutputShapeMismatch": r.get("NoOutputShapeMismatch", False),
        "CorrectnessSucceeded": r.get("CorrectnessSucceeded", False),
        "NoRuntimeErrorDuringPerformanceCheck": r.get("NoRuntimeErrorDuringPerformanceCheck", False),
    }


def create_fitness_fn(
    lm,
    baselines: dict[str, float],
    use_rag: bool = True,
):
    """Create fitness function for GEPA with RefinerConfig.

    Args:
        lm: DSPy language model
        baselines: Dict mapping problem_id -> baseline_time_ms
        use_rag: Whether to use RAG for CUDA documentation
    """
    gen_predictor = dspy.Predict(KernelGenSig)

    def fitness_fn(candidate: dict, **kwargs) -> tuple[float, dict]:
        """Evaluate a kernel generation prompt on a single problem.

        The adapter's RefinerConfig handles refinement automatically:
        1. fitness_fn is called with candidate
        2. If score is below threshold, adapter calls refiner LLM
        3. Refiner produces improved candidate
        4. fitness_fn is called again with improved candidate
        5. Repeat until max_refinements or no improvement
        """
        ex = kwargs["example"]
        ref_arch = ex.ref_arch
        problem_id = ex.problem_id
        baseline = baselines[problem_id]

        kernel_gen_prompt = candidate["kernel_gen_prompt"]

        # Generate kernel with optional RAG
        cuda_docs = ""
        if use_rag:
            cuda_docs = agentic_retrieve(f"CUDA kernel for:\n{ref_arch[:1500]}", verbose=False)

        with dspy.context(lm=lm):
            result = gen_predictor(prompt=kernel_gen_prompt, ref_arch=ref_arch, cuda_docs=cuda_docs)
        code = extract_code(result.code) or result.code or ""

        # Evaluate kernel
        eval_result = execute_kernel(code, ref_arch, timeout=TIMEOUT)
        score = compute_score(eval_result, baseline)
        runtime = eval_result.get("PerformanceStatsMean")
        speedup = baseline / runtime if runtime else None

        # Build side_info with all relevant information
        side_info = {
            "scores": {"score": score},
            # Problem info
            "problem_id": problem_id,
            "level": ex.level,
            "baseline_ms": baseline,
            # Generated code
            "Code": code,
            "cuda_docs": cuda_docs,
            # Evaluation results
            "stages": get_stages(eval_result),
            "runtime_ms": runtime,
            "speedup": speedup,
            "is_correct": eval_result.get("CorrectnessSucceeded", False),
            # Error info (used by refiner for feedback)
            "error_type": eval_result.get("ErrorType"),
            "error_detail": eval_result.get("ErrorDetail"),
            "error_feedback": format_error(eval_result) if eval_result.get("ErrorType") else None,
        }

        return score, side_info

    return fitness_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-metric-calls", type=int, default=2000)
    parser.add_argument("--max-refinements", type=int, default=5)
    parser.add_argument("--levels", type=str, default="level1,level2,level3")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-rag", action="store_true")
    parser.add_argument("--force-baseline", action="store_true", help="Force re-measurement of baselines")
    parser.add_argument("--gpus", type=str, default=None,
                        help="GPU indices to use, e.g. '0,1,2,3' or '4' for first 4 GPUs")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel fitness evaluation")
    parser.add_argument("--max-workers", type=int, default=None, help="Max parallel workers (default: num GPUs)")
    args = parser.parse_args()

    run_name = args.run_name or time.strftime("%y%m%d_%H%M%S")
    log_dir = f"outputs/artifacts/kernelbench_refiner/{run_name}"
    os.makedirs(log_dir, exist_ok=True)

    # Determine parallelization settings
    parallel_mode = args.parallel
    max_workers = args.max_workers

    print(f"\n{'='*60}")
    parallel_banner = f"parallel={args.max_workers or 'auto'}" if parallel_mode else "sequential"
    print(f"KernelBench (RefinerConfig) | {LLM} | RAG={'off' if args.no_rag else 'on'} | {parallel_banner}")
    print(f"Output: {log_dir}")
    print(f"{'='*60}\n")

    levels = [x.strip() for x in args.levels.split(",")]
    dataset = load_dataset(levels=levels)
    print(f"Dataset: {len(dataset)} problems ({', '.join(levels)})")

    baselines = load_or_measure_baselines(dataset, force=args.force_baseline)
    print(f"Baselines loaded: {len(baselines)} problems")
    for pid, btime in sorted(baselines.items()):
        print(f"  {pid}: {btime:.2f} ms")

    # Initialize GPU manager
    if args.gpus:
        if "," in args.gpus:
            available_gpus = [int(x.strip()) for x in args.gpus.split(",")]
        else:
            available_gpus = list(range(int(args.gpus)))
    else:
        available_gpus = get_free_gpus()
        if not available_gpus:
            available_gpus = list(range(4))
            print(f"WARNING: No free GPUs detected, using {available_gpus}")
    gpu_lock_dir = os.path.join(log_dir, "gpu_locks")
    init_gpu_manager(device_list=available_gpus, lock_dir=gpu_lock_dir)

    if max_workers is None:
        max_workers = len(available_gpus) if parallel_mode else 1

    parallel_str = f"parallel={max_workers} workers" if parallel_mode else "sequential"
    print(f"GPUManager initialized: GPUs={available_gpus}, {parallel_str}")

    lm = dspy.LM(LLM, temperature=1.0, max_tokens=32000)

    # GEPA config with RefinerConfig - adapter handles caching and refinement
    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=log_dir,
            max_metric_calls=args.max_metric_calls,
            cache_evaluation=True,  # Adapter handles caching
            track_best_outputs=True,
            parallel=parallel_mode,
            max_workers=max_workers,
        ),
        reflection=ReflectionConfig(
            reflection_minibatch_size=3,
            reflection_lm=LLM,
        ),
        refiner=RefinerConfig(  # Adapter handles refinement
            refiner_lm=LLM,
            max_refinements=args.max_refinements,
        ),
    )

    objective = "Generate an LLM prompt that produces fast, correct CUDA kernels outperforming PyTorch baselines."

    # Single-component seed - refiner uses DEFAULT_REFINER_PROMPT
    seed = {
        "kernel_gen_prompt": KERNEL_GEN_PROMPT,
    }

    fitness_fn = create_fitness_fn(
        lm,
        baselines=baselines,
        use_rag=not args.no_rag,
    )

    result = optimize_anything(
        seed_candidate=seed,
        fitness_fn=fitness_fn,
        dataset=dataset,
        config=config,
        objective=objective,
        background=BACKGROUND,
    )

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Best GEPA score: {result.best_score:.4f}")
    print(f"Best candidate: {result.best_candidate}")


if __name__ == "__main__":
    main()
