#!/usr/bin/env python3
from examples.kernelbench.utils.agentic_rag import retrieve_cuda_docs
from examples.kernelbench.utils.background import BACKGROUND
from examples.kernelbench.utils.eval import (
    compute_score,
    execute_kernel,
    get_baseline,
    load_dataset,
)
from gepa.core.adapter import DataInst
from gepa.optimize_anything import EngineConfig, GEPAConfig, RefinerConfig, SideInfo, optimize_anything


def evaluate(candidate: str, example: DataInst) -> tuple[float, SideInfo]:
    problem_id = example.problem_id
    ref_arch = example.ref_arch
    baseline = get_baseline(problem_id, ref_arch)

    eval_result = execute_kernel(candidate, ref_arch)
    score = compute_score(eval_result, baseline)
    cuda_docs = retrieve_cuda_docs(eval_result, ref_arch)

    runtime = eval_result.get("PerformanceStatsMean")
    speedup = baseline / runtime if runtime else None

    side_info: SideInfo = {
        "score": score,
        "problem_id": problem_id,
        "level": example.level,
        "baseline_ms": baseline,
        "reference_kernel": ref_arch,
        "new_kernel": candidate,
        "runtime_ms": runtime,
        "speedup": speedup,
        "compilation_succeeded": eval_result.get("CompilationSucceeded", False),
        "model_initialize_succeeded": eval_result.get("ModelInitializeSucceeded", False),
        "no_runtime_error": eval_result.get("NoRuntimeErrorDuringCorrectnessCheck", False),
        "no_output_shape_mismatch": eval_result.get("NoOutputShapeMismatch", False),
        "correctness_succeeded": eval_result.get("CorrectnessSucceeded", False),
        "no_perf_error": eval_result.get("NoRuntimeErrorDuringPerformanceCheck", False),
        "error_type": eval_result.get("ErrorType"),
        "error_detail": eval_result.get("ErrorDetail"),
        "cuda_docs": cuda_docs,
    }

    status = f"speedup={speedup:.2f}x" if speedup else "speedup=N/A"
    print(f"\033[94m[EVAL] {problem_id} | score={score:.3f} | {status}\033[0m")
    return score, side_info


def main():
    optimize_anything(
        seed_candidate=None,
        evaluator=evaluate,
        dataset=load_dataset(),
        config=GEPAConfig(
            engine=EngineConfig(
                run_dir="outputs/kernelbench/",
                max_metric_calls=3000,
                cache_evaluation=True,
                track_best_outputs=True,
            ),
            refiner=RefinerConfig(),
        ),
        objective="Produce faster, correct CUDA kernels that outperform PyTorch baselines.",
        background=BACKGROUND,
    )


if __name__ == "__main__":
    main()
