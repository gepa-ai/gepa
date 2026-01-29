#!/usr/bin/env python3
"""
KernelBench optimization with GEPA.
We collect the baseline (ms) in advance and use it to compute the speedup.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
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
    REFINER_PROMPT,
    KernelGenSig,
    RefinerSig,
)
from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, SideInfo, optimize_anything


class PromptCache:
    """Cache for LLM prompt outputs and their evaluation results.

    Used for both initial kernel generation and refinement steps.
    Keyed on (prompt, problem_id, input_code_hash) where input_code_hash is
    empty for initial generation and the hash of failed_code for refinements.
    """

    def __init__(self, cache_dir: str, name: str = "prompt"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self._hit_count = 0
        self._miss_count = 0

    def _make_key(self, prompt: str, problem_id: str, input_code: str = "") -> str:
        """Create a unique cache key."""
        data = json.dumps({
            "prompt": prompt,
            "problem_id": problem_id,
            "input_code_hash": hashlib.sha256(input_code.encode()).hexdigest() if input_code else "",
        }, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()

    def get(self, prompt: str, problem_id: str, input_code: str = "") -> dict | None:
        """Get cached output code, eval result, and cuda_docs if they exist.

        Returns:
            dict with "code", "eval_result", and "cuda_docs" keys, or None if not cached.
        """
        key = self._make_key(prompt, problem_id, input_code)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            self._hit_count += 1
            with open(cache_file) as f:
                return json.load(f)
        return None

    def put(self, prompt: str, problem_id: str, code: str, eval_result: dict, cuda_docs: str = "", input_code: str = "") -> None:
        """Store output code, eval result, and cuda_docs in cache."""
        key = self._make_key(prompt, problem_id, input_code)
        cache_file = self.cache_dir / f"{key}.json"
        self._miss_count += 1
        with open(cache_file, "w") as f:
            json.dump({"code": code, "eval_result": eval_result, "cuda_docs": cuda_docs}, f)

    def stats(self) -> dict:
        """Return cache hit/miss statistics."""
        total = self._hit_count + self._miss_count
        return {
            "hits": self._hit_count,
            "misses": self._miss_count,
            "total": total,
            "hit_rate": self._hit_count / total if total > 0 else 0,
        }


class StateTracker:
    """Tracker for metric calls and best speedups.

    Uses JSONL (append-only) for metric logs to avoid O(n²) I/O.
    """

    def __init__(self, log_dir: str, total_problems: int):
        self.log_dir = log_dir
        self.total_problems = total_problems
        self._logs_path = os.path.join(log_dir, "metric_logs.jsonl")
        self._best_path = os.path.join(log_dir, "best_kernels.json")
        self._old_logs_path = os.path.join(log_dir, "metric_logs.json")

        # Convert old JSON format to JSONL if needed
        if os.path.exists(self._old_logs_path) and not os.path.exists(self._logs_path):
            print(f"[StateTracker] Converting {self._old_logs_path} to JSONL format...")
            import shutil
            shutil.copy(self._old_logs_path, self._old_logs_path + ".bak")
            print(f"[StateTracker] Backup saved to {self._old_logs_path}.bak")
            with open(self._old_logs_path) as f:
                old_logs = json.load(f)
            with open(self._logs_path, "w") as f:
                for entry in old_logs:
                    f.write(json.dumps(entry) + "\n")
            print(f"[StateTracker] Converted {len(old_logs)} entries to {self._logs_path}")

        # Try to load existing state for resume
        if os.path.exists(self._logs_path) and os.path.exists(self._best_path):
            # Count lines in JSONL for metric_calls
            with open(self._logs_path) as f:
                self.metric_calls = sum(1 for _ in f)

            with open(self._best_path) as f:
                best_data = json.load(f)
            self.best_speedups = {pid: d["speedup"] for pid, d in best_data.items()}
            self.best_codes = {pid: d.get("code", "") for pid, d in best_data.items()}
            self.best_prompts = {pid: d.get("prompts", {}) for pid, d in best_data.items()}

            print(f"[StateTracker] Resumed: {self.metric_calls} metric calls, {len(self.best_speedups)} problems with solutions")
        else:
            # Fresh start
            self.metric_calls = 0
            self.best_speedups: dict[str, float] = {}  # problem_id -> best speedup
            self.best_codes: dict[str, str] = {}  # problem_id -> best kernel code
            self.best_prompts: dict[str, dict[str, str]] = {}  # problem_id -> {"kernel_gen_prompt": ..., "refiner_prompt": ...}

    def log(
        self,
        problem_id: str,
        attempt: str,
        result: dict,
        score: float,
        baseline: float,
        code: str = "",
        prompts: dict[str, str] | None = None,
        **extra,
    ):
        """Log a metric call."""
        self.metric_calls += 1
        runtime = result.get("PerformanceStatsMean")
        speedup = baseline / runtime if runtime else None
        correct = result.get("CorrectnessSucceeded", False)

        # Update best speedup, code, and prompts
        is_new_best = False
        if speedup and correct:
            if speedup > self.best_speedups.get(problem_id, 0):
                is_new_best = True
                self.best_speedups[problem_id] = speedup
                self.best_codes[problem_id] = code
                if prompts:
                    self.best_prompts[problem_id] = prompts

        # Compute fast_p metrics
        f_0 = len(self.best_speedups) / max(self.total_problems, 1)  # Any correct solution
        f_1_0 = sum(1 for s in self.best_speedups.values() if s >= 1.0) / max(self.total_problems, 1)
        f_1_1 = sum(1 for s in self.best_speedups.values() if s >= 1.1) / max(self.total_problems, 1)
        f_1_2 = sum(1 for s in self.best_speedups.values() if s >= 1.2) / max(self.total_problems, 1)

        entry = {
            "metric_call": self.metric_calls,
            "problem_id": problem_id,
            "attempt": attempt,
            "score": score,
            "compiled": result.get("CompilationSucceeded", False),
            "correct": correct,
            "runtime_ms": runtime,
            "baseline_ms": baseline,
            "speedup": speedup,
            "error_type": result.get("ErrorType"),
            "error_detail": result.get("ErrorDetail"),
            "f_0": f_0,
            "f_1_0": f_1_0,
            "f_1_1": f_1_1,
            "f_1_2": f_1_2,
            "best_speedups": dict(self.best_speedups),
            "best_code_for_problem": self.best_codes.get(problem_id, ""),
            "best_prompts_for_problem": self.best_prompts.get(problem_id, {}),
            "timestamp": time.time(),
            **extra,
        }

        # Append to JSONL (O(1) per write)
        with open(self._logs_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # Update best_kernels.json only when there's a new best
        if is_new_best:
            self._save_best()

        # Print status
        status = "\033[92mok\033[95m" if correct else "\033[91mX\033[95m"
        spd = f"{speedup:.2f}x" if speedup else "N/A"
        runtime_str = f"{runtime:.2f}ms" if runtime else "N/A"
        new_best = " \033[93m★ NEW BEST\033[95m" if is_new_best else ""
        error_info = ""
        if not correct:
            etype = result.get("ErrorType", "")
            edetail = result.get("ErrorDetail", "")[:100]  # truncate long errors
            if etype:
                error_info = f"\n    \033[91m└─ {etype}: {edetail}{'...' if len(result.get('ErrorDetail', '')) > 100 else ''}\033[0m"
        print(f"\033[95m[#{self.metric_calls}] {problem_id} {attempt}: {status}, {runtime_str}/{baseline:.2f}ms={spd}, score={score:.3f}, f_1_0={f_1_0:.1%}, f_1_1={f_1_1:.1%}, f_1_2={f_1_2:.1%}{new_best}\033[0m{error_info}")

    def _save_best(self):
        """Save best kernels to JSON."""
        best_results = {
            pid: {
                "speedup": self.best_speedups[pid],
                "code": self.best_codes.get(pid, ""),
                "prompts": self.best_prompts.get(pid, {}),
            }
            for pid in self.best_speedups
        }
        with open(self._best_path, "w") as f:
            json.dump(best_results, f, indent=2)


LLM = "openai/gpt-5"
TIMEOUT = 360


def truncate_code(code: str, head: int = 5, tail: int = 4) -> str:
    """Truncate code to first `head` and last `tail` lines."""
    lines = code.strip().split("\n")
    if len(lines) <= head + tail:
        return code
    return "\n".join(lines[:head] + [f"    ... ({len(lines) - head - tail} lines omitted) ..."] + lines[-tail:])


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


def generate_initial_kernel(
    gen_prompt: str,
    problem_id: str,
    ref_arch: str,
    lm,
    gen_predictor,
    use_rag: bool,
    cache: PromptCache | None,
) -> tuple[str, dict, str]:
    """Generate initial kernel with caching.

    Returns:
        (code, eval_result, cuda_docs)
    """
    # Check cache first
    if cache is not None:
        cached = cache.get(gen_prompt, problem_id)
        if cached is not None:
            code = cached["code"]
            eval_result = cached["eval_result"]
            cuda_docs = cached.get("cuda_docs", "")
            print(f"\033[92m[GEN CACHE HIT] {problem_id}\033[0m")
            print(f"\033[33m[KERNEL] {problem_id} initial (cached):\n{truncate_code(code)}\033[0m")
            return code, eval_result, cuda_docs

    # RAG
    cuda_docs = ""
    if use_rag:
        cuda_docs = agentic_retrieve(f"CUDA kernel for:\n{ref_arch[:1500]}", verbose=False)
        print(f"\033[36m[RAG] {problem_id} initial: {len(cuda_docs)} chars\033[0m")

    # Generate
    with dspy.context(lm=lm):
        result = gen_predictor(prompt=gen_prompt, ref_arch=ref_arch, cuda_docs=cuda_docs)
    code = extract_code(result.code) or result.code or ""
    print(f"\033[33m[KERNEL] {problem_id} initial:\n{truncate_code(code)}\033[0m")

    # Evaluate
    eval_result = execute_kernel(code, ref_arch, timeout=TIMEOUT)

    # Cache
    if cache is not None:
        cache.put(gen_prompt, problem_id, code, eval_result, cuda_docs=cuda_docs)
        print(f"\033[93m[GEN CACHE MISS] {problem_id} - stored\033[0m")

    return code, eval_result, cuda_docs


def refine_kernel(
    ref_prompt: str,
    problem_id: str,
    ref_arch: str,
    input_code: str,
    prev_eval_result: dict,
    iteration: int,
    lm,
    refine_predictor,
    use_rag: bool,
    cache: PromptCache | None,
) -> tuple[str, dict, str]:
    """Refine kernel with caching.

    Args:
        input_code: The code to refine (used as cache key)
        prev_eval_result: Eval result of input_code (for error formatting)
        iteration: 1-indexed refinement iteration

    Returns:
        (code, eval_result, cuda_docs)
    """
    # Check cache first
    if cache is not None:
        cached = cache.get(ref_prompt, problem_id, input_code)
        if cached is not None:
            code = cached["code"]
            eval_result = cached["eval_result"]
            cuda_docs = cached.get("cuda_docs", "")
            print(f"\033[92m[REF CACHE HIT] {problem_id} refine_{iteration}\033[0m")
            print(f"\033[33m[KERNEL] {problem_id} refine_{iteration} (cached):\n{truncate_code(code)}\033[0m")
            return code, eval_result, cuda_docs

    error = format_error(prev_eval_result)

    # RAG
    cuda_docs = ""
    if use_rag:
        cuda_docs = agentic_retrieve(f"Fix error:\n{error[:800]}\nTask:\n{ref_arch[:700]}", verbose=False)
        print(f"\033[36m[RAG] {problem_id} refine_{iteration}: {len(cuda_docs)} chars\033[0m")

    # Refine
    with dspy.context(lm=lm):
        result = refine_predictor(prompt=ref_prompt, failed_code=input_code, ref_arch=ref_arch, error=error, cuda_docs=cuda_docs)
    code = extract_code(result.code) or result.code or ""
    print(f"\033[33m[KERNEL] {problem_id} refine_{iteration}:\n{truncate_code(code)}\033[0m")

    # Evaluate
    eval_result = execute_kernel(code, ref_arch, timeout=TIMEOUT)

    # Cache
    if cache is not None:
        cache.put(ref_prompt, problem_id, code, eval_result, cuda_docs=cuda_docs, input_code=input_code)
        print(f"\033[93m[REF CACHE MISS] {problem_id} refine_{iteration} - stored\033[0m")

    return code, eval_result, cuda_docs


def create_fitness_fn(
    lm,
    baselines: dict[str, float],
    use_rag: bool = True,
    max_refinements: int = 5,
    tracker: StateTracker | None = None,
    kernel_gen_cache: PromptCache | None = None,
    refiner_cache: PromptCache | None = None,
):
    """Create fitness function for GEPA.

    Args:
        lm: DSPy language model
        baselines: Dict mapping problem_id -> baseline_time_ms (from load_or_measure_baselines)
        use_rag: Whether to use RAG for CUDA documentation
        max_refinements: Max refinement attempts
        tracker: Optional StateTracker for logging
        kernel_gen_cache: Cache for initial kernel generation (keyed on kernel_gen_prompt + problem_id)
        refiner_cache: Cache for refinement steps (keyed on refiner_prompt + problem_id + input_code)
    """
    gen_predictor = dspy.Predict(KernelGenSig)
    refine_predictor = dspy.Predict(RefinerSig)

    def fitness_fn(candidate: dict, **kwargs) -> tuple[float, dict, SideInfo]:
        ex = kwargs["example"]
        ref_arch = ex.ref_arch
        problem_id = ex.problem_id
        baseline = baselines[problem_id]

        gen_prompt = candidate["kernel_gen_prompt"]
        ref_prompt = candidate["refiner_prompt"]

        print(f"\033[94m[START] {problem_id} | baseline={baseline:.2f}ms\033[0m")

        # === Initial Generation ===
        code, eval_result, cuda_docs = generate_initial_kernel(
            gen_prompt, problem_id, ref_arch, lm, gen_predictor, use_rag, kernel_gen_cache
        )
        initial_score = compute_score(eval_result, baseline)
        initial_result = eval_result

        if tracker:
            tracker.log(problem_id, "initial", eval_result, initial_score, baseline, code=code,
                        prompts={"kernel_gen_prompt": gen_prompt, "refiner_prompt": ref_prompt}, level=ex.level)

        # Track attempts for side_info
        attempts = [{"code": code, "result": eval_result, "score": initial_score, "cuda_docs": cuda_docs}]
        best_score, best_code = initial_score, code

        # === Refinements ===
        # Always run at least one refinement, then continue until correct or max_refinements
        refinement_count = 0
        while refinement_count < max_refinements:
            # After first refinement, stop if kernel is correct
            if refinement_count > 0 and eval_result.get("CorrectnessSucceeded", False):
                print(f"\033[92m[EARLY STOP] {problem_id} - kernel correct after {refinement_count} refinements\033[0m")
                break

            input_code = code
            refinement_count += 1
            code, eval_result, cuda_docs = refine_kernel(
                ref_prompt, problem_id, ref_arch, input_code, eval_result, refinement_count,
                lm, refine_predictor, use_rag, refiner_cache
            )
            score = compute_score(eval_result, baseline)
            attempts.append({"code": code, "result": eval_result, "score": score, "cuda_docs": cuda_docs})

            if tracker:
                tracker.log(problem_id, f"refine_{refinement_count}", eval_result, score, baseline, code=code,
                            prompts={"kernel_gen_prompt": gen_prompt, "refiner_prompt": ref_prompt}, level=ex.level)

            if score > best_score:
                best_score, best_code = score, code

        # === Build output and side_info ===
        improvement = (best_score - initial_score) / abs(initial_score) if initial_score != 0 else best_score
        best_attempt = max(attempts, key=lambda a: a["score"])
        best_runtime = best_attempt["result"].get("PerformanceStatsMean")
        best_speedup = baseline / best_runtime if best_runtime else None

        output = {
            "problem_id": problem_id,
            "level": ex.level,
            "best_code": best_code,
            "best_score": best_score,
            "baseline_ms": baseline,
            "best_runtime_ms": best_runtime,
            "speedup": best_speedup,
            "is_correct": best_attempt["result"].get("CorrectnessSucceeded", False),
        }

        side_info: SideInfo = {
            "scores": {
                "kernel_gen_prompt": initial_score,
                "refiner_prompt": improvement,
            },
            "problem_id": problem_id,
            "level": ex.level,
            "baseline_ms": baseline,
            "best_speedup": best_speedup,
            "is_correct": best_attempt["result"].get("CorrectnessSucceeded", False),
            "kernel_gen_prompt_specific_info": {
                "scores": {"initial_score": initial_score},
                "stages": get_stages(initial_result),
                "runtime_ms": initial_result.get("PerformanceStatsMean"),
                "speedup": baseline / initial_result["PerformanceStatsMean"] if initial_result.get("PerformanceStatsMean") else None,
                "error_type": initial_result.get("ErrorType"),
                "error_detail": initial_result.get("ErrorDetail"),
                "cuda_docs": attempts[0]["cuda_docs"],
            },
            "refiner_prompt_specific_info": {
                "scores": {"improvement": improvement},
                "attempts": [
                    {
                        "phase": "initial" if j == 0 else f"refine_{j}",
                        "score": a["score"],
                        "stages": get_stages(a["result"]),
                        "runtime_ms": a["result"].get("PerformanceStatsMean"),
                        "speedup": baseline / a["result"]["PerformanceStatsMean"] if a["result"].get("PerformanceStatsMean") else None,
                        "error_type": a["result"].get("ErrorType"),
                        "error_detail": a["result"].get("ErrorDetail"),
                        "cuda_docs": a["cuda_docs"],
                    }
                    for j, a in enumerate(attempts)
                ],
                "refinements_used": len(attempts) - 1,
                "final_score": best_score,
                "final_speedup": best_speedup,
            },
        }

        print(f"\033[94m[END] {problem_id} | best_score={best_score:.3f} | speedup={best_speedup:.2f}x\033[0m" if best_speedup else f"\033[94m[END] {problem_id} | best_score={best_score:.3f} | speedup=N/A\033[0m")
        return best_score, output, side_info

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
    log_dir = f"outputs/artifacts/kernelbench/{run_name}"
    os.makedirs(log_dir, exist_ok=True)

    # Determine parallelization settings
    parallel_mode = args.parallel
    max_workers = args.max_workers  # Will be set after GPU detection if None

    print(f"\n{'='*60}")
    parallel_banner = f"parallel={args.max_workers or 'auto'}" if parallel_mode else "sequential"
    print(f"KernelBench | {LLM} | RAG={'off' if args.no_rag else 'on'} | {parallel_banner}")
    print(f"Output: {log_dir}")
    print(f"{'='*60}\n")

    levels = [x.strip() for x in args.levels.split(",")]
    dataset = load_dataset(levels=levels)
    print(f"Dataset: {len(dataset)} problems ({', '.join(levels)})")

    # Measure baselines through same path as LLM kernels (fair comparison)
    baselines = load_or_measure_baselines(dataset, force=args.force_baseline)
    print(f"Baselines loaded: {len(baselines)} problems")
    for pid, btime in sorted(baselines.items()):
        print(f"  {pid}: {btime:.2f} ms")

    # Initialize GPU manager for parallel execution with proper locking
    if args.gpus:
        # Parse --gpus argument: "0,1,2,3" or just "4" (meaning first 4)
        if "," in args.gpus:
            available_gpus = [int(x.strip()) for x in args.gpus.split(",")]
        else:
            available_gpus = list(range(int(args.gpus)))
    else:
        available_gpus = get_free_gpus()
        if not available_gpus:
            available_gpus = list(range(4))  # Default to 4 GPUs
            print(f"WARNING: No free GPUs detected, using {available_gpus}")
    gpu_lock_dir = os.path.join(log_dir, "gpu_locks")
    init_gpu_manager(device_list=available_gpus, lock_dir=gpu_lock_dir)

    # Set max_workers: 1 for sequential, num GPUs for parallel
    if max_workers is None:
        max_workers = len(available_gpus) if parallel_mode else 1

    parallel_str = f"parallel={max_workers} workers" if parallel_mode else "sequential"
    print(f"GPUManager initialized: GPUs={available_gpus}, {parallel_str}")

    tracker = StateTracker(log_dir=log_dir, total_problems=len(dataset))
    kernel_gen_cache = PromptCache(cache_dir=os.path.join(log_dir, "kernel_gen_cache"), name="kernel_gen")
    refiner_cache = PromptCache(cache_dir=os.path.join(log_dir, "refiner_cache"), name="refiner")
    lm = dspy.LM(LLM, temperature=1.0, max_tokens=32000)

    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=log_dir,
            max_metric_calls=args.max_metric_calls,
            cache_evaluation=True,
            track_best_outputs=True,
            parallel=parallel_mode,
            max_workers=max_workers,
        ),
        reflection=ReflectionConfig(
            reflection_minibatch_size=3,
            reflection_lm=LLM,
        ),
    )

    objective = "Generate an LLM prompt that produces fast, correct CUDA kernels outperforming PyTorch baselines."

    seed = {
        "kernel_gen_prompt": KERNEL_GEN_PROMPT,
        "refiner_prompt": REFINER_PROMPT.format(objective=objective),
    }

    fitness_fn = create_fitness_fn(
        lm,
        baselines=baselines,
        use_rag=not args.no_rag,
        max_refinements=args.max_refinements,
        tracker=tracker,
        kernel_gen_cache=kernel_gen_cache,
        refiner_cache=refiner_cache,
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
    print(f"Total metric calls: {tracker.metric_calls}")
    print(f"\nBest speedups per problem:")
    for pid in sorted(tracker.best_speedups.keys()):
        spd = tracker.best_speedups[pid]
        baseline = baselines.get(pid, 0)
        runtime = baseline / spd if spd else None
        status = ">=1.1x" if spd >= 1.1 else (">=1.0x" if spd >= 1.0 else "<1.0x")
        runtime_str = f"{runtime:.2f}ms" if runtime else "N/A"
        print(f"  {pid}: {spd:.2f}x ({runtime_str}/{baseline:.2f}ms) [{status}]")

    # Final fast_p
    f_0 = len(tracker.best_speedups) / max(len(dataset), 1)
    f_1_0 = sum(1 for s in tracker.best_speedups.values() if s >= 1.0) / max(len(dataset), 1)
    f_1_1 = sum(1 for s in tracker.best_speedups.values() if s >= 1.1) / max(len(dataset), 1)
    print(f"\nFast_p(0):   {f_0:.1%} ({len(tracker.best_speedups)}/{len(dataset)}) - any correct")
    print(f"Fast_p(1.0): {f_1_0:.1%} ({sum(1 for s in tracker.best_speedups.values() if s >= 1.0)}/{len(dataset)}) - matches baseline")
    print(f"Fast_p(1.1): {f_1_1:.1%} ({sum(1 for s in tracker.best_speedups.values() if s >= 1.1)}/{len(dataset)}) - 10% faster")

    # Cache stats
    gen_stats = kernel_gen_cache.stats()
    ref_stats = refiner_cache.stats()
    print(f"\nKernel Gen Cache: {gen_stats['hits']} hits, {gen_stats['misses']} misses ({gen_stats['hit_rate']:.1%} hit rate)")
    print(f"Refiner Cache:    {ref_stats['hits']} hits, {ref_stats['misses']} misses ({ref_stats['hit_rate']:.1%} hit rate)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
