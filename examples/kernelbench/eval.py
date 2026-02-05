"""Kernel evaluation utilities."""

import json
import math
import multiprocessing as mp
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import dspy

# Paths - KernelBench is now in the same directory
KERNELBENCH_ROOT = Path(__file__).parent / "KernelBench"


def _get_baseline_path() -> Path:
    """Find baseline times file for current hardware."""
    timing_dir = KERNELBENCH_ROOT / "results" / "timing"
    for hw in ("Tesla-V100-SXM2-32GB-LS", "V100-SXM2-32GB-LS", "H100_PCIe_LambdaLabs", "H100_Modal"):
        path = timing_dir / hw / "baseline_time_torch.json"
        if path.exists():
            return path
    return timing_dir / "H100_PCIe_LambdaLabs" / "baseline_time_torch.json"


BASELINE_PATH = _get_baseline_path()


# --- GPU Utils ---


def get_free_gpus() -> list[int]:
    """Get GPU indices with no running processes."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10,
    )
    all_gpus = set(int(x) for x in result.stdout.strip().split("\n") if x.strip())

    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
    if "Processes:" not in result.stdout:
        return sorted(all_gpus)

    process_section = result.stdout.split("Processes:")[1]
    pattern = re.compile(r"^\|\s+(\d+)\s+.*$", re.MULTILINE)
    busy = set(int(x) for x in pattern.findall(process_section))
    return sorted(all_gpus - busy)


def acquire_gpu(timeout: float | None = None) -> int:
    """Wait for and return a free GPU index."""
    start = time.time()
    while True:
        free = get_free_gpus()
        if free:
            print(f"[acquire_gpu] Got GPU {free[0]}")
            return free[0]
        if timeout and time.time() - start >= timeout:
            raise RuntimeError(f"No free GPU after {timeout}s")
        time.sleep(1.0)


# --- Kernel Execution ---


def _find_cuda_home() -> str | None:
    """Find CUDA toolkit matching torch version."""
    if os.environ.get("CUDA_HOME"):
        return os.environ["CUDA_HOME"]
    try:
        import torch
        ver = torch.version.cuda
        for suffix in (ver, ver.split(".")[0]):
            path = f"/usr/local/cuda-{suffix}"
            if os.path.isfile(f"{path}/bin/nvcc"):
                return path
    except Exception:
        pass
    return None


def _run_eval(queue, code, ref_arch, device, cuda_home):
    """Subprocess target: run kernelbench evaluation."""
    import io
    import sys
    from contextlib import redirect_stdout, redirect_stderr

    # Capture stdout/stderr to include in error feedback
    captured_out = io.StringIO()
    captured_err = io.StringIO()

    try:
        if cuda_home:
            os.environ["CUDA_HOME"] = cuda_home

        import torch
        from kernelbench.eval import eval_kernel_against_ref

        # Reset memory stats before eval
        torch.cuda.reset_peak_memory_stats(device)

        # Run eval with stdout/stderr capture
        with redirect_stdout(captured_out), redirect_stderr(captured_err):
            with tempfile.TemporaryDirectory() as build_dir:
                result = eval_kernel_against_ref(
                    original_model_src=ref_arch,
                    custom_model_src=code,
                    measure_performance=True,
                    build_dir=build_dir,
                    device=device,
                )

        # Get captured output
        stdout_str = captured_out.getvalue()
        stderr_str = captured_err.getvalue()
        combined_output = (stdout_str + "\n" + stderr_str).strip()

        # Handle None result (kernelbench returns None for some compilation errors)
        if result is None:
            # Include captured output which has the actual error details
            error_msg = "Compilation failed (kernelbench returned None)"
            if combined_output:
                error_msg = combined_output[-2000:]  # Last 2000 chars of output
            queue.put({"ok": False, "error": error_msg})
            return

        # Capture profiling info
        profiling_info = {
            "peak_memory_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
            "runtime_stats": getattr(result, "runtime_stats", None),
        }

        queue.put({"ok": True, "result": result.model_dump(), "profiling": profiling_info})
    except Exception as e:
        stdout_str = captured_out.getvalue()
        stderr_str = captured_err.getvalue()
        combined_output = (stdout_str + "\n" + stderr_str).strip()
        error_msg = str(e)
        if combined_output:
            error_msg = f"{e}\n\nOutput:\n{combined_output[-1500:]}"
        queue.put({"ok": False, "error": error_msg})


def execute_baseline(ref_arch: str, timeout: int = 360, device: int | None = None) -> dict:
    """Execute baseline (PyTorch ref_arch) through same path as LLM kernels.

    Wraps ref_arch so ModelNew = Model, ensuring identical measurement conditions.
    """
    wrapped_code = f'''{ref_arch}

# For baseline measurement: ModelNew = Model (identical implementation)
class ModelNew(Model):
    pass
'''
    return execute_kernel(code=wrapped_code, ref_arch=ref_arch, timeout=timeout, device=device)


def execute_kernel(code: str, ref_arch: str, timeout: int = 360, device: int | None = None) -> dict:
    """Execute kernel in subprocess with crash isolation and GPU locking.

    Args:
        code: CUDA kernel code to execute.
        ref_arch: Reference PyTorch architecture.
        timeout: Execution timeout in seconds.
        device: Explicit GPU device index. If None, auto-acquires a free GPU.

    Returns:
        Dict with compilation, correctness, and performance results.
    """
    if device is None:
        device = acquire_gpu()
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_run_eval, args=(queue, code, ref_arch, device, _find_cuda_home()))
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()  # Force kill if SIGTERM didn't work
            proc.join()
        return {"CompilationSucceeded": False, "ErrorType": "Timeout", "ErrorDetail": f"Timeout after {timeout}s"}

    if queue.empty():
        return {"CompilationSucceeded": False, "ErrorType": "Crash", "ErrorDetail": f"Exit code {proc.exitcode}"}

    msg = queue.get_nowait()
    if not msg["ok"]:
        return {"CompilationSucceeded": False, "ErrorType": "Exception", "ErrorDetail": msg["error"]}

    return _parse_result(msg["result"], msg.get("profiling"))


def _parse_result(raw: dict, profiling: dict | None = None) -> dict:
    """Parse kernelbench result to our format with all 6 stages."""
    compiled = raw.get("compiled", False)
    correct = raw.get("correctness", False)
    runtime = raw.get("runtime", -1.0)
    meta = raw.get("metadata", {})

    # Initialize all 6 stages as False
    d = {
        # Stage 1: Code compiles
        "CompilationSucceeded": compiled,
        # Stage 2: Model.__init__() runs
        "ModelInitializeSucceeded": False,
        # Stage 3: forward() runs without crash
        "NoRuntimeErrorDuringCorrectnessCheck": False,
        # Stage 4: Output shape matches
        "NoOutputShapeMismatch": False,
        # Stage 5: Output values match
        "CorrectnessSucceeded": correct,
        # Stage 6: Performance benchmark runs
        "NoRuntimeErrorDuringPerformanceCheck": False,
        "PerformanceStatsMean": None,
        "ErrorType": None,
        "ErrorDetail": None,
    }

    if not compiled:
        d["ErrorType"] = "CompilationFailure"
        d["ErrorDetail"] = str(meta.get("compilation_error", ""))
        return d

    # Compiled - check for runtime errors
    if "runtime_error" in meta:
        d["ErrorType"] = "RuntimeFailure"
        d["ErrorDetail"] = str(meta.get("runtime_error", ""))
        return d

    # Model initialized and forward() ran
    d["ModelInitializeSucceeded"] = True
    d["NoRuntimeErrorDuringCorrectnessCheck"] = True

    # Check correctness
    if not correct:
        issue = str(meta.get("correctness_issue", ""))
        if "shape" in issue.lower():
            d["ErrorType"] = "OutputShapeMismatch"
        else:
            d["NoOutputShapeMismatch"] = True
            d["ErrorType"] = "OutputMismatch"
        d["ErrorDetail"] = issue
        return d

    # Correctness passed
    d["NoOutputShapeMismatch"] = True

    # Check performance measurement
    if runtime > 0:
        d["NoRuntimeErrorDuringPerformanceCheck"] = True
        d["PerformanceStatsMean"] = runtime
    elif "error_during_performance" in meta:
        d["ErrorType"] = "PerformanceError"
        d["ErrorDetail"] = str(meta["error_during_performance"])

    # Add profiling info if available
    if profiling:
        runtime_stats = profiling.get("runtime_stats") or {}
        mean = d.get("PerformanceStatsMean")
        std = runtime_stats.get("std")

        d["ProfilingInfo"] = {
            "peak_memory_mb": profiling.get("peak_memory_mb"),
            "runtime_std_ms": std,
            "runtime_min_ms": runtime_stats.get("min"),
            "runtime_max_ms": runtime_stats.get("max"),
            "cv_percent": (std / mean * 100) if (std and mean) else None,
        }

    return d


# --- Scoring ---


def compute_score(result: dict, baseline_time: float) -> float:
    """Compute score: higher = better. +0.01 per stage, 0.5 at baseline speed."""
    score = -0.07  # Base penalty

    # 6 stages, +0.01 each
    if result.get("CompilationSucceeded"):
        score += 0.01
    if result.get("ModelInitializeSucceeded"):
        score += 0.01
    if result.get("NoRuntimeErrorDuringCorrectnessCheck"):
        score += 0.01
    if result.get("NoOutputShapeMismatch"):
        score += 0.01
    if result.get("CorrectnessSucceeded"):
        score += 0.01
    if result.get("NoRuntimeErrorDuringPerformanceCheck"):
        score += 0.01

    # Performance bonus: exponential decay, 0.5 at baseline
    runtime = result.get("PerformanceStatsMean")
    if runtime and runtime > 0 and baseline_time > 0:
        decay = -math.log(0.5) / baseline_time
        score += math.exp(-decay * runtime)

    return score


def format_error(result: dict) -> str:
    """Format error for LLM feedback."""
    etype = result.get("ErrorType", "Unknown")
    detail = result.get("ErrorDetail", "")
    return f"{etype}: {detail}"


# --- Code Extraction ---


def extract_code(text: str) -> str | None:
    """Extract code from markdown code blocks."""
    blocks = re.findall(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
    return blocks[0].strip() if blocks else None


# --- Dataset ---


LEVEL_PROBLEMS = {
    "level1": [
        "1_Square_matrix_multiplication_.py",
        "3_Batched_matrix_multiplication.py",
        "6_Matmul_with_large_K_dimension_.py",
        "18_Matmul_with_transposed_both.py",
        # "23_Softmax.py",  # OOM on V100-32GB
        # "26_GELU_.py",  # OOM on V100-32GB
        "33_BatchNorm.py",
        # "36_RMSNorm_.py",  # OOM on V100-32GB
        "40_LayerNorm.py",
        "42_Max_Pooling_2D.py",
        "48_Mean_reduction_over_a_dimension.py",
        "54_conv_standard_3D__square_input__square_kernel.py",
        "57_conv_transposed_2D__square_input__square_kernel.py",
        "65_conv_transposed_2D__square_input__asymmetric_kernel.py",
        "77_conv_transposed_3D_square_input_square_kernel___padded____dilated____strided__.py",
        "82_conv_depthwise_2D_square_input_square_kernel.py",
        "86_conv_depthwise_separable_2D.py",
        # "87_conv_pointwise_2D.py",  # OOM on V100-32GB
    ],
    "level2": [
        "1_Conv2D_ReLU_BiasAdd.py",
        "2_ConvTranspose2d_BiasAdd_Clamp_Scaling_Clamp_Divide.py",
        "8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum.py",
        "18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp.py",
        "23_Conv3d_GroupNorm_Mean.py",
        "28_BMM_InstanceNorm_Sum_ResidualAdd_Multiply.py",
        "33_Gemm_Scale_BatchNorm.py",
        "43_Conv3d_Max_LogSumExp_ReLU.py",
    ],
    "level3": [
        "1_MLP.py",
        "5_AlexNet.py",
        "8_ResNetBasicBlock.py",
        "11_VGG16.py",
        "20_MobileNetV2.py",
        "21_EfficientNetMBConv.py",
        "33_VanillaRNN.py",
        "38_LSTMBidirectional.py",
        "43_MinGPTCausalAttention.py",
    ],
}


def load_dataset() -> list[dspy.Example]:
    """Load KernelBench problems as dspy.Examples.

    Note: baseline_time is loaded from JSON as fallback only.
    Use load_or_measure_baselines() for fair dynamic measurement.
    """
    # Load JSON baselines as fallback (may not have all levels)
    fallback_baselines = {}
    if BASELINE_PATH.exists():
        with open(BASELINE_PATH) as f:
            fallback_baselines = json.load(f)
    LEVELS = ["level1", "level2", "level3"]

    examples = []
    for level in LEVELS:
        for problem_id in LEVEL_PROBLEMS.get(level, []):
            path = KERNELBENCH_ROOT / "KernelBench" / level / problem_id
            if not path.exists():
                continue

            ref_arch = path.read_text()
            # Use JSON baseline as fallback, or 1.0 if not available
            baseline_info = fallback_baselines.get(level, {}).get(problem_id, {})
            baseline = baseline_info.get("mean", 1.0)

            examples.append(dspy.Example(
                level=level,
                problem_id=problem_id,
                ref_arch=ref_arch,
                baseline_time=baseline,
            ).with_inputs("level", "problem_id", "ref_arch", "baseline_time"))

    return examples


def _get_hardware_name() -> str:
    """Get GPU hardware name for cache file."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "--id=0"],
        capture_output=True, text=True, timeout=10,
    )
    return result.stdout.strip().replace(" ", "-")


def load_or_measure_baselines(
    dataset: list[dspy.Example],
    cache_dir: Path | str | None = None,
    force: bool = False,
) -> dict[str, float]:
    """Load baselines from cache or measure them through execute_baseline().

    Ensures fair comparison by measuring baselines through same subprocess
    path as LLM kernels.

    Args:
        dataset: List of dspy.Examples with problem_id and ref_arch
        cache_dir: Directory for cache file (default: KERNELBENCH_ROOT/results/timing)
        force: Force re-measurement even if cache exists

    Returns:
        Dict mapping problem_id -> baseline_time_ms
    """
    if cache_dir is None:
        cache_dir = KERNELBENCH_ROOT / "results" / "timing"
    cache_dir = Path(cache_dir)

    hw_name = _get_hardware_name()
    cache_path = cache_dir / hw_name / "dynamic_baseline.json"

    # Load from cache if exists and not forcing
    if cache_path.exists() and not force:
        print(f"Loading baselines from {cache_path}")
        with open(cache_path) as f:
            data = json.load(f)
        # Verify hardware matches
        if data.get("hardware") == hw_name:
            return data["baselines"]
        print(f"Hardware mismatch (cached: {data.get('hardware')}, current: {hw_name}), re-measuring...")

    # Measure baselines
    print(f"Measuring baselines for {len(dataset)} problems on {hw_name}...")
    baselines = {}

    for i, ex in enumerate(dataset):
        problem_id = ex.problem_id
        print(f"  [{i+1}/{len(dataset)}] {problem_id}...", end=" ", flush=True)

        result = execute_baseline(ex.ref_arch, timeout=360)

        if result.get("CorrectnessSucceeded") and result.get("PerformanceStatsMean"):
            baseline_time = result["PerformanceStatsMean"]
            baselines[problem_id] = baseline_time
            print(f"{baseline_time:.4f} ms")
        else:
            # Fall back to JSON baseline if measurement fails
            print(f"FAILED ({result.get('ErrorType', 'Unknown')}), using JSON fallback")
            baselines[problem_id] = ex.baseline_time

    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({
            "hardware": hw_name,
            "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "baselines": baselines,
        }, f, indent=2)
    print(f"Baselines saved to {cache_path}")

    return baselines
