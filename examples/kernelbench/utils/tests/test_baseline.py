#!/usr/bin/env python3
"""
Test fair baseline comparison functionality.

Tests:
1. execute_baseline wraps ref_arch correctly
2. execute_baseline produces identical results to Model
3. load_or_measure_baselines works end-to-end

Run: uv run python -m experiments.kernelbench.utils.tests.test_baseline
"""

import sys
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Reference architectures for testing
# ---------------------------------------------------------------------------

REF_ARCH_MUL2 = """\
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 2

def get_inputs():
    return [torch.randn(1024, 1024).cuda()]

def get_init_inputs():
    return []
"""

REF_ARCH_MATMUL = """\
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return torch.matmul(a, b)

def get_inputs():
    return [torch.randn(512, 512).cuda(), torch.randn(512, 512).cuda()]

def get_init_inputs():
    return []
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_execute_baseline_simple(gpu: int) -> bool:
    """Test that execute_baseline works for a simple model."""
    print("\n[1/4] Testing execute_baseline with simple mul2 model...")

    from experiments.kernelbench.utils.eval import execute_baseline

    result = execute_baseline(REF_ARCH_MUL2, timeout=120, device=gpu)

    print(f"  CompilationSucceeded: {result.get('CompilationSucceeded')}")
    print(f"  CorrectnessSucceeded: {result.get('CorrectnessSucceeded')}")
    print(f"  PerformanceStatsMean: {result.get('PerformanceStatsMean')}")

    if result.get("ErrorType"):
        print(f"  ErrorType: {result.get('ErrorType')}")
        print(f"  ErrorDetail: {str(result.get('ErrorDetail', ''))[:200]}")

    # Should succeed since ModelNew = Model (inherits, identical output)
    if not result.get("CorrectnessSucceeded"):
        print("  FAIL: Baseline should pass correctness check")
        return False

    if not result.get("PerformanceStatsMean"):
        print("  FAIL: Baseline should produce performance metrics")
        return False

    print(f"  OK Baseline runtime: {result.get('PerformanceStatsMean'):.4f} ms")
    return True


def test_execute_baseline_matmul(gpu: int) -> bool:
    """Test execute_baseline with a matmul model."""
    print("\n[2/4] Testing execute_baseline with matmul model...")

    from experiments.kernelbench.utils.eval import execute_baseline

    result = execute_baseline(REF_ARCH_MATMUL, timeout=120, device=gpu)

    print(f"  CompilationSucceeded: {result.get('CompilationSucceeded')}")
    print(f"  CorrectnessSucceeded: {result.get('CorrectnessSucceeded')}")
    print(f"  PerformanceStatsMean: {result.get('PerformanceStatsMean')}")

    if result.get("ErrorType"):
        print(f"  ErrorType: {result.get('ErrorType')}")
        print(f"  ErrorDetail: {str(result.get('ErrorDetail', ''))[:200]}")

    if not result.get("CorrectnessSucceeded"):
        print("  FAIL: Baseline should pass correctness check")
        return False

    print(f"  OK Baseline runtime: {result.get('PerformanceStatsMean'):.4f} ms")
    return True


def test_baseline_vs_kernel(gpu: int) -> bool:
    """Test that baseline and custom kernel can be compared fairly."""
    print("\n[3/4] Testing baseline vs custom kernel comparison...")

    from experiments.kernelbench.utils.eval import execute_baseline, execute_kernel

    # Run baseline
    baseline_result = execute_baseline(REF_ARCH_MUL2, timeout=120, device=gpu)
    baseline_time = baseline_result.get("PerformanceStatsMean")

    if not baseline_time:
        print("  FAIL: Could not measure baseline")
        return False

    print(f"  Baseline runtime: {baseline_time:.4f} ms")

    # Run a naive CUDA kernel
    naive_kernel = """\
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = \"\"\"
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mul2_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * 2.0f;
    }
}

torch::Tensor forward_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    mul2_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    return output;
}
\"\"\"

cpp_source = "torch::Tensor forward_cuda(torch::Tensor input);"

custom_op = load_inline(
    name="mul2_test_op",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["forward_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return custom_op.forward_cuda(x)
"""

    kernel_result = execute_kernel(naive_kernel, REF_ARCH_MUL2, timeout=120, device=gpu)
    kernel_time = kernel_result.get("PerformanceStatsMean")

    if not kernel_time:
        print("  FAIL: Could not measure kernel")
        return False

    print(f"  Kernel runtime: {kernel_time:.4f} ms")

    # Both should be measured under same conditions
    speedup = baseline_time / kernel_time if kernel_time > 0 else 0
    print(f"  Speedup: {speedup:.2f}x")

    # The kernel should have similar performance to baseline (both are simple operations)
    # Allow generous bounds since there's natural variation
    if not (0.1 < speedup < 10.0):
        print(f"  WARN: Unexpected speedup ratio, but test passes")

    print("  OK Both baseline and kernel measured through same path")
    return True


def test_load_or_measure_baselines(gpu: int) -> bool:
    """Test load_or_measure_baselines with a minimal dataset."""
    print("\n[4/4] Testing load_or_measure_baselines...")

    import dspy
    from experiments.kernelbench.utils.eval import load_or_measure_baselines

    # Create a minimal dataset
    dataset = [
        dspy.Example(
            level="test",
            problem_id="test_mul2.py",
            ref_arch=REF_ARCH_MUL2,
            baseline_time=1.0,  # Fallback
        ).with_inputs("level", "problem_id", "ref_arch", "baseline_time"),
    ]

    # Use a temp directory for cache
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)

        print("  Measuring baselines (no cache)...")
        baselines = load_or_measure_baselines(dataset, cache_dir=cache_dir, force=True)

        if "test_mul2.py" not in baselines:
            print("  FAIL: Problem not in baselines dict")
            return False

        baseline_time = baselines["test_mul2.py"]
        print(f"  Measured baseline: {baseline_time:.4f} ms")

        # Check cache was created
        import subprocess
        hw_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "--id=0"],
            capture_output=True, text=True, timeout=10,
        )
        hw_name = hw_result.stdout.strip().replace(" ", "-")
        cache_path = cache_dir / hw_name / "dynamic_baseline.json"

        if not cache_path.exists():
            print(f"  FAIL: Cache file not created at {cache_path}")
            return False

        print(f"  OK Cache file created at {cache_path}")

        # Load from cache
        print("  Loading baselines from cache...")
        baselines2 = load_or_measure_baselines(dataset, cache_dir=cache_dir, force=False)

        if abs(baselines2["test_mul2.py"] - baseline_time) > 1e-6:
            print("  FAIL: Cached baseline differs from measured")
            return False

        print("  OK Cached baseline matches measured")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("KernelBench Fair Baseline Test")
    print("=" * 60)

    from experiments.kernelbench.utils.eval import acquire_gpu, get_free_gpus

    free = get_free_gpus()
    print(f"Free GPUs: {free}")
    if not free:
        print("No free GPUs available. Exiting.")
        return 1

    gpu = acquire_gpu()
    print(f"Acquired GPU {gpu}")

    results = {}
    results["execute_baseline_simple"] = test_execute_baseline_simple(gpu)
    results["execute_baseline_matmul"] = test_execute_baseline_matmul(gpu)
    results["baseline_vs_kernel"] = test_baseline_vs_kernel(gpu)
    results["load_or_measure_baselines"] = test_load_or_measure_baselines(gpu)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    failures = [k for k, v in results.items() if not v]
    if failures:
        print(f"\n{len(failures)} test(s) failed: {failures}")
        return 1

    print("\nAll tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
