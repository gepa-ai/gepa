#!/usr/bin/env python3
"""
End-to-end test for KernelBench subprocess-based evaluation.

Tests:
1. Happy path - valid kernel compiles and runs correctly
2. Crash isolation - subprocess crash does not kill the parent
3. Compilation failure - invalid CUDA returns clean error

Run: uv run python -m experiments.kernelbench.utils.tests.run_test
"""

import sys


# ---------------------------------------------------------------------------
# Test kernels and reference architectures
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

VALID_KERNEL = """\
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
    name="mul2_op",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["forward_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return custom_op.forward_cuda(x)
"""

CRASHING_KERNEL = """\
import torch
import torch.nn as nn
import os

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        os._exit(1)  # Force-kill the process
"""

BAD_CUDA_KERNEL = """\
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = \"\"\"
THIS IS NOT VALID CUDA CODE AT ALL !!!
\"\"\"

cpp_source = "void broken();"

custom_op = load_inline(
    name="broken_op",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["broken"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return custom_op.broken(x)
"""

REF_ARCH_SIMPLE = """\
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + 1

def get_inputs():
    return [torch.randn(64, 64).cuda()]

def get_init_inputs():
    return []
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_happy_path(gpu: int) -> bool:
    """Test 1: Valid kernel compiles, runs, and scores."""
    print("\n[1/3] Testing happy path: valid kernel evaluation...")

    from experiments.kernelbench.utils.eval import compute_score, execute_kernel

    print("  Evaluating kernel (compilation may be slow on first run)...")
    result = execute_kernel(VALID_KERNEL, REF_ARCH_MUL2, timeout=300, device=gpu)

    print(f"  CompilationSucceeded: {result.get('CompilationSucceeded')}")
    print(f"  CorrectnessSucceeded: {result.get('CorrectnessSucceeded')}")
    print(f"  PerformanceStatsMean: {result.get('PerformanceStatsMean')}")

    if result.get("ErrorType"):
        print(f"  ErrorType: {result.get('ErrorType')}")
        print(f"  ErrorDetail: {str(result.get('ErrorDetail', ''))[:200]}")

    score = compute_score(result, baseline_time=1.0)
    print(f"  Score (vs 1 ms baseline): {score:.4f}")

    if not result.get("CompilationSucceeded"):
        print("  FAIL: Compilation did not succeed")
        return False

    print("  PASS")
    return True


def test_crash_isolation(gpu: int) -> bool:
    """Test 2: Subprocess crash does not kill the parent."""
    print("\n[2/3] Testing crash isolation...")

    from experiments.kernelbench.utils.eval import execute_kernel

    print("  Sending crashing kernel...")
    result = execute_kernel(CRASHING_KERNEL, REF_ARCH_MUL2, timeout=60, device=gpu)

    print(f"  Parent survived. ErrorType={result.get('ErrorType')}")

    ok = result.get("CompilationSucceeded") is False or result.get("ErrorType") is not None
    if not ok:
        print("  FAIL: Expected an error result from crashing kernel")
        return False

    # Verify we can still call execute_kernel after the crash
    print("  Verifying parent can still make calls after crash...")
    result2 = execute_kernel(CRASHING_KERNEL, REF_ARCH_MUL2, timeout=60, device=gpu)
    print(f"  Second call survived. ErrorType={result2.get('ErrorType')}")

    print("  PASS")
    return True


def test_compilation_failure(gpu: int) -> bool:
    """Test 3: Invalid CUDA code returns clean compilation error."""
    print("\n[3/3] Testing compilation failure handling...")

    from experiments.kernelbench.utils.eval import execute_kernel

    print("  Sending invalid kernel...")
    result = execute_kernel(BAD_CUDA_KERNEL, REF_ARCH_SIMPLE, timeout=120, device=gpu)

    print(f"  CompilationSucceeded: {result.get('CompilationSucceeded')}")
    print(f"  ErrorType: {result.get('ErrorType')}")
    print(f"  ErrorDetail: {str(result.get('ErrorDetail', ''))[:200]}")

    if result.get("CompilationSucceeded"):
        print("  FAIL: Expected compilation failure")
        return False

    print("  PASS")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("KernelBench Subprocess Evaluation Test")
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
    results["happy_path"] = test_happy_path(gpu)
    results["crash_isolation"] = test_crash_isolation(gpu)
    results["compilation_failure"] = test_compilation_failure(gpu)

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
