#!/usr/bin/env python3
"""
End-to-end scoring validation for KernelBench evaluation.

Evaluates kernel variants (optimized, wrong, broken) and verifies the scoring gradient:
  correct > wrong > broken

Run: uv run python -m examples.kernelbench.tests.test_scoring_e2e
"""

import sys


REF_ARCH = """\
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 2

def get_inputs():
    return [torch.randn(4096, 4096).cuda()]

def get_init_inputs():
    return []
"""

# 1. Optimized: float4 vectorized loads
KERNEL_OPTIMIZED = """\
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = \"\"\"
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mul2_vec4_kernel(const float4* __restrict__ input,
                                  float4* __restrict__ output, int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        float4 v = input[idx];
        v.x *= 2.0f;
        v.y *= 2.0f;
        v.z *= 2.0f;
        v.w *= 2.0f;
        output[idx] = v;
    }
}

torch::Tensor forward_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    int n4 = size / 4;
    const int threads = 256;
    const int blocks = (n4 + threads - 1) / threads;
    mul2_vec4_kernel<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(input.data_ptr<float>()),
        reinterpret_cast<float4*>(output.data_ptr<float>()),
        n4
    );
    return output;
}
\"\"\"

cpp_source = "torch::Tensor forward_cuda(torch::Tensor input);"

custom_op = load_inline(
    name="mul2_vec4_op",
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

# 2. Wrong output: x * 3 instead of x * 2
KERNEL_WRONG = """\
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = \"\"\"
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mul3_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * 3.0f;
    }
}

torch::Tensor forward_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    mul3_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    return output;
}
\"\"\"

cpp_source = "torch::Tensor forward_cuda(torch::Tensor input);"

custom_op = load_inline(
    name="mul3_wrong_op",
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

# 3. Broken: invalid CUDA code
KERNEL_BROKEN = """\
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = \"\"\"
THIS IS NOT VALID CUDA CODE !!!
\"\"\"

cpp_source = "void broken();"

custom_op = load_inline(
    name="broken_op",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["broken"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return custom_op.broken(x)
"""


def main():
    print("=" * 70)
    print("KernelBench End-to-End Scoring Validation")
    print("=" * 70)

    from examples.kernelbench.eval import (
        acquire_gpu,
        compute_score,
        execute_baseline,
        execute_kernel,
        get_free_gpus,
    )

    free = get_free_gpus()
    print(f"Free GPUs: {free}")
    if not free:
        print("No free GPUs available. Exiting.")
        return 1

    gpu = acquire_gpu()
    print(f"Using GPU {gpu}")

    # Phase 1: Get baseline from PyTorch reference
    print("\n" + "-" * 70)
    print("Phase 1: Measure PyTorch baseline")
    print("-" * 70)

    baseline_result = execute_baseline(REF_ARCH, timeout=120, device=gpu)
    baseline_time = baseline_result.get("PerformanceStatsMean")

    if baseline_time and baseline_time > 0:
        print(f"  Baseline runtime: {baseline_time:.4f} ms")
    else:
        print(f"  WARNING: Baseline failed. Using fallback 1.0 ms.")
        baseline_time = 1.0

    # Phase 2: Evaluate variants
    print("\n" + "-" * 70)
    print(f"Phase 2: Evaluate variants (baseline = {baseline_time:.4f} ms)")
    print("-" * 70)

    variants = [
        ("optimized", KERNEL_OPTIMIZED),
        ("wrong", KERNEL_WRONG),
        ("broken", KERNEL_BROKEN),
    ]

    results = []
    for i, (name, kernel_code) in enumerate(variants):
        print(f"\n  [{i+1}/{len(variants)}] {name}...")
        result = execute_kernel(kernel_code, REF_ARCH, timeout=120, device=gpu)
        score = compute_score(result, baseline_time=baseline_time)
        runtime = result.get("PerformanceStatsMean")

        print(f"    Compiled: {result.get('CompilationSucceeded', False)}, "
              f"Correct: {result.get('CorrectnessSucceeded', False)}, "
              f"Score: {score:.4f}")

        results.append((name, score, runtime, result))

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for name, score, runtime, _ in results:
        rt = f"{runtime:.4f} ms" if runtime else "N/A"
        print(f"  {name:<12} score={score:.4f}  runtime={rt}")

    # Gradient verification
    print("\n" + "-" * 70)
    print("GRADIENT CHECK: optimized > wrong > broken")
    print("-" * 70)

    scores = [s for _, s, _, _ in results]
    all_ok = True

    if scores[0] > scores[1]:
        print(f"  OK   optimized ({scores[0]:.4f}) > wrong ({scores[1]:.4f})")
    else:
        print(f"  FAIL optimized ({scores[0]:.4f}) <= wrong ({scores[1]:.4f})")
        all_ok = False

    if scores[1] > scores[2]:
        print(f"  OK   wrong ({scores[1]:.4f}) > broken ({scores[2]:.4f})")
    else:
        print(f"  FAIL wrong ({scores[1]:.4f}) <= broken ({scores[2]:.4f})")
        all_ok = False

    if scores[2] < 0:
        print(f"  OK   broken ({scores[2]:.4f}) < 0")
    else:
        print(f"  FAIL broken ({scores[2]:.4f}) >= 0")
        all_ok = False

    print()
    if all_ok:
        print("All checks passed!")
        return 0
    else:
        print("Some checks failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
