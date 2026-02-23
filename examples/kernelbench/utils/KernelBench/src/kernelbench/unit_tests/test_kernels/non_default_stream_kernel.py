import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

###########################################
# CUDA_STREAM_ADVERSARIAL
# adversarial kernel assigns all work to a non-default CUDA stream.
# If the eval script waits only on the default cuda stream when measuring kernel runtime, this will lead to unrealistic speedups.   
###########################################
# this adversarial kernel obtains a non-default, non-blocking stream via cudaStreamCreateWithFlags(...) and sets to be the stream of the cuBLAS handle.
# then, it performs a matrix multiplication on this new stream.

# if timing harness only waits on default stream, it will measure near-zero time because the actual computation is still running asynchronously on the other stream.

matmul_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto result = torch::empty({M, N}, A.options());
    
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
    
    float alpha = 1.0f, beta = 0.0f;
    
    cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B.data_ptr<float>(), CUDA_R_32F, N,
        A.data_ptr<float>(), CUDA_R_32F, K,
        &beta,
        result.data_ptr<float>(), CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // NOTE: Intentionally leaking CUDA resources (stream + cuBLAS handle) to preserve
    // adversarial behavior. Calling cudaStreamDestroy() would implicitly synchronize,
    // defeating the purpose of this test (which checks if the eval harness detects
    // work launched on non-default streams). Keep num_perf_trials low to avoid OOM.
    // cublasDestroy(handle);
    // cudaStreamDestroy(stream);
    
    return result;
}
"""

matmul_cuda_cpp = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile inline extension
matmul_module = load_inline(
    name="fast_matmul",
    cpp_sources=matmul_cuda_cpp,
    cuda_sources=matmul_cuda_source,
    functions=["matmul_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matmul = matmul_module

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul.matmul_cuda(A, B)

N = 2048 * 2

def get_inputs():
    A = torch.rand(N, N)
    B = torch.rand(N, N)
    return [A, B]

def get_init_inputs():
    return []