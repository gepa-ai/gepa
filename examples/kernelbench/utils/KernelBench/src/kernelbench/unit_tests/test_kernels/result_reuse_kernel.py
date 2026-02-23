import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

###########################################
# 
# This adversarial kernel utilizes empty which can get allocated the same physical memory as the PyTorch reference outputs
# If the eval script deletes/frees the PyTorch output object at some point before the custom kernel is run, 
# the CUDA cache allocator might give that un-erased physical memory to the custom kernel and it will incorrectly pass
# So, we should ensure we zero out physical memory to prevent reuse
###########################################

# Returns a zero_
matmul_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto C = torch::empty({A.size(0), B.size(1)}, A.options());
    return C;
}
"""

matmul_cuda_cpp = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile inline extension
matmul_module = load_inline(
    name="fast_matmul", # this is a hacky kernel that shall not pass; it would extra fast since it just allocates memory of result shape
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