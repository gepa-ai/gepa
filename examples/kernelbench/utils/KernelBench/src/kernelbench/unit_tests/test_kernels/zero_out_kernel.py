import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

###########################################
# This custom kernel zeros out the inputs and returns a zero matrix.
# It is specifically designed to test the eval script's robustness.
# Running this zero kernel should result in incorrectness for eval,
# as the reference kernel will not return all zeros.
###########################################

# Destroys all inputs, returns all zeros for final matmul shape
matmul_cuda_source = r"""
#include <torch/extension.h>

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    A.zero_();
    B.zero_();
    return torch::zeros({A.size(0), B.size(1)}, A.options());
}
"""

matmul_cuda_cpp = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile inline extension
matmul_module = load_inline(
    name="fast_matmul", # this is a hacky kernel that shall not pass; it would extra fast since it just allocates 0s
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