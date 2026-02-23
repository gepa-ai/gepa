import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

// Macro to check if a tensor is a CUDA tensor
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")

// Macro to check if a tensor is contiguous in memory
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Macro to check both CUDA and contiguity requirements
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Utility function for ceiling division
inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}

__global__ void my_gelu_kernel(float* out, float* inp, int n) {
    // Calculate global thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Return if thread index is out of bounds
    if (i >= n) return;

    // Load input value
    float x = inp[i];

    // Compute GELU (Gaussian Error Linear Unit) activation
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    out[i] = 0.5f * x * (1.0f + tanhf(sqrtf(2.0f/3.141592653589793f) * (x + 0.044715f * (x*x*x))));
}

torch::Tensor my_gelu_out(torch::Tensor output, const torch::Tensor& inp) {
    CHECK_INPUT(inp);  // Validate input tensor
    int n = inp.numel();  // Get total number of elements in input tensor

    // Ensure output tensor has same properties as input tensor
    TORCH_CHECK((output.sizes() == inp.sizes()) || (output.device() == inp.device())
                || (output.scalar_type() == inp.scalar_type()));

    int threads = 256;  // Set number of threads per block

    // Launch CUDA kernel
    my_gelu_kernel<<<cdiv(n, threads), threads>>>(
        output.data_ptr<float>(), inp.data_ptr<float>(), n);

    C10_CUDA_KERNEL_LAUNCH_CHECK();  // Check for CUDA errors
    return output;
}

torch::Tensor my_gelu(const torch::Tensor& inp) {
    CHECK_INPUT(inp);  // Validate input tensor
    auto output = torch::empty_like(inp);  // Create output tensor with same properties as input
    my_gelu_out(output, inp);  // Compute GELU activation
    return output;
}
"""

# Define C++ source code as a string
cpp_src = """
torch::Tensor my_gelu(const torch::Tensor& inp);
torch::Tensor my_gelu_out(torch::Tensor output, const torch::Tensor& inp);
"""

# Load and compile the CUDA extension
fused_gelu = torch.utils.cpp_extension.load_inline(
    name="fused_gelu",  # Name of the extension
    cpp_sources=cpp_src,  # C++ source code
    cuda_sources=source,  # CUDA source code
    functions=['my_gelu', 'my_gelu_out'],  # Functions to expose
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fused_gelu = fused_gelu

    def forward(self, x):
        return self.fused_gelu.my_gelu(x)
