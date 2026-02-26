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


constexpr int TILE_SIZE = 16;  // Size of each tile for matrix multiplication

/**
 * @brief Tiled matrix multiplication kernel.
 * 
 * This kernel performs matrix multiplication using shared memory tiles to improve performance.
 *
 * @param out Pointer to the output matrix
 * @param M Pointer to the first input matrix
 * @param N Pointer to the second input matrix
 * @param h Height of matrix M
 * @param w Width of matrix N
 * @param k Width of matrix M / Height of matrix N
 */
__global__ void tiled_matmul_kernel(float* out, float* M, float* N, int h, int w, int k) {
    __shared__ float M_tile[TILE_SIZE][TILE_SIZE];  // Shared memory for M matrix tile
    __shared__ float N_tile[TILE_SIZE][TILE_SIZE];  // Shared memory for N matrix tile

    // Thread indices within a tile
    int ir = threadIdx.y;
    int ic = threadIdx.x;

    // Global thread indices
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    float res = 0.0f;  // Accumulator for dot product result

    // Iterate over tiles
    for (int K_tileidx = 0; K_tileidx < (k + TILE_SIZE -1) / TILE_SIZE; K_tileidx++) {
        // Load data into shared memory tiles, with bounds checking
        M_tile[ir][ic] = (((r < h) && (K_tileidx * TILE_SIZE + ic < k)) ? M[r * k + K_tileidx * TILE_SIZE + ic] : 0.f);
        N_tile[ir][ic] = ((((K_tileidx * TILE_SIZE + ir) < k) && (c < w)) ? N[(K_tileidx * TILE_SIZE + ir) * w + c] : 0.f);

        __syncthreads();  // Ensure all threads have loaded data before computation

        // Compute dot product for this tile
        for (int idx = 0; idx < TILE_SIZE; idx++) {
           res += M_tile[ir][idx] * N_tile[idx][ic];
        }

        __syncthreads();  // Ensure all computations are done before loading next tile
    }

    // Write result to global memory if within bounds
    if ((r < h) && (c < w)) {
        out[r * w + c] = res;
    }
}

/**
 * @brief Wrapper function for tiled matrix multiplication kernel.
 * 
 * This function checks input tensors, sets up kernel parameters, and launches the CUDA kernel.
 *
 * @param m First input matrix
 * @param n Second input matrix
 * @return torch::Tensor Result of matrix multiplication
 */
torch::Tensor tiled_matmul_cuda(const torch::Tensor& m, const torch::Tensor& n) {
    CHECK_INPUT(m); CHECK_INPUT(n);
    int h = m.size(0);
    int w = n.size(1);
    int k = m.size(1);
    TORCH_CHECK(k==n.size(0), "Size mismatch");

    auto output = torch::empty({h, w}, m.options());

    // Define thread block and grid dimensions
    dim3 tpb(TILE_SIZE, TILE_SIZE);
    dim3 blocks(cdiv(w, tpb.x), cdiv(h, tpb.y));

    // Launch kernel
    tiled_matmul_kernel<<<blocks, tpb>>>(
        output.data_ptr<float>(), m.data_ptr<float>(), n.data_ptr<float>(), h, w, k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

# C++ interface definition
cpp_src = """
torch::Tensor tiled_matmul_cuda(const torch::Tensor& m, const torch::Tensor& n);
"""

# Load the CUDA kernel as a PyTorch C++ extension
tiled_matmul = torch.utils.cpp_extension.load_inline(
    "tiled_matmul",  # Name of the extension
    cpp_sources=cpp_src,  # C++ interface
    cuda_sources=source,  # CUDA source code
    functions=['tiled_matmul_cuda'],  # Exported functions
    extra_cuda_cflags=['--ptxas-options=-v'],  # Additional CUDA compilation flags
    verbose=True              # Enable verbose output during compilation
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tiled_matmul = tiled_matmul

    def forward(self, a, b):
        return self.tiled_matmul.tiled_matmul_cuda(a, b)
