import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

tk_root = os.environ.get("THUNDERKITTENS_ROOT", "/root/ThunderKittens")
tk_include_path = os.path.join(tk_root, "include")
tk_prototype_path = os.path.join(tk_root, "prototype")

extra_include_paths = [tk_root, tk_include_path]
if os.path.isdir(tk_prototype_path):
    extra_include_paths.append(tk_prototype_path)

thunderkittens_add_source = """
#include "kittens.cuh"
#include <torch/extension.h>

using namespace kittens;

constexpr int BLOCK_SIZE = 16;

#define NUM_WORKERS (1)
#define NUM_THREADS (NUM_WORKERS * kittens::WARP_THREADS)

struct add_globals {
    using sub_tile = st_bf<BLOCK_SIZE, BLOCK_SIZE>;
    using tile_gl = gl<bf16, 1, 1, -1, -1, sub_tile>;
    tile_gl A;
    tile_gl B;
    tile_gl C;
};

__global__ void add_tk(const __grid_constant__ add_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<BLOCK_SIZE, BLOCK_SIZE> &As = al.allocate<st_bf<BLOCK_SIZE, BLOCK_SIZE>>();
    st_bf<BLOCK_SIZE, BLOCK_SIZE> &Bs = al.allocate<st_bf<BLOCK_SIZE, BLOCK_SIZE>>();
    rt_bf<BLOCK_SIZE, BLOCK_SIZE> A_reg;
    rt_bf<BLOCK_SIZE, BLOCK_SIZE> B_reg;
    rt_bf<BLOCK_SIZE, BLOCK_SIZE> C_reg;
    int col = blockIdx.x;
    int row = blockIdx.y;
    // Load A and B tiles from global to shared
    kittens::warp::load(As, g.A, {0, 0, row, col});
    kittens::warp::load(Bs, g.B, {0, 0, row, col});
    __syncthreads();
    // Load from shared to register
    kittens::warp::load(A_reg, As);
    kittens::warp::load(B_reg, Bs);
    __syncthreads();
    // Element-wise add: C = A + B
    kittens::warp::add(C_reg, A_reg, B_reg);
    __syncthreads();
    // Store result back to global
    kittens::warp::store(g.C, C_reg, {0, 0, row, col});
}

torch::Tensor thunderkittens_add_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must be bfloat16");
    TORCH_CHECK(B.dtype() == torch::kBFloat16, "B must be bfloat16");
    
    int M = A.size(0);
    int N = A.size(1);
    
    auto C = torch::empty_like(A);
    
    using tile_gl = add_globals::tile_gl;
    tile_gl a_arg{(bf16*)A.data_ptr(), nullptr, nullptr, (size_t)M, (size_t)N};
    tile_gl b_arg{(bf16*)B.data_ptr(), nullptr, nullptr, (size_t)M, (size_t)N};
    tile_gl c_arg{(bf16*)C.data_ptr(), nullptr, nullptr, (size_t)M, (size_t)N};
    add_globals g{a_arg, b_arg, c_arg};
    
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    unsigned long mem_size = 50480;
    cudaFuncSetAttribute(add_tk, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    add_tk<<<blocks, NUM_THREADS, mem_size>>>(g);
    
    return C;
}
"""

thunderkittens_add_cpp_source = """
torch::Tensor thunderkittens_add_cuda(torch::Tensor A, torch::Tensor B);
"""

thunderkittens_add = load_inline(
    name="thunderkittens_add",
    cpp_sources=thunderkittens_add_cpp_source,
    cuda_sources=thunderkittens_add_source,
    functions=["thunderkittens_add_cuda"],
    verbose=True,
    extra_include_paths=extra_include_paths,
    extra_cflags=["-std=c++20", "-O3", "-DNDEBUG"],
    extra_ldflags=["-lcuda"],
    extra_cuda_cflags=[
        "-std=c++20",
        "-O3",
        "-DNDEBUG",
        "-arch=sm_90a",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-DKITTENS_HOPPER",
        "-DKITTENS_BLACKWELL",
        "-diag-suppress=20012",
        "-Xcompiler", "-fPIC",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
    ],
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.thunderkittens_add = thunderkittens_add

    def forward(self, a, b):
        return self.thunderkittens_add.thunderkittens_add_cuda(a, b)
