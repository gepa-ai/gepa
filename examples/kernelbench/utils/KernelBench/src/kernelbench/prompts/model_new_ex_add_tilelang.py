import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_elementwise_add_kernel(M: int, N: int, block_M: int = 128, block_N: int = 256, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def elementwise_add_kernel(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            start_x = bx * block_N
            start_y = by * block_M

            for local_y, local_x in T.Parallel(block_M, block_N):
                y = start_y + local_y
                x = start_x + local_x

                C[y, x] = A[y, x] + B[y, x]

    return tilelang.compile(elementwise_add_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self._kernel_cache = {}

    def _get_kernel(self, M: int, N: int, tl_dtype: str):
        key = (M, N, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_elementwise_add_kernel(M, N, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A_c = A.contiguous()
        B_c = B.contiguous()
        
        # Get original shape for reshaping output
        original_shape = A_c.shape
        
        A_c = A_c.view(-1, A_c.size(-1))
        B_c = B_c.view(-1, B_c.size(-1))

        M, N = A_c.shape
        kernel = self._get_kernel(M, N, "float16")
        C = kernel(A_c, B_c)

        return C.view(original_shape)