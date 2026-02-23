import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def elementwise_add_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor): 
    tidx, _, _ = cute.arch.thread_idx()  
    bidx, _, _ = cute.arch.block_idx()  
    bdim, _, _ = cute.arch.block_dim()  

    thread_idx = bidx * bdim + tidx

    m, n = gA.shape
    ni = thread_idx % n  
    mi = thread_idx // n  

    a_val = gA[mi, ni]
    b_val = gB[mi, ni]

    gC[mi, ni] = a_val + b_val

@cute.jit
def elementwise_add_host(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    M = mA.shape[0]
    N = mA.shape[1]

    threads_per_block = 256
    total_elems = M * N
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    elementwise_add_kernel(mA, mB, mC).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.compiled = {}

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        M, N = A.shape
        A = A.contiguous().cuda()
        B = B.contiguous().cuda()
        C = torch.empty((M, N), dtype=A.dtype, device=A.device)

        mA = from_dlpack(A, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mB = from_dlpack(B, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mC = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

        key = (A.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(elementwise_add_host, mA, mB, mC)
            self.compiled[key] = compiled

        compiled(mA, mB, mC)
        return C
