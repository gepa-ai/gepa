"""
Let us think about how to optimize the code step by step.
"""

# Step 1: Let us break down the PyTorch module into step-by-step instructions.
class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        """
        Perform matrix multiplication between two tensors `a` and `b`.

        1. The input tensors `a` and `b` must have compatible shapes for matrix multiplication.
        2. Each element of the resulting tensor is computed as the dot product of a row of `a` and a column of `b`.
        
        Args:
            a (torch.Tensor): A tensor of shape (m, n).
            b (torch.Tensor): A tensor of shape (n, p).

        Returns:
            torch.Tensor: The resulting tensor of shape (m, p) after matrix multiplication.
        """
        return a @ b

#Step 2: Let us describe how each step could be implemented inside of a CUDA kernel.
"""
1. Load the input tensor elements into shared memory:
   - Each thread block loads a tile of `a` and `b` into shared memory to reduce global memory accesses.

2. Compute the dot product:
   - For each element in the resulting matrix `c`, compute the dot product of the corresponding row from `a` and column from `b`.
   - Use shared memory to efficiently accumulate partial sums across threads.

3. Store the result in the output tensor:
   - Each thread computes one element of the result matrix and writes it back to global memory.
"""

# Step 3. Let us put all of the steps together into CUDA kernel code.
