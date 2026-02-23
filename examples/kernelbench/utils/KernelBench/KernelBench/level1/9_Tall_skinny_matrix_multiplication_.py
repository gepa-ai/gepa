import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) where one of the matrices is tall and skinny (M >> N or N >> M)
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): Input matrix of shape (M, K) or (K, M) where M >> N or N >> M.
            B (torch.Tensor): Input matrix of shape (K, N) or (N, K) where M >> N or N >> M.

        Returns:
            torch.Tensor: Output matrix of shape (M, N) or (N, M)
        """
        return torch.matmul(A, B)

M = 16384 * 2
N = 16 * 2

def get_inputs():
    A = torch.rand(M, N)
    B = torch.rand(N, M)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed