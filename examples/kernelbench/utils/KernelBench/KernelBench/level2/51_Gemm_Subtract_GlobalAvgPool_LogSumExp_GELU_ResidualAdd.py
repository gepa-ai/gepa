import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a series of operations: Gemm, Subtract, GlobalAvgPool, LogSumExp, GELU, and ResidualAdd.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        original_x = x.clone().detach()
        # Gemm
        x = self.gemm(x)

        # Subtract
        x = x - self.subtract

        # GlobalAvgPool
        x = torch.mean(x, dim=1, keepdim=True)

        # LogSumExp
        x = torch.logsumexp(x, dim=1, keepdim=True)

        # GELU
        x = torch.nn.functional.gelu(x)

        # ResidualAdd
        x = x + original_x

        return x

batch_size = 2048
in_features = 8192
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]