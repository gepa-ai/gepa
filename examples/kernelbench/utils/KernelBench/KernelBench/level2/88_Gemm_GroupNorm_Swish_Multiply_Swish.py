import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a GEMM, GroupNorm, Swish, Multiply, and Swish operations.
    """
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape)) 

    def forward(self, x):
        # (batch_size, in_features) -> (batch_size, out_features)
        x = self.gemm(x)
        # (batch_size, out_features) -> (batch_size, out_features)
        x = self.group_norm(x)
        # (batch_size, out_features) -> (batch_size, out_features)
        x = x * torch.sigmoid(x)
        # (batch_size, out_features) -> (batch_size, out_features)
        x = x * self.multiply_weight
        # (batch_size, out_features) -> (batch_size, out_features)
        x = x * torch.sigmoid(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
num_groups = 256
multiply_weight_shape = (out_features,)

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, num_groups, multiply_weight_shape]