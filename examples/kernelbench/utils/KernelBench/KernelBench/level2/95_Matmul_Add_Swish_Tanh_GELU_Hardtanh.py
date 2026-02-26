import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, adds a value, applies Swish, Tanh, GELU, and Hardtanh activation functions.
    """
    def __init__(self, in_features, out_features, add_value_shape):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape)) 

    def forward(self, x):
        x = self.matmul(x)
        x = x + self.add_value
        x = torch.sigmoid(x) * x # Swish
        x = torch.tanh(x)
        x = torch.nn.functional.gelu(x) # GELU
        x = torch.nn.functional.hardtanh(x, min_val=-1, max_val=1) # Hardtanh
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
add_value_shape = (out_features,)

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, add_value_shape]