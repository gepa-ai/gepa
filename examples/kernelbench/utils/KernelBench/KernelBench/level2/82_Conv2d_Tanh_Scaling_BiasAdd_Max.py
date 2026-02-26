import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a convolution, applies tanh, scaling, adds a bias term, and then max-pools.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.max_pool = nn.MaxPool2d(pool_kernel_size)

    def forward(self, x):
        # Convolution
        x = self.conv(x)
        # Tanh activation
        x = torch.tanh(x)
        # Scaling
        x = x * self.scaling_factor
        # Bias addition
        x = x + self.bias
        # Max-pooling
        x = self.max_pool(x)
        return x

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3
scaling_factor = 2.0
bias_shape = (out_channels, 1, 1)
pool_kernel_size = 4

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size]