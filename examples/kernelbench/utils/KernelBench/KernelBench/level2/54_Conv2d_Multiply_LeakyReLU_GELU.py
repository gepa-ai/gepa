import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a convolution, multiplies by a learnable scalar, applies LeakyReLU, and then GELU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape)) 
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = x * self.multiplier
        x = self.leaky_relu(x)
        x = torch.nn.functional.gelu(x)
        return x

batch_size = 64
in_channels = 64
out_channels = 64
height, width = 256, 256
kernel_size = 3
multiplier_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, multiplier_shape]