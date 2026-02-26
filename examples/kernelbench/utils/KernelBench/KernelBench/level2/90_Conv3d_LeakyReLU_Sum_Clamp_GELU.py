import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D convolution, applies LeakyReLU, sums with a tensor, clamps, and applies GELU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))

    def forward(self, x):
        x = self.conv(x)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)
        x = x + self.sum_tensor
        x = torch.clamp(x, min=-1.0, max=1.0)
        x = torch.nn.functional.gelu(x)
        return x

batch_size = 128
in_channels = 8
out_channels = 64
depth, height, width = 16, 64, 64
kernel_size = 3
sum_tensor_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, sum_tensor_shape]