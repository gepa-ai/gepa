import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D convolution, scales the output, applies tanh, multiplies by a scaling factor, and applies sigmoid.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape)) 

    def forward(self, x):
        x = self.conv(x)
        x = x * self.scaling_factor 
        x = torch.tanh(x)
        x = x * self.bias
        x = torch.sigmoid(x)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 64, 64
kernel_size = 3
scaling_factor = 2
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape]