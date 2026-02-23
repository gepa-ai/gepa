import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a convolution, subtracts a value, applies HardSwish, MaxPool, and Mish activation functions.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.pool = nn.MaxPool2d(pool_kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = x - self.subtract_value
        x = torch.nn.functional.hardswish(x)
        x = self.pool(x)
        x = torch.nn.functional.mish(x)
        return x

batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
subtract_value = 0.5
pool_kernel_size = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size]