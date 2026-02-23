import torch
import torch.nn as nn

class Model(nn.Module):
    """
    This model performs a convolution, average pooling, applies sigmoid, and sums the result.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.avg_pool = nn.AvgPool2d(pool_kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.avg_pool(x)
        x = torch.sigmoid(x)
        x = torch.sum(x, dim=[1,2,3]) # Sum over all spatial dimensions
        return x

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 384, 384
kernel_size = 3
pool_kernel_size = 4

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]