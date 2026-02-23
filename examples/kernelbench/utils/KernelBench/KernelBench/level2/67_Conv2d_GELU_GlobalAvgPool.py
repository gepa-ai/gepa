import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a convolution, applies GELU, and then performs global average pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels)
        """
        x = self.conv(x)
        x = torch.nn.functional.gelu(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]