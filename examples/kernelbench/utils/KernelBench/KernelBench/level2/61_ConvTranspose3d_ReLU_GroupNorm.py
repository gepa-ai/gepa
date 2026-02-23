import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a transposed 3D convolution, applies ReLU, and then applies group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        self.relu = nn.ReLU()
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D, H, W).
        """
        x = self.conv_transpose(x)
        x = self.relu(x)
        x = self.group_norm(x)
        return x

batch_size = 16
in_channels = 64
out_channels = 128
D, H, W = 32, 32, 32
kernel_size = 3
groups = 8
bias = False

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups, bias]