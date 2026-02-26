import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D convolution, applies Group Normalization, computes the mean
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        x = self.conv(x)
        x = self.group_norm(x)
        x = x.mean(dim=[1, 2, 3, 4]) # Compute mean across all dimensions except batch
        return x

batch_size = 128
in_channels = 3
out_channels = 24
D, H, W = 24, 32, 32
kernel_size = 3
num_groups = 8

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups]