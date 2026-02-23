import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a 3D convolution, applies minimum operation along a specific dimension, 
    and then applies softmax.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, H, W)
        """
        x = self.conv(x)
        x = torch.min(x, dim=self.dim)[0]  # Apply minimum along the specified dimension
        x = torch.softmax(x, dim=1)  # Apply softmax along the channel dimension
        return x

batch_size = 128
in_channels = 3
out_channels = 24  # Increased output channels
D, H, W = 24, 32, 32  # Increased depth
kernel_size = 3
dim = 2  # Dimension along which to apply minimum operation (e.g., depth)

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, dim]