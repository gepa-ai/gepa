import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a transposed convolution, multiplies by a scalar, applies global average pooling, 
    another global average pooling
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = multiplier

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x * self.multiplier
        x = torch.mean(x, dim=[2, 3], keepdim=True)  # First global average pooling
        x = torch.mean(x, dim=[2, 3], keepdim=True)  # Second global average pooling
        return x

batch_size = 16
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
multiplier = 0.5

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier]