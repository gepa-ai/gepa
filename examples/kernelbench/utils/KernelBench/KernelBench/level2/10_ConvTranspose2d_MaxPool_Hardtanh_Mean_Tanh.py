import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a transposed convolution, followed by max pooling, hardtanh activation, mean operation, and tanh activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=maxpool_stride)
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.maxpool(x)
        x = self.hardtanh(x)
        x = torch.mean(x, dim=(2, 3), keepdim=True)
        x = torch.tanh(x)
        return x

batch_size = 128
in_channels  = 64  
out_channels = 64  
height = width = 256  
kernel_size  = 3
stride = 1
padding = 1
maxpool_kernel_size = 2
maxpool_stride = 2
hardtanh_min = -1
hardtanh_max = 1

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max]