import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Max Pooling 3D.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        """
        Initializes the Max Pooling 3D layer.

        Args:
            kernel_size (int): Size of the kernel for the max pooling operation.
            stride (int, optional): Stride of the pooling operation. Defaults to None, which means stride is equal to kernel_size.
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            return_indices (bool, optional): Whether to return indices of the maximum values. Defaults to False.
            ceil_mode (bool, optional): When True, the output size is ceil(input_size / stride) instead of floor. Defaults to False.
        """
        super(Model, self).__init__()
        self.maxpool = nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 3D to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, dim1, dim2, dim3).

        Returns:
            torch.Tensor: Output tensor with Max Pooling 3D applied.
        """
        return self.maxpool(x)

batch_size = 16
channels = 32
dim1 = 128
dim2 = 128
dim3 = 128
kernel_size = 3
stride = 2
padding = 1
dilation = 3

def get_inputs():
    x = torch.rand(batch_size, channels, dim1, dim2, dim3)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]