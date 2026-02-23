import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        """
        :param num_input_features: The number of input feature maps
        :param num_output_features: The number of output feature maps
        """
        super(Model, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        """
        :param x: Input tensor of shape (batch_size, num_input_features, height, width)
        :return: Downsampled tensor with reduced number of feature maps
        """
        return self.transition(x)

batch_size = 128
num_input_features = 32
num_output_features = 64
height, width = 256, 256

def get_inputs():
    return [torch.rand(batch_size, num_input_features, height, width)]

def get_init_inputs():
    return [num_input_features, num_output_features]
