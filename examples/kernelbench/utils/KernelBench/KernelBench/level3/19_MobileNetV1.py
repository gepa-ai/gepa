import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, num_classes=1000, input_channels=3, alpha=1.0):
        """
        MobileNetV1 architecture implementation.

        :param num_classes: The number of output classes (default: 1000)
        :param input_channels: The number of input channels (default: 3 for RGB images)
        :param alpha: Width multiplier (default: 1.0)
        """
        super(Model, self).__init__()
        
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )
        
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        
        self.model = nn.Sequential(
            conv_bn(input_channels, int(32 * alpha), 2),
            conv_dw(int(32 * alpha), int(64 * alpha), 1),
            conv_dw(int(64 * alpha), int(128 * alpha), 2),
            conv_dw(int(128 * alpha), int(128 * alpha), 1),
            conv_dw(int(128 * alpha), int(256 * alpha), 2),
            conv_dw(int(256 * alpha), int(256 * alpha), 1),
            conv_dw(int(256 * alpha), int(512 * alpha), 2),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(1024 * alpha), 2),
            conv_dw(int(1024 * alpha), int(1024 * alpha), 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(int(1024 * alpha), num_classes)
    
    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, input_channels, height, width)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Test code
batch_size = 10
input_channels = 3
height = 224
width = 224
num_classes = 1000
alpha = 1.0

def get_inputs():
    return [torch.rand(batch_size, input_channels, height, width)]

def get_init_inputs():
    return [num_classes, input_channels, alpha]