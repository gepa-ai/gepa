import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, num_classes=1000):
        """
        EfficientNetB1 architecture implementation.

        :param num_classes: The number of output classes (default is 1000 for ImageNet).
        """
        super(Model, self).__init__()
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # MBConv blocks
        self.mbconv1 = self._make_mbconv_block(32, 16, 1, 1)
        self.mbconv2 = self._make_mbconv_block(16, 24, 2, 6)
        self.mbconv3 = self._make_mbconv_block(24, 40, 2, 6)
        self.mbconv4 = self._make_mbconv_block(40, 80, 2, 6)
        self.mbconv5 = self._make_mbconv_block(80, 112, 1, 6)
        self.mbconv6 = self._make_mbconv_block(112, 192, 2, 6)
        self.mbconv7 = self._make_mbconv_block(192, 320, 1, 6)
        
        # Final convolutional layer
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        
        # Fully connected layer
        self.fc = nn.Linear(1280, num_classes)
    
    def _make_mbconv_block(self, in_channels, out_channels, stride, expand_ratio):
        """
        Creates a MBConv block.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param stride: Stride of the depthwise convolution.
        :param expand_ratio: Expansion ratio for the hidden layer.
        :return: A sequential MBConv block.
        """
        hidden_dim = round(in_channels * expand_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x):
        """
        Forward pass of the EfficientNetB1 model.

        :param x: Input tensor, shape (batch_size, 3, 240, 240)
        :return: Output tensor, shape (batch_size, num_classes)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = self.mbconv6(x)
        x = self.mbconv7(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# Test code
batch_size = 10
input_shape = (3, 240, 240)
num_classes = 1000

def get_inputs():
    return [torch.rand(batch_size, *input_shape)]

def get_init_inputs():
    return [num_classes]