import torch
import torch.nn as nn
import torch.nn.functional as F

class ShuffleNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        """
        ShuffleNet unit implementation.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param groups: Number of groups for group convolution.
        """
        super(ShuffleNetUnit, self).__init__()
        
        # Ensure the output channels are divisible by groups
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4
        
        # First 1x1 group convolution
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        # Depthwise 3x3 convolution
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        # Second 1x1 group convolution
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Shuffle operation
        self.shuffle = ChannelShuffle(groups)
        
        # Shortcut connection if input and output channels are the same
        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        """
        Forward pass for ShuffleNet unit.

        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.shuffle(out)
        out = F.relu(self.bn3(self.conv3(out)))
        
        out += self.shortcut(x)
        return out

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        """
        Channel shuffle operation.

        :param groups: Number of groups for shuffling.
        """
        super(ChannelShuffle, self).__init__()
        self.groups = groups
    
    def forward(self, x):
        """
        Forward pass for channel shuffle.

        :param x: Input tensor, shape (batch_size, channels, height, width)
        :return: Output tensor, shape (batch_size, channels, height, width)
        """
        batch_size, channels, height, width = x.size()
        channels_per_group = channels // self.groups
        
        # Reshape
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        
        # Transpose
        x = x.transpose(1, 2).contiguous()
        
        # Flatten
        x = x.view(batch_size, -1, height, width)
        
        return x

class Model(nn.Module):
    def __init__(self, num_classes=1000, groups=3, stages_repeats=[3, 7, 3], stages_out_channels=[24, 240, 480, 960]):
        """
        ShuffleNet architecture.

        :param num_classes: Number of output classes.
        :param groups: Number of groups for group convolution.
        :param stages_repeats: List of ints specifying the number of repeats for each stage.
        :param stages_out_channels: List of ints specifying the output channels for each stage.
        """
        super(Model, self).__init__()
        
        self.conv1 = nn.Conv2d(3, stages_out_channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stages_out_channels[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.stage2 = self._make_stage(stages_out_channels[0], stages_out_channels[1], stages_repeats[0], groups)
        self.stage3 = self._make_stage(stages_out_channels[1], stages_out_channels[2], stages_repeats[1], groups)
        self.stage4 = self._make_stage(stages_out_channels[2], stages_out_channels[3], stages_repeats[2], groups)
        
        self.conv5 = nn.Conv2d(stages_out_channels[3], 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(1024)
        
        self.fc = nn.Linear(1024, num_classes)
    
    def _make_stage(self, in_channels, out_channels, repeats, groups):
        """
        Helper function to create a stage of ShuffleNet units.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param repeats: Number of ShuffleNet units in the stage.
        :param groups: Number of groups for group convolution.
        :return: nn.Sequential containing the stage.
        """
        layers = []
        layers.append(ShuffleNetUnit(in_channels, out_channels, groups))
        for _ in range(1, repeats):
            layers.append(ShuffleNetUnit(out_channels, out_channels, groups))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass for ShuffleNet.

        :param x: Input tensor, shape (batch_size, 3, height, width)
        :return: Output tensor, shape (batch_size, num_classes)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

# Test code
batch_size = 10
input_channels = 3
height = 224
width = 224
num_classes = 1000

def get_inputs():
    return [torch.rand(batch_size, input_channels, height, width)]

def get_init_inputs():
    return [num_classes]