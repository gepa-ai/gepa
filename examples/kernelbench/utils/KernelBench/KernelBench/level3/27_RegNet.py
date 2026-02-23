import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_channels, stages, block_widths, output_classes):
        """
        :param input_channels: int, Number of input channels for the first layer
        :param stages: int, Number of stages in the RegNet architecture
        :param block_widths: List[int], Width (number of channels) for each block in the stages
        :param output_classes: int, Number of output classes for classification
        """
        super(Model, self).__init__()

        self.stages = stages
        self.block_widths = block_widths
        
        layers = []
        current_channels = input_channels
        
        # Construct the stages with their respective blocks
        for i in range(stages):
            layers.append(self._make_stage(current_channels, block_widths[i]))
            current_channels = block_widths[i]
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Final fully connected layer for classification
        self.fc = nn.Linear(block_widths[-1], output_classes)
    
    def _make_stage(self, in_channels, out_channels):
        """
        Creates a simple block for each stage.
        :param in_channels: int, number of input channels
        :param out_channels: int, number of output channels
        :return: nn.Sequential block with convolutional layers
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        """
        Forward pass through the RegNet model.
        :param x: torch.Tensor of shape (batch_size, input_channels, height, width)
        :return: torch.Tensor of shape (batch_size, output_classes)
        """
        x = self.feature_extractor(x)
        x = torch.mean(x, dim=[2, 3])  # Global Average Pooling
        x = self.fc(x)
        return x

# Test code for the RegNet model
batch_size = 8
input_channels = 3
image_height, image_width = 224, 224
stages = 3
block_widths = [64, 128, 256]
output_classes = 10

def get_inputs():
    """ Generates random input tensor of shape (batch_size, input_channels, height, width) """
    return [torch.rand(batch_size, input_channels, image_height, image_width)]

def get_init_inputs():
    """ Initializes model parameters """
    return [input_channels, stages, block_widths, output_classes]