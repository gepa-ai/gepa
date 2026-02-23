import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, num_classes=1000):
        """
        MobileNetV2 architecture implementation in PyTorch.

        :param num_classes: The number of output classes. Default is 1000.
        """
        super(Model, self).__init__()
        
        def _make_divisible(v, divisor, min_value=None):
            """
            This function ensures that the number of channels is divisible by the divisor.
            """
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            # Make sure that round down does not go down by more than 10%.
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        def _inverted_residual_block(inp, oup, stride, expand_ratio):
            """
            Inverted Residual Block for MobileNetV2.
            """
            hidden_dim = int(inp * expand_ratio)
            use_res_connect = stride == 1 and inp == oup

            layers = []
            if expand_ratio != 1:
                # Pointwise convolution
                layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
                layers.append(nn.BatchNorm2d(hidden_dim))
                layers.append(nn.ReLU6(inplace=True))

            layers.extend([
                # Depthwise convolution
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # Pointwise linear convolution
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ])

            if use_res_connect:
                return nn.Sequential(*layers), True
            else:
                return nn.Sequential(*layers), False

        # MobileNetV2 architecture
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # Building first layer
        features = [nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(input_channel),
                    nn.ReLU6(inplace=True)]

        # Building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(_inverted_residual_block(input_channel, output_channel, stride, expand_ratio=t)[0])
                input_channel = output_channel

        # Building last several layers
        features.append(nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False))
        features.append(nn.BatchNorm2d(last_channel))
        features.append(nn.ReLU6(inplace=True))

        # Final layer
        features.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.features = nn.Sequential(*features)

        # Linear layer
        self.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(last_channel, num_classes),
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass of the MobileNetV2 model.

        :param x: The input tensor, shape (batch_size, 3, 224, 224)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Test code
batch_size = 10
num_classes = 1000

def get_inputs():
    return [torch.rand(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [num_classes]