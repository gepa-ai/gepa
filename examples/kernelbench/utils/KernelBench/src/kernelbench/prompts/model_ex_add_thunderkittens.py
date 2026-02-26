import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return a + b


def get_inputs():
    # Use shapes compatible with ThunderKittens 16x16 tiles, bf16 dtype
    a = torch.randn(128, 128, dtype=torch.bfloat16).cuda()
    b = torch.randn(128, 128, dtype=torch.bfloat16).cuda()
    return [a, b]


def get_init_inputs():
    return []

