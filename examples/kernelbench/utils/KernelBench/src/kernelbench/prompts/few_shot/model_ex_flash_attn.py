import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Model that performs an attention operation
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, Q, K, V):
        att = (Q @ K.transpose(-2, -1) * (1.0 / math.sqrt(K.size(-1))))
        att = F.softmax(att, dim=-1)
        y = att @ V
        return y

batch_size = 32
n_head = 12
seq_len = 64
head_embd = 32

def get_inputs():
    # randomly generate input tensors based on the model architecture
    Q = torch.randn(batch_size, n_head, seq_len, head_embd)
    K = torch.randn(batch_size, n_head, seq_len, head_embd)
    V = torch.randn(batch_size, n_head, seq_len, head_embd)
    return [Q, K, V]


def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return []
