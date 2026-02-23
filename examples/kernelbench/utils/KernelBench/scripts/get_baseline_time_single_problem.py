import torch
import numpy as np
from kernelbench.timing import measure_ref_program_time

if __name__ == "__main__":
    ref_arch_name = "softmax"
    ref_arch_src = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x, dim=1)

batch_size = 4096
dim = 65536

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed
    """
    print(measure_ref_program_time(ref_arch_name, ref_arch_src, use_torch_compile=False, timing_method="cuda_event", precision="fp32"))