BACKGROUND = """
Given a PyTorch model (Model), produce ModelNew — a drop-in replacement
(same __init__ args, same forward signature, numerically identical output)
that uses custom CUDA kernels via torch.utils.cpp_extension.load_inline to run faster.

## Code Structure

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = \"\"\"
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void custom_kernel(...) { /* ... */ }

torch::Tensor custom_op(torch::Tensor input, ...) {
    // Launch kernel and return output
}
\"\"\"

cpp_source = "torch::Tensor custom_op(torch::Tensor input, ...);"

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["custom_op"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, ...):  # Same args as Model
        super().__init__()
        ...

    def forward(self, ...):  # Same signature as Model
        return custom_ops.custom_op(...)
```
"""