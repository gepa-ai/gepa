# PyTorch CUDA Extensions with load_inline

## Required Headers

For PyTorch CUDA extensions using `torch.utils.cpp_extension.load_inline`, use these headers:

```cpp
// In .cpp file - torch/extension.h includes everything needed
#include <torch/extension.h>

// In .cu file - for CUDA kernels
#include <torch/extension.h>
#include <cuda_runtime.h>

// DO NOT use #include <cuda.h> - it's not needed and may cause errors
```

## Getting CUDA Stream

```cpp
// CORRECT - include the header first
#include <ATen/cuda/CUDAContext.h>

// Then use:
cudaStream_t stream = at::cuda::getCurrentCUDAStream();

// Alternative using c10:
#include <c10/cuda/CUDAStream.h>
cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
```

## load_inline Structure

When using `load_inline`, you provide:
- `cpp_sources`: C++ code with the Python bindings (PYBIND11_MODULE is auto-generated if you use `functions` parameter)
- `cuda_sources`: CUDA kernel code (.cu)
- `functions`: List of function names to expose to Python

```python
from torch.utils.cpp_extension import load_inline

# Example structure
cuda_source = '''
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void my_kernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * 2.0f;
    }
}

torch::Tensor my_cuda_op(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int n = input.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    my_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        n
    );
    return output;
}
'''

cpp_source = "torch::Tensor my_cuda_op(torch::Tensor input);"

module = load_inline(
    name='my_extension',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['my_cuda_op'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)
```

## Common Errors and Fixes

### Error: `cuda.h: No such file or directory`
**Cause**: Using `#include <cuda.h>` which is the driver API header
**Fix**: Use `#include <cuda_runtime.h>` instead, or just `#include <torch/extension.h>` which includes what you need

### Error: `namespace "at::cuda" has no member "getCurrentCUDAStream"`
**Cause**: Missing header include
**Fix**: Add `#include <ATen/cuda/CUDAContext.h>` at the top of your CUDA file

### Error: `multiple definition of 'PyInit_*'`
**Cause**: PYBIND11_MODULE defined in both .cpp and .cu files, or both provided when using `functions` parameter
**Fix**: When using `functions` parameter in load_inline, do NOT define PYBIND11_MODULE yourself. Only provide:
- `cpp_sources`: Just the function declarations (e.g., `"torch::Tensor my_func(torch::Tensor x);"`)
- `cuda_sources`: The actual implementation

### Error: `undefined symbol` for CUDA functions
**Cause**: Function not declared in cpp_sources
**Fix**: Ensure every function in `functions` list has a declaration in `cpp_sources`

## Tensor Operations in CUDA

```cpp
// Get raw pointer
float* data = tensor.data_ptr<float>();

// Get dimensions
int64_t batch = tensor.size(0);
int64_t height = tensor.size(1);
int64_t width = tensor.size(2);

// Get total elements
int64_t numel = tensor.numel();

// Create output tensor
auto output = torch::empty_like(input);
auto output = torch::zeros({batch, height, width}, input.options());

// Check device
TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
```

## Kernel Launch Pattern

```cpp
torch::Tensor my_cuda_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");

    const int n = input.numel();
    auto output = torch::empty_like(input);

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    // Get current CUDA stream for async execution
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    my_kernel<<<blocks, threads, 0, stream>>>(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        n
    );

    return output;
}
```

## 2D/3D Grid Configuration

```cpp
// For 2D operations (e.g., matrix ops)
dim3 threads(16, 16);
dim3 blocks((width + 15) / 16, (height + 15) / 16);
kernel<<<blocks, threads>>>(output, input, height, width);

// For batched operations
dim3 threads(16, 16, 1);
dim3 blocks((width + 15) / 16, (height + 15) / 16, batch);
kernel<<<blocks, threads>>>(output, input, batch, height, width);
```

## Half Precision (FP16) Support

```cpp
#include <cuda_fp16.h>

// Check tensor dtype
if (input.scalar_type() == torch::kHalf) {
    half* data = reinterpret_cast<half*>(input.data_ptr<at::Half>());
}
```
