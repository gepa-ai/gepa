import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Inline CUDA code for custom max_pool2d kernel
source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void max_pool2d_kernel(float* input, float* output, int channels, int input_height, int input_width, int pool_height, int pool_width, int stride, int output_height, int output_width) {
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    if (w_out < output_width && h_out < output_height) {
        float max_val = -FLT_MAX;
        int start_h = h_out * stride;
        int start_w = w_out * stride;

        for (int h = 0; h < pool_height; ++h) {
            for (int w = 0; w < pool_width; ++w) {
                int h_in = start_h + h;
                int w_in = start_w + w;
                if (h_in < input_height && w_in < input_width) {
                    max_val = fmaxf(max_val, input[c * input_height * input_width + h_in * input_width + w_in]);
                }
            }
        }
        output[c * output_height * output_width + h_out * output_width + w_out] = max_val;
    }
}

torch::Tensor max_pool2d_cuda(torch::Tensor input, int kernel_size, int stride) {
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    const int output_height = (input_height - kernel_size) / stride + 1;
    const int output_width = (input_width - kernel_size) / stride + 1;

    auto output = torch::empty({input.size(0), channels, output_height, output_width}, input.options());

    dim3 block_size(16, 16);
    dim3 num_blocks((output_width + block_size.x - 1) / block_size.x, (output_height + block_size.y - 1) / block_size.y, channels);

    max_pool2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        channels, 
        input_height, 
        input_width, 
        kernel_size, 
        kernel_size, 
        stride, 
        output_height, 
        output_width
    );

    return output;
}
"""
cpp_src = "torch::Tensor max_pool2d_cuda(torch::Tensor input, int kernel_size, int stride);"

# Compile the inline CUDA code
custom_max_pool = load_inline(
    name='custom_max_pool',
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=['max_pool2d_cuda'],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Custom MNIST model using inlined max_pool2d_cuda
class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.custom_max_pool = custom_max_pool

    def forward(self, x):
        # Use the custom max_pool2d operator
        x = self.custom_max_pool.max_pool2d_cuda(F.relu(self.conv1(x)), 2, 2)
        x = self.custom_max_pool.max_pool2d_cuda(F.relu(self.conv2(x)), 2, 2)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
