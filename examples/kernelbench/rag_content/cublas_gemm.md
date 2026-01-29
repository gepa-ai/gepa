# cuBLAS for Matrix Multiplication in PyTorch CUDA Extensions

## Key Insight
For matrix multiplication (GEMM), cuBLAS is almost always faster than hand-written CUDA kernels. cuBLAS is highly optimized by NVIDIA with:
- Tensor Core utilization (when available)
- Optimal memory access patterns
- Auto-tuned for different matrix sizes

## Using cuBLAS via ATen (Recommended)

The simplest way to use cuBLAS in a PyTorch CUDA extension is through ATen's `at::matmul` or `at::bmm`, which internally call cuBLAS:

```cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Simple matrix multiplication using cuBLAS via ATen
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    return at::matmul(A, B);  // Uses cuBLAS internally
}

// Batched matrix multiplication using cuBLAS via ATen
torch::Tensor bmm_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    return at::bmm(A, B);  // Uses cublasSgemmBatched internally
}
```

## Direct cuBLAS API (Advanced)

For more control, you can call cuBLAS directly:

```cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDABlas.h>

torch::Tensor matmul_cublas(torch::Tensor A, torch::Tensor B) {
    // A: [M, K], B: [K, N] -> C: [M, N]
    auto M = A.size(0);
    auto K = A.size(1);
    auto N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    float alpha = 1.0f;
    float beta = 0.0f;

    // Note: cuBLAS uses column-major, so we compute C^T = B^T * A^T
    // which gives us C in row-major (what PyTorch expects)
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,  // Note: swapped M and N for column-major
        &alpha,
        B.data_ptr<float>(), N,
        A.data_ptr<float>(), K,
        &beta,
        C.data_ptr<float>(), N);

    return C;
}
```

## Batched GEMM with cuBLAS

For batched matrix multiplication (torch.bmm):

```cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

torch::Tensor bmm_cublas(torch::Tensor A, torch::Tensor B) {
    // A: [batch, M, K], B: [batch, K, N] -> C: [batch, M, N]
    TORCH_CHECK(A.dim() == 3 && B.dim() == 3, "Inputs must be 3D");

    auto batch = A.size(0);
    auto M = A.size(1);
    auto K = A.size(2);
    auto N = B.size(2);

    auto C = torch::empty({batch, M, N}, A.options());

    // Use ATen's bmm which calls cublasSgemmStridedBatched
    return at::bmm(A, B);
}
```

## When to Write Custom Kernels vs Use cuBLAS

**Use cuBLAS (via at::matmul/at::bmm) when:**
- Standard GEMM operations (matmul, bmm)
- Large matrices (M, N, K > 128)
- Need maximum performance with minimal effort

**Write custom kernels when:**
- Fused operations (e.g., matmul + bias + relu in one kernel)
- Non-standard memory layouts
- Very small matrices where kernel launch overhead matters
- Specialized operations not in cuBLAS (e.g., sparse, quantized)

## Fused Operations Example

The main advantage of custom kernels is fusion. Here's matmul + bias + ReLU fused:

```cpp
__global__ void matmul_bias_relu_kernel(
    const float* A, const float* B, const float* bias,
    float* C, int M, int N, int K) {

    // Standard tiled matmul...
    float sum = 0.0f;
    // ... compute sum ...

    // Fused bias and ReLU
    sum += bias[col];
    C[row * N + col] = sum > 0 ? sum : 0;
}
```

## Performance Tips

1. **Contiguous tensors**: Ensure inputs are contiguous before cuBLAS
   ```cpp
   A = A.contiguous();
   B = B.contiguous();
   ```

2. **Same device**: Both tensors must be on same CUDA device
   ```cpp
   TORCH_CHECK(A.device() == B.device(), "Tensors must be on same device");
   ```

3. **Data type matching**: cuBLAS has different functions for float/half/double
   - `cublasSgemm` for float32
   - `cublasHgemm` for float16
   - `cublasDgemm` for float64

4. **Tensor Cores**: For V100/A100, use TF32 or FP16 for tensor core acceleration
   ```cpp
   // Enable TF32 for better performance on Ampere+
   at::globalContext().setAllowTF32CuBLAS(true);
   ```
