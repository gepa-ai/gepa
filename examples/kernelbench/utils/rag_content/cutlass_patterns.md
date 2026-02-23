# CUTLASS-Inspired CUDA Kernel Patterns

CUTLASS (CUDA Templates for Linear Algebra) shows optimal patterns for matrix operations. Here are key patterns you can use in hand-written CUDA kernels.

## Tiled Matrix Multiplication Pattern

The core CUTLASS pattern uses hierarchical tiling:
- Thread block tile (e.g., 128x128)
- Warp tile (e.g., 64x64)
- Thread tile (e.g., 8x8)

### Basic Tiled GEMM

```cpp
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_M 32
#define TILE_N 32
#define TILE_K 32

__global__ void tiled_gemm_kernel(
    const float* __restrict__ A,  // [M, K]
    const float* __restrict__ B,  // [K, N]
    float* __restrict__ C,        // [M, N]
    int M, int N, int K) {

    // Shared memory for tiles
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    float acc = 0.0f;

    // Loop over tiles along K dimension
    for (int t = 0; t < (K + TILE_K - 1) / TILE_K; t++) {
        // Collaborative loading into shared memory
        int k_idx = t * TILE_K + threadIdx.x;
        if (row < M && k_idx < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + k_idx];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        k_idx = t * TILE_K + threadIdx.y;
        if (k_idx < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[k_idx * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

torch::Tensor tiled_gemm(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    tiled_gemm_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, N, K);

    return C;
}
```

## Register Tiling (Thread-Level)

Each thread computes multiple output elements using registers:

```cpp
#define THREAD_TILE_M 4
#define THREAD_TILE_N 4

__global__ void register_tiled_gemm(
    const float* A, const float* B, float* C,
    int M, int N, int K) {

    // Each thread computes a 4x4 tile
    float acc[THREAD_TILE_M][THREAD_TILE_N] = {0};
    float a_reg[THREAD_TILE_M];
    float b_reg[THREAD_TILE_N];

    int row_base = blockIdx.y * blockDim.y * THREAD_TILE_M + threadIdx.y * THREAD_TILE_M;
    int col_base = blockIdx.x * blockDim.x * THREAD_TILE_N + threadIdx.x * THREAD_TILE_N;

    for (int k = 0; k < K; k++) {
        // Load A elements into registers
        #pragma unroll
        for (int i = 0; i < THREAD_TILE_M; i++) {
            int row = row_base + i;
            a_reg[i] = (row < M) ? A[row * K + k] : 0.0f;
        }

        // Load B elements into registers
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; j++) {
            int col = col_base + j;
            b_reg[j] = (col < N) ? B[k * N + col] : 0.0f;
        }

        // Outer product
        #pragma unroll
        for (int i = 0; i < THREAD_TILE_M; i++) {
            #pragma unroll
            for (int j = 0; j < THREAD_TILE_N; j++) {
                acc[i][j] += a_reg[i] * b_reg[j];
            }
        }
    }

    // Write results
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; j++) {
            int row = row_base + i;
            int col = col_base + j;
            if (row < M && col < N) {
                C[row * N + col] = acc[i][j];
            }
        }
    }
}
```

## Large K Dimension Handling

For matrices with very large K (e.g., K > 100000), use split-K parallelism:

```cpp
// Split K across multiple blocks, then reduce
__global__ void split_k_gemm_kernel(
    const float* A, const float* B, float* C_partial,
    int M, int N, int K, int split_k, int k_per_split) {

    int split_idx = blockIdx.z;
    int k_start = split_idx * k_per_split;
    int k_end = min(k_start + k_per_split, K);

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = k_start; k < k_end; k++) {
        acc += A[row * K + k] * B[k * N + col];
    }

    // Write partial result
    C_partial[split_idx * M * N + row * N + col] = acc;
}

__global__ void reduce_split_k(
    const float* C_partial, float* C,
    int M, int N, int split_k) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int s = 0; s < split_k; s++) {
        sum += C_partial[s * M * N + row * N + col];
    }
    C[row * N + col] = sum;
}
```

## Batched GEMM Pattern

For batched matrix multiplication:

```cpp
__global__ void batched_gemm_kernel(
    const float* A,  // [B, M, K]
    const float* B,  // [B, K, N]
    float* C,        // [B, M, N]
    int batch, int M, int N, int K) {

    int b = blockIdx.z;  // Batch index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch || row >= M || col >= N) return;

    // Offset pointers for this batch
    const float* A_b = A + b * M * K;
    const float* B_b = B + b * K * N;
    float* C_b = C + b * M * N;

    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        acc += A_b[row * K + k] * B_b[k * N + col];
    }
    C_b[row * N + col] = acc;
}

torch::Tensor batched_gemm(torch::Tensor A, torch::Tensor B) {
    int batch = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    auto C = torch::zeros({batch, M, N}, A.options());

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16, batch);

    batched_gemm_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        batch, M, N, K);

    return C;
}
```

## Memory Coalescing Tips

1. **Coalesced reads**: Threads in a warp should access consecutive memory
   ```cpp
   // Good: consecutive threads read consecutive elements
   float val = A[row * K + threadIdx.x];

   // Bad: strided access
   float val = A[threadIdx.x * K + col];
   ```

2. **Bank conflict avoidance**: Pad shared memory
   ```cpp
   __shared__ float As[TILE][TILE + 1];  // +1 padding to avoid bank conflicts
   ```

3. **Vectorized loads**: Use float4 for 128-bit loads
   ```cpp
   float4 a4 = *reinterpret_cast<const float4*>(&A[idx]);
   ```

## When Custom Kernels Beat cuBLAS

1. **Fused operations**: matmul + activation in one kernel
2. **Custom reductions**: Non-standard aggregations
3. **Specialized shapes**: Very tall/skinny matrices
4. **Memory-bound cases**: When fusion reduces memory traffic
