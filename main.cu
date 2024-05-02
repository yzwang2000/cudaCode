#include <stdio.h>

__global__ void kernel_A(double* A, int N, int M)
{
    double d = 0.0;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // printf("Kernel A\n");

    if (idx < N) {

#pragma unroll(100)
        for (int j = 0; j < M; ++j) {
            d += A[idx];
        }

        A[idx] = d;

    }
}

__global__ void kernel_B(double* A, int N, int M)
{
    double d = 0.0;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N) {

#pragma unroll(100)
        for (int j = 0; j < M; ++j) {
            d += A[idx];
        }

        A[idx] = d;

    }
}

__global__ void kernel_C(double* A, const double* B, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // printf("Kernel C\n");

    // Strided memory access: warp 0 accesses (0, stride, 2*stride, ...), warp 1 accesses
    // (1, stride + 1, 2*stride + 1, ...).
    const int stride = 16;
    int strided_idx = threadIdx.x * stride + blockIdx.x % stride + (blockIdx.x / stride) * stride * blockDim.x;

    if (strided_idx < N) {
        A[idx] = B[strided_idx] + B[strided_idx];
    }
}

int main() {

    double* A;
    double* B;

    int N = 80 * 2048 * 100;
    size_t sz = N * sizeof(double);

    cudaMalloc((void**) &A, sz);
    cudaMalloc((void**) &B, sz);

    cudaMemset(A, 0, sz);
    cudaMemset(B, 0, sz);

    int threadsPerBlock = 64;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    int M = 10000;
    kernel_A<<<numBlocks, threadsPerBlock>>>(A, N, M);

    // cudaFuncSetAttribute(kernel_B, cudaFuncAttributeMaxDynamicSharedMemorySize, 48 * 1024);
    // kernel_B<<<numBlocks, threadsPerBlock, 48 * 1024>>>(A, N, M);
    kernel_C<<<numBlocks, threadsPerBlock>>>(A, B, N);

    cudaDeviceSynchronize();

}