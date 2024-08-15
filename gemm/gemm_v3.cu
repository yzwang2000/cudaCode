#include <iostream>
#include <cuda.h>

// A is M*K, B is K*N, C is M*N
constexpr int M = 1024;
constexpr int K = 256;
constexpr int N = 1024;

// split C into many blocks, size of each block
constexpr int block_m = 128;
constexpr int block_n = 128;
constexpr int block_k = 8;
constexpr int maxIter = K/block_k;
constexpr int num_block_each_grid = M/block_m * N/block_n;

// split each block into many threads, size of each small block
constexpr int thread_m = 8;
constexpr int thread_n = 8;
constexpr int thread_k = 1;
constexpr int minIter = block_k/thread_k;
constexpr int num_thread_each_block = block_m/thread_m * block_n/thread_n;  // 256

__global__ void gemm(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C){
    // Calculate the global thread indices
    int row = blockIdx.y * block_m + threadIdx.y * thread_m;
    int col = blockIdx.x * block_n + threadIdx.x * thread_n;

    // Declare shared memory
    __shared__ float smA[block_k][block_m];
    __shared__ float smB[block_k][block_n];

    // Registers for each thread
    float regA[thread_m];
    float regB[thread_n];

    // Initialize accumulation array to zero
    float accum[thread_m][thread_n] = {0.f};

    // Outer loop over the K dimension, iterating through blocks of size block_k
    #pragma unroll
    for(int i = 0; i < maxIter; i++){
        // Load A and B from global memory to shared memory
        for(int j = 0; j < block_k; j++){
            smA[j][threadIdx.y * thread_m + threadIdx.x] = A[(row + threadIdx.y * thread_m) * K + i * block_k + j];
            smB[j][threadIdx.y * thread_n + threadIdx.x] = B[(i * block_k + j) * N + col + threadIdx.x * thread_n];
        }
        __syncthreads();

        // Inner loop to perform the computation for the current block
        #pragma unroll
        for(int j = 0; j < minIter; j++) {
            // Load from shared memory into registers
            for(int k = 0; k < thread_m; k++) {
                regA[k] = smA[j * thread_k][threadIdx.y * thread_m + k];
            }
            for(int g = 0; g < thread_n; g++) {
                regB[g] = smB[j * thread_k][threadIdx.x * thread_n + g];
            }
            // Perform the matrix multiplication and accumulate the results
            for(int k = 0; k < thread_m; ++k) {
                for(int g = 0; g < thread_n; ++g) {
                    accum[k][g] += regA[k] * regB[g];
                }
            }
        }
        __syncthreads();
    }

    // Write the accumulated results back to global memory
    for(int k = 0; k < thread_m; ++k) {
        for(int g = 0; g < thread_n; ++g) {
            C[(row + k) * N + col + g] = accum[k][g];
        }
    }
}

int main(){
    // Allocate memory on the host
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C = (float*)malloc(M * N * sizeof(float));

    // Initialize matrices A and B with random values
    for(int i = 0; i < M * K; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for(int i = 0; i < K * N; i++) {
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate memory on the device
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 grid(N/block_n, M/block_m, 1);
    dim3 block(block_n/thread_n, block_m/thread_m);

    // Launch the GEMM kernel
    gemm<<<grid, block>>>(d_A, d_B, d_C);

    // Copy the result matrix C from device to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
