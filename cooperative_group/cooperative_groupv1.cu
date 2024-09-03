#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <iostream>
#include <algorithm>
#include <memory>

using namespace cooperative_groups;

template <int tile_sz>
__device__ int reduce_sum_tile_shfl(thread_block_tile<tile_sz> g, int val) {
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    for (int i = g.size() / 2; i > 0; i /= 2) {
        val += g.shfl_down(val, i);
    }

    return val; // Note: only thread 0 will return full sum
}

__device__ int thread_sum(int* input, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int sum = 0;
    for (int i = tid; i < n; i += stride) {
        sum += input[i];
    }
    return sum;
}

template<int tile_sz>
__global__ void sum_kernel_tile_shfl(int *sum, int *input, int n) {
    int my_sum = thread_sum(input, n);

    auto tile = tiled_partition<tile_sz>(this_thread_block());
    int tile_sum = reduce_sum_tile_shfl<tile_sz>(tile, my_sum);

    if (tile.thread_rank() == 0) atomicAdd(sum, tile_sum);
}

int main() {
    // Step 1: Prepare data
    const int N = 1024 * 1024; // Size of input vector
    int* h_input = new int[N]; // Host input
    int h_sum = 0; // Host output

    // Initialize the input array with some values
    std::fill_n(h_input, N, 1); // Fill the array with 1s for simplicity

    // Step 2: Allocate device memory
    int *d_input, *d_sum;
    cudaMalloc((void**)&d_input, N * sizeof(int));
    cudaMalloc((void**)&d_sum, sizeof(int));
    
    // Copy input data from host to device
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_sum, 0, sizeof(int)); // Initialize sum to 0 on device

    // Step 3: Configure and launch the kernel
    const int blockSize = 256; // Threads per block
    const int tileSize = 32;   // Cooperative group tile size
    const int numBlocks = (N + blockSize - 1) / blockSize; // Number of blocks

    sum_kernel_tile_shfl<tileSize><<<numBlocks, blockSize>>>(d_sum, d_input, N);

    // Step 4: Copy result from device to host
    cudaMemcpy(&h_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "Sum of array elements: " << h_sum << std::endl;

    // Step 5: Free memory
    cudaFree(d_input);
    cudaFree(d_sum);
    delete[] h_input;

    return 0;
}
