#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// batchSize 为 2, seq_len 为 64, num_head 为 32, head_dim 为 128
#define NUM_PAIRS 64  // head_dim / 2
#define SEQ_LEN 64

// Precomputed θ_i values stored in constant memory, 存储角度值
__constant__ float theta[NUM_PAIRS];

// Precomputed cosine and sine values for θ_i * p, 存储 m*theta 的 sin 和 cos 值, 形状是 [seq_len, num_pairs], 先存 seq_len 的所有 num_pair
__constant__ float cos_theta_p[NUM_PAIRS * SEQ_LEN];
__constant__ float sin_theta_p[NUM_PAIRS * SEQ_LEN];

__global__ void RoPE(float *x, int batchSize, int seq_len, int num_head, int head_dim) {
    // Each block processes one (batch_id, seq_id, head_id)
    int batch_id = blockIdx.z;
    int seq_id = blockIdx.y;
    int head_id = blockIdx.x;
    int tid = threadIdx.x;

    int num_pairs = head_dim / 2;
    int total_threads = blockDim.x;    // 一个 block 中的总的线程个数

    int p = seq_id;
    int theta_offset = p * num_pairs;  // 找 theta 的偏移

    for (int i = tid; i < num_pairs; i += total_threads) {
        int idx = 2 * i;

        // Retrieve precomputed cosine and sine values
        float cos_theta = cos_theta_p[theta_offset + i];
        float sin_theta = sin_theta_p[theta_offset + i];

        // Compute offset
        int offset = batch_id * seq_len * num_head * head_dim 
                        + seq_id * num_head * head_dim + 
                            head_id * head_dim + idx;

        float x1 = x[offset];
        float x2 = x[offset + 1];

        // Apply rotation
        float x1_new = x1 * cos_theta - x2 * sin_theta;
        float x2_new = x1 * sin_theta + x2 * cos_theta;

        // Write back
        x[offset] = x1_new;
        x[offset + 1] = x2_new;
    }
}

int main() {
    int batchSize = 2;
    int seq_len = SEQ_LEN;
    int num_head = 32;
    int head_dim = 128;

    // 总元素个数
    int total_elements = batchSize * seq_len * num_head * head_dim;
    size_t size = total_elements * sizeof(float);

    // Allocate host memory
    float *h_x = (float *)malloc(size);
    // Initialize h_x with some values
    for (int i = 0; i < total_elements; i++) {
        h_x[i] = 1.0f;  // or any value you prefer
    }

    // Precompute θ_i values
    float h_theta[NUM_PAIRS];
    for (int i = 0; i < NUM_PAIRS; i++) {  // 注意这里, 直接将幂次变成负数, 相当于取了倒数
        h_theta[i] = powf(10000.0f, -2.0f * i / head_dim);
    }
    cudaMemcpyToSymbol(theta, h_theta, NUM_PAIRS * sizeof(float));

    // Precompute cosine and sine values for θ_i * p
    float h_cos_theta_p[NUM_PAIRS * SEQ_LEN];
    float h_sin_theta_p[NUM_PAIRS * SEQ_LEN];

    // 先是 seq_len, 再是 h_theta
    for (int p = 0; p < seq_len; p++) {
        for (int i = 0; i < NUM_PAIRS; i++) {
            int idx = p * NUM_PAIRS + i;
            float theta_p = h_theta[i] * p;
            h_cos_theta_p[idx] = cosf(theta_p);
            h_sin_theta_p[idx] = sinf(theta_p);
        }
    }
    cudaMemcpyToSymbol(cos_theta_p, h_cos_theta_p, NUM_PAIRS * SEQ_LEN * sizeof(float));
    cudaMemcpyToSymbol(sin_theta_p, h_sin_theta_p, NUM_PAIRS * SEQ_LEN * sizeof(float));

    // Allocate device memory
    float *d_x;
    cudaMalloc((void **)&d_x, size);

    // Copy data from host to device, 输入与输出大小相等
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 grid {num_head, seq_len, batchSize};  // 2 * 64 * 32 = 4096
    dim3 block {32};  // One warp per block, 一个 block 中包含一个 warp

    RoPE<<<grid, block>>>(d_x, batchSize, seq_len, num_head, head_dim);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_x, d_x, size, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_x);
    free(h_x);

    return 0;
}
