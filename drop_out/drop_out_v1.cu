#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>

#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

// Dropout kernel function
__global__ void DropoutKernel(const float* __restrict__ x,
                              uint8_t* __restrict__ mask,
                              float* __restrict__ y,
                              size_t n,
                              float dropout_prob,
                              bool is_upscale_in_train,
                              unsigned int seed) {

    // Calculate global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 计算线程的 id
    int stride = gridDim.x * blockDim.x;              // 所有线程的总数

    // Initialize the random number generator state
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);

    float inv_prob = 1.0f / (1.0f - dropout_prob);  // 提前计算出来

    // Loop over the elements, processing four elements per thread for efficiency
    for (size_t i = idx * 4; i < n; i += stride * 4) {
        // Generate 4 random numbers
        float4 rand = curand_uniform4(&state);  // 一次得到 float4 的随机数

        // Load 4 input values, 因为输入不一定是 4 的整数倍, 所以可以直接展开, 超的直接置为 0 就可以了
        float4 x_val;
        x_val.x = (i + 0 < n) ? x[i + 0] : 0.0f;
        x_val.y = (i + 1 < n) ? x[i + 1] : 0.0f;
        x_val.z = (i + 2 < n) ? x[i + 2] : 0.0f;
        x_val.w = (i + 3 < n) ? x[i + 3] : 0.0f;

        // Initialize output values and masks, 保存输出值和mask
        float4 y_val;
        uint8_t m_val[4];

        // Apply dropout, 小于 dropout 概率, 都置为 0, 还需要依据 is_upscale_in_train 来判断是否需要缩放
        y_val.x = (rand.x >= dropout_prob) ? (is_upscale_in_train ? x_val.x * inv_prob : x_val.x) : 0.0f;
        y_val.y = (rand.y >= dropout_prob) ? (is_upscale_in_train ? x_val.y * inv_prob : x_val.y) : 0.0f;
        y_val.z = (rand.z >= dropout_prob) ? (is_upscale_in_train ? x_val.z * inv_prob : x_val.z) : 0.0f;
        y_val.w = (rand.w >= dropout_prob) ? (is_upscale_in_train ? x_val.w * inv_prob : x_val.w) : 0.0f;

        // Set mask values, 存储 mask
        m_val[0] = (rand.x >= dropout_prob);
        m_val[1] = (rand.y >= dropout_prob);
        m_val[2] = (rand.z >= dropout_prob);
        m_val[3] = (rand.w >= dropout_prob);

        // Store the results back to global memory, 展开着写回, 防止超越边界
        if (i + 0 < n) {
            y[i + 0] = y_val.x;
            mask[i + 0] = m_val[0];
        }
        if (i + 1 < n) {
            y[i + 1] = y_val.y;
            mask[i + 1] = m_val[1];
        }
        if (i + 2 < n) {
            y[i + 2] = y_val.z;
            mask[i + 2] = m_val[2];
        }
        if (i + 3 < n) {
            y[i + 3] = y_val.w;
            mask[i + 3] = m_val[3];
        }
    }
}

int main() {
    const size_t num_eles = 2050;  // Number of elements

    // Allocate host memory
    float* x = (float*)malloc(num_eles * sizeof(float));  // 输入
    float* y = (float*)malloc(num_eles * sizeof(float));  // 输出
    uint8_t* mask = (uint8_t*)malloc(num_eles * sizeof(uint8_t));  // mask 掩码矩阵, 当 mask 为 1 的时候保留, 为 0 就置为 0

    // Initialize input data
    for (size_t i = 0; i < num_eles; ++i) {
        x[i] = 1.0f;
    }

    // Allocate device memory
    float* d_x;
    float* d_y;
    uint8_t* d_mask;
    CHECK(cudaMalloc((void**)&d_x, num_eles * sizeof(float)));
    CHECK(cudaMalloc((void**)&d_y, num_eles * sizeof(float)));
    CHECK(cudaMalloc((void**)&d_mask, num_eles * sizeof(uint8_t)));

    // Copy input data to device
    CHECK(cudaMemcpy(d_x, x, num_eles * sizeof(float), cudaMemcpyHostToDevice));

    // Dropout parameters
    const bool is_test = false;             // 是否是推理阶段
    const bool is_upscale_in_train = true;  // 训练阶段是否是进行 up scale
    const float dropout_prob = 0.5f;        // 以多少的概率进行 drop
    const unsigned int seed = 10000;        // 随机数种子

    if (is_test) {  // 推理阶段, 直接进行 copy, 没有多余的动作
        // In inference mode, output is the same as input
        CHECK(cudaMemcpy(d_y, d_x, num_eles * sizeof(float), cudaMemcpyDeviceToDevice));
    } else {  // 训练阶段
        if (dropout_prob == 1.0f) {  // drop 概率为 1, 就全置为 0, mask 也全置为 0
            // If dropout probability is 1, set output and mask to zero
            CHECK(cudaMemset(d_y, 0, num_eles * sizeof(float)));
            CHECK(cudaMemset(d_mask, 0, num_eles * sizeof(uint8_t)));
        } else {  // drop 概率小于 1
            // Launch the dropout kernel, 一个 block 设置有 256 个线程
            int threads_per_block = 256;
            // 每个线程处理四个元素, 向上取整
            int num_blocks = (num_eles + threads_per_block * 4 - 1) / (threads_per_block * 4);
            DropoutKernel<<<num_blocks, threads_per_block>>>(d_x, d_mask, d_y,
                                                             num_eles, dropout_prob,
                                                             is_upscale_in_train, seed);
            CHECK(cudaGetLastError());
        }
    }

    // Copy results back to host
    CHECK(cudaMemcpy(y, d_y, num_eles * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(mask, d_mask, num_eles * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    // Free memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_mask);
    free(x);
    free(y);
    free(mask);

    return 0;
}
