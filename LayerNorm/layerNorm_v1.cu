#include <cuda_runtime.h>

// 输入矩阵的大小为 (B, T, C) = (8, 1024, 768)  B 为 batchSize, T 为 sequence 的长度, C 是 embedding 的维度
constexpr int B = 8;
constexpr int T = 1024;
constexpr int C = 768;
constexpr int eps = 1e-5;
constexpr int blockSize = 128;

// 这个没什么心意, 就是注意这里可以进行 unroll
template <typename T1>
__device__ T1 warpReduceSum(T1 val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// 1. 利用了个公式 Var(X) = E(x^2) - E(X)^2, 这样就能节省一个循环
// 2. 每个 warp 处理一行
// 3. 因为 输入数据会被多次用到, 所以利用 shared_memory 来进行缓存输入数据
__global__ void layernorm_forward_kernel6(float *input, float *out, float *weight,
                                          float *bias, float eps, int B, int T, int C) {
    // one warp one row
    // use smem to store the shift (x - mean) values
    // use D(X) = E(X^2) - E(X)^2
    assert((C % warpSize) == 0);
    extern __shared__ float sharedX[];
    int warpsPerBlock = blockDim.x / warpSize;           // 每个 block 中的 warp 个数
    int warpId = threadIdx.x / warpSize;                 // 当前线程的 warp id
    int laneId = threadIdx.x % warpSize;                 // 当前线程的 lane id
    int numWarps = gridDim.x * warpsPerBlock;            // grid 的所有 warp 个数
    float *const xSharedWarp = sharedX + warpId * C;     // 当前 warp 用到的 SM 的起始地址

    for (int row = blockIdx.x * warpsPerBlock + warpId; row < B * T; row += numWarps)
        if (row < B * T) {
            // 定位到当前 warp 处理的全局内存的起始地址
            float *const x = input + row * C;
            float *const y = out + row * C;

            float partialSum = 0.0f, partialSum2 = 0.0f;
            for (int i = laneId; i < C; i += warpSize) {
                float xi = x[i];     // 先从共享内存读取到寄存器中, 这样再从寄存器到其他的寄存器和 SM 就快了
                xSharedWarp[i] = xi;
                partialSum += xi;
                partialSum += xi * xi;
            }

            float mean = warpReduceSum(partialSum) / C;    // warp 规约求 E(X)
            float mean2 = warpReduceSum(partialSum2) / C;  // warp 规约求 E(X^2)
            float var = (mean2 - mean * mean);             // 依据公式求得方差
            float inv_std = 1.0f / sqrt(var / C + eps);

            // 写到输出中
            for (int i = laneId; i < C; i += warpSize) {
                y[i] = weight[i] * (sharedX[i] - mean) * inv_std + bias[i];
            }
        }
}

int main(){

    float *inputGPU = nullptr;
    float *outputGPU = nullptr;
    float *weightGPU = nullptr;
    float *biasGPU = nullptr;
    cudaMalloc(&inputGPU, B * T * C * sizeof(float));
    cudaMalloc(&outputGPU, B * T * C * sizeof(float));
    // weight 和 bias 都是在 C 这个维度展开的
    cudaMalloc(&weightGPU, C * sizeof(float));
    cudaMalloc(&biasGPU, C * sizeof(float));

    const int smemSize = blockSize / 32 * C * sizeof(float);  // 每个 warp 处理一个 C, 那就需要 一个block中 warp 个数*C*sizeof(float)
    layernorm_forward_kernel6<<<B * T * 32 / blockSize, blockSize, smemSize>>>(inputGPU, outputGPU, weightGPU, biasGPU, eps, B, T, C); 

    return 0;
}