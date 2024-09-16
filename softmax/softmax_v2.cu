#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "common.h"

#define M 8196            // M 是行
#define N 8196            // N 是列
#define BLOCK_SIZE 128    // 每个块的线程个数
#define REPEAT_TIMES 100  // 测试性能时, 重复的循环次数

/* Softmax forward implementation

Usage: ./softmax_forward <kernel> [blockSize]
e.g. ./softmax_forward 1

softmax_forward_cpu(): CPU implementation

softmax_forward_kernel1(): Naive implementation on CUDA. Each thread handles
one row of the input.

softmax_forward_kernel2(): Optimized implementation on CUDA. Compares to
kernel1, each warp (32 threads) handles one row.

online_softmax_forward_kernel3(): Online softmax forward implementation on CUDA.
Also each warp handles one row of the input.

softmax_forward_kernel4(): Online softmax forward implementation on CUDA.
Using block reduce.

softmax_forward_kernel5(): Online softmax forward implementation on CUDA.
Each warp handles one row of the input.
Use float4 to acclerate memory access.
*/

// CPU 版本的实现
void softmax_cpu(float* input, float* output, const int M, const int N) {
    // 一行一行的处理
    for (int m = 0; m < M; ++m) {
        // 先获取这一行的最大值
        float maxval = -INFINITY;
        const float* x = input + m * N;
        for (int n = 0; n < N; ++n) {
            maxval = maxval > x[n] ? maxval : x[n];
        }

        // 依据最大值获取这行的指数和
        float s = 0.0f;
        for (int n = 0; n < N; ++n) {
            s += exp(x[n] - maxval);
        }

        // 依据这行的最大值和指数和得到输出
        float* y = output + m * N;
        for (int n = 0; n < N; ++n) {
            y[n] = exp(x[n] - maxval) / s;
        }
    }
}

// online 的 softmax cpu 版本
void online_softmax_cpu(float* input, float* output, const int M, const int N) {
    // 一行一行的处理
    for (int m = 0; m < M; ++m) {
        const float* x = input + m * N;
        // 两个循环合并成一个(合并求最大值和求指数和)
        float maxval = -INFINITY;
        float s = 0.0f;
        for (int n = 0; n < N; ++n) {
            // 维护全局最大值和当前所遍历元素的指数和, 当得到新的最大值的时候, 更新全局最大值和当前所遍历元素的指数和
            if (maxval < x[n]) {
                s *= exp(maxval - x[n]);  // 初始时为 exp(-INFINITY), 值为0, 更新指数和
                maxval = x[n];            // 更新最大值, 注意这两行不能调换
            }
            s += exp(x[n] - maxval);
        }

        float* y = output + m * N;
        for (int n = 0; n < N; ++n) {
            y[n] = exp(x[n] - maxval) / s;
        }
    }
}

__global__ void softmax_kernel1(float* input, float* output, const int M, const int N) {
    // naive implementation
    // one thread one row
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int idx = bid * blockDim.x + tid;
    if (idx < M) {
        float maxval = -INFINITY;
        const float* x = input + idx * N;
        for (int n = 0; n < N; ++n) {
            maxval = maxval > x[n] ? maxval : x[n];
        }
        float s = 0.0f;
        for (int n = 0; n < N; ++n) {
            s += exp(x[n] - maxval);
        }
        float* const y = output + idx * N;
        for (int n = 0; n < N; ++n) {
            y[n] = exp(x[n] - maxval) / s;
        }
    }
}

// blockSize = 128, 每个 warp 处理一个行。
// dim3 grid((M*32 + blockSize-1)/blockSize);  dim3 block(blockSize);
__global__ void softmax_kernel2(float* input, float* output, const int M, const int N) {
    // use more threads per row than kernel1
    // one warp (32 threads) process one row
    // use warp reduce functions
    const int tid = threadIdx.x;
    const int warpId = tid / warpSize;                 // 块中第几个 warp, 其实就是以 warp 为单位了
    const int laneId = tid % warpSize;
    const int warpsPerBlock = blockDim.x / warpSize;   // 一个 block 中的 warp 个数
    const int numWarps = warpsPerBlock * gridDim.x;    // 总的 warp 个数
    const int idx = warpsPerBlock * blockIdx.x + warpId;  // warp 的id
    for (int m = idx; m < M; m += numWarps) {  // 这里并不跨步, 而是把超过边界的全部排除掉
        // each lane (thread in a warp) calculate the maxval among
        // data with indices [landId, landId + 32, laneId + 64, ...]
        const float* x = input + m * N;   // 输入数组中的值不可修改
        float* const y = output + m * N;  // 不能修改指向的地址

        // 先将值归约到每个线程的寄存器中
        float maxval = -INFINITY;
        for (int i = laneId; i < N; i += warpSize) {
            maxval = fmaxf(maxval, x[i]);
        }
        // warpReduce 得到结果
        // warp-reduce to calculate the MAX of maxval among all lanes
        // and the 0-th lane will store the result
        maxval = warpReduceMax(maxval);

        // 将值归约到每个线程的寄存器中
        float sum = 0.0f;
        for (int i = laneId; i < N; i += warpSize) {
            sum += expf(x[i] - maxval);
        }
        // warpReduce 得到结果
        sum = warpReduceSum(sum);

        // 逐个元素遍历, 得到每个元素的输出结果
        for (int i = laneId; i < N; i += warpSize) {
            y[i] = expf(x[i] - maxval) / sum;
        }
    }
}

// blockSize = 128, 每个 warp 处理一个行。
// dim3 grid((M*32 + blockSize-1)/blockSize);  dim3 block(blockSize);
__global__ void online_softmax_kernel3(float* input, float* output, const int M, const int N) {
    const int tid = threadIdx.x;                        // 块中的线程索引 id
    const int warpId = tid / warpSize;                  // 此线程位于块内的第几个 warp
    const int laneId = tid % warpSize;                  // 此线程位于一个 warp 中的索引
    const int warpsPerBlock = blockDim.x / warpSize;    // 一个 block 的 warp 个数
    const int numWarps = warpsPerBlock * gridDim.x;
    const int idx = warpsPerBlock * blockIdx.x + warpId;
    for (int m = idx; m < M; m += numWarps) {
        const float* x = input + m * N;
        float* const y = output + m * N;
        float maxval = -INFINITY, sum = 0.0f, bigger;  // bigger 其实是为了记录新的最大值

        // 仍然是先将值汇总到每个线程寄存器中。经过这里的步骤之后, 每个线程中保留了其负责的那部分最大的值和指数和。
        for (int i = laneId; i < N; i += warpSize) {
            bigger = fmaxf(maxval, x[i]);  // 得到更大的值
            sum = sum * expf(maxval - bigger) + expf(x[i] - bigger);  // 这里就没有判断了, 直接做就可以啦
            maxval = bigger;
        }

        // warp 内的规约, 要规约最大值和指数和
        float offsetMax, offsetSum;  // 每个线程都维护全局的最大值和指数和
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            __syncwarp();
            offsetMax = __shfl_xor_sync(0xFFFFFFFF, maxval, offset);  // 得到其他局部块的最大值
            offsetSum = __shfl_xor_sync(0xFFFFFFFF, sum, offset);     // 得到其他局部块的指数和
            if (offsetMax > maxval) {  // 另一个块比本块大
                sum *= expf(maxval - offsetMax);  // 将自己的 sum 乘以 exp
                maxval = offsetMax;               // 将自己的 maxval 更新
            } else {  // 另一个块比本快小
                offsetSum *= expf(offsetMax - maxval);  // 更新另一个块的
            }
            sum += offsetSum;  // 最后肯定是 更新自己块+其他块   或    自己块+更新其他块
        }

        for (int i = laneId; i < N; i += warpSize) {
            y[i] = expf(x[i] - maxval) / sum;
        }
    }
}

// 每个 block 处理一行, dim3 grid(M);  dim3 block(blockSize);  shared_memory 每个 block 中分配 2*warp 个数的信 float 的共享内存
__global__ void online_softmax_kernel4(float* input, float* output, const int M, const int N) {
    // further, each block handles one row
    if (blockIdx.x < M) {
        extern __shared__ float shared[];
        const int laneId = threadIdx.x % warpSize;
        const int warpId = threadIdx.x / warpSize;
        const int warpsPerBlock = ceilDiv(blockDim.x, warpSize);  // 每个 block 中 warp 的个数
        const int dataPerWarp = ceilDiv(N, warpsPerBlock);  // 每个 warp 负责的数据总个数
        const int start = dataPerWarp * warpId;             // 每个 warp 负责的数据开始的位置
        const int end = min((warpId + 1) * dataPerWarp, N); // 每个 warp 负责的数据结束的位置
        const float* x = input + blockIdx.x * N;
        float* const y = output + blockIdx.x * N;

        float* const maxVals = shared;                  // 先存放每个 warp 的最大值
        float* const sumVals = shared + warpsPerBlock;  // 再存放每个 warp 的指数和

        // each lane calculates its own max and sum
        float maxval = -INFINITY, sumval = 0.f;
        for (int i = start + laneId; i < end; i += warpSize) {
            float newLaneMax = fmaxf(maxval, x[i]);  // 每检索一个 x, 就求一次最大值
            sumval = sumval * expf(maxval - newLaneMax) + expf(x[i] - newLaneMax);  // 依据最大值进行更新
            maxval = newLaneMax;
        }

        // have 32 sub-max and sub-sum values hold by each lane in the same warp
        // warp reduce, get the warpwise max and sum
        // 每个 warp 内进行规约, 得到新的最大值
        float maxvalOld = maxval;
        maxval = warpReduceMax(maxval); // now maxval is warpwise maxval
        sumval *= expf(maxvalOld - maxval);  // 更新每个线程中包含的前缀和
        sumval = warpReduceSum(sumval); // now sumval is warpwise sumval

        // 进行 warp 间的分块计算
        // store each warp's max and sum to smem
        if (laneId == 0) {
            maxVals[warpId] = maxval;
            sumVals[warpId] = sumval;
        }
        __syncthreads();
        // re-init for the next block reduce process

        // let warp 0 do the block reduce
        if (warpId == 0) {
            // 给初始化的值
            if (laneId < warpsPerBlock) {
                maxval = maxVals[laneId];
                sumval = sumVals[laneId];
            } else {
                maxval = -INFINITY;
                sumval = 0.f;
            }
            for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
                float otherMax = __shfl_xor_sync(0xFFFFFFFF, maxval, offset);
                float otherSum = __shfl_xor_sync(0xFFFFFFFF, sumval, offset);
                if (maxval < otherMax) {
                    sumval = sumval * expf(maxval - otherMax) + otherSum;
                    maxval = otherMax;
                } else {
                    sumval += otherSum * expf(otherMax - maxval);
                }
            }
        }
        __syncthreads();

        // 利用共享内存做为中间介质
        // update the blockwise sum and max to smem
        if (warpId == 0 && laneId == 0) {
            maxVals[0] = maxval;
            sumVals[0] = sumval;
        }
        __syncthreads();

        // 将值传递给每个线程
        // warps take the blockwise max and sum and do the final calculation
        float blockSum = sumVals[0], blockMax = maxVals[0];
        for (int i = start + laneId; i < end; i += 32) {
            y[i] = expf(x[i] - blockMax) / blockSum;
        }
    }
}

__global__ void online_softmax_kernel5(float* __restrict__ input, float* __restrict__ output, const int M,
                                       const int N) {
    // this kernel is f*cking faster than any other kernels!
    // use float4 to acclerate memory access
    // each warp (32 threads) handles one row
    // TODO: fix bug of misaligned data memory access
    using f128 = Package128<float>;
    const int tid = threadIdx.x;
    const int warpId = tid / warpSize;
    const int laneId = tid % warpSize;
    const int warpsPerBlock = blockDim.x / warpSize;
    int row = warpsPerBlock * blockIdx.x + warpId;
    if (row < M) {
        float* x = input + row * N;
        float* y = output + row * N;
        float laneMax = -INFINITY, laneSum = 0.0f;
        int i = ceilDiv(N, f128::size) + laneId - warpSize;
        while ((i + 1) * f128::size >= N) {
            for (int k = 0; k < f128::size; ++k) {
                if (i * f128::size + k >= N) {
                    break;
                }
                float newLaneMax = fmaxf(laneMax, x[i * f128::size + k]);
                laneSum = laneSum * expf(laneMax - newLaneMax) + expf(x[i * f128::size + k] - newLaneMax);
                laneMax = newLaneMax;
            }
            i -= warpSize;
        }

        for (; i >= 0; i -= warpSize) {
            f128 xi = load128cs(x + i * f128::size);
            float packMax = -INFINITY, packSum = 0.0f;
#pragma unroll
            for (int k = 0; k < f128::size; ++k) {
                float newPackMax = fmaxf(packMax, xi[k]);
                packSum = expf(packMax - newPackMax) * packSum + expf(xi[k] - newPackMax);
                packMax = newPackMax;
            }
            float newLaneMax = fmaxf(laneMax, packMax);
            laneSum = laneSum * expf(laneMax - newLaneMax) + packSum * expf(packMax - newLaneMax);
            laneMax = newLaneMax;
        }

        float maxVal = laneMax, sumVal = laneSum;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            float offsetMax = __shfl_xor_sync(0xFFFFFFFF, maxVal, offset);
            float offsetSum = __shfl_xor_sync(0xFFFFFFFF, sumVal, offset);
            if (maxVal > offsetMax) {
                sumVal += expf(offsetMax - maxVal) * offsetSum;
            } else {
                sumVal = sumVal * expf(maxVal - offsetMax) + offsetSum;
                maxVal = offsetMax;
            }
        }

        i = ceilDiv(N, f128::size) + laneId - warpSize;
        while ((i + 1) * f128::size >= N) {
            for (int k = 0; k < f128::size; ++k) {
                if (i * f128::size + k >= N) {
                    break;
                }
                y[i * f128::size + k] = expf(x[i * f128::size + k] - maxVal) / sumVal;
            }
            i -= warpSize;
        }

        for (; i >= 0; i -= warpSize) {
            f128 out;
            f128 xi = load128cs(x + i * f128::size);
#pragma unroll
            for (int k = 0; k < f128::size; ++k) {
                out[k] = expf(xi[k] - maxVal) / sumVal;
            }
            store128cs(y + i * f128::size, out);
        }
    }
}


int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: softmax_forward <kernel> [blockSize] [benchmarkRepeatTimes]\n");
        return EXIT_FAILURE;
    }
    int kernel = atoi(argv[1]);

    int blockSize = BLOCK_SIZE;
    if (argc > 2) {
        blockSize = atoi(argv[2]);
    }
    int repeatTimes = REPEAT_TIMES;
    if (argc > 3) {
        repeatTimes = atoi(argv[3]);
    }

    float* input = (float*)malloc(M * N * sizeof(float));
    float* output = (float*)malloc(M * N * sizeof(float));
    float* resFromGPU = (float*)malloc(M * N * sizeof(float));
    initArrFloat(input, M * N);

    float *inputGPU, *outputGPU;
    cudaErrorCheck(cudaMalloc(&inputGPU, M * N * sizeof(float)));
    cudaErrorCheck(cudaMemcpy(inputGPU, input, M * N * sizeof(float), cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMalloc(&outputGPU, M * N * sizeof(float)));

    online_softmax_cpu(input, output, M, N);

    switch (kernel) {
        case 1:
            softmax_kernel1<<<ceilDiv(M, blockSize), blockSize>>>(inputGPU, outputGPU, M, N);
            break;
        case 2:
            softmax_kernel2<<<ceilDiv(M * 32, blockSize), blockSize>>>(inputGPU, outputGPU, M, N);
            break;

        case 3:
            online_softmax_kernel3<<<ceilDiv(M * 32, blockSize), blockSize>>>(inputGPU, outputGPU, M, N);
            break;

        case 4:
            online_softmax_kernel4<<<M, blockSize, blockSize / 32 * 2 * sizeof(float)>>>(inputGPU,
                                                                                         outputGPU, M, N);
            break;

        case 5:
            online_softmax_kernel5<<<ceilDiv(M * 32, blockSize), blockSize, 0>>>(inputGPU, outputGPU, M, N);
            break;
        default:
            printf("Error: Invalid kernel type: %i\n", kernel);
            return EXIT_FAILURE;
    }
    cudaErrorCheck(cudaDeviceSynchronize());
    cudaErrorCheck(cudaMemcpy(resFromGPU, outputGPU, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    float elapsedTime;
    if (checkResults(output, resFromGPU, M * N)) {
        switch (kernel) {
            case 1:
                benchmarkKernel(repeatTimes, softmax_kernel1, M * N / blockSize, blockSize, 0, 0,
                                &elapsedTime, inputGPU, outputGPU, M, N);
                break;
            case 2:
                benchmarkKernel(repeatTimes, softmax_kernel2, ceilDiv(M * 32, blockSize), blockSize, 0, 0,
                                &elapsedTime, inputGPU, outputGPU, M, N);
                break;
            case 3:
                benchmarkKernel(repeatTimes, online_softmax_kernel3, ceilDiv(M * 32, blockSize), blockSize,
                                0, 0, &elapsedTime, inputGPU, outputGPU, M, N);
                break;
            case 4:
                benchmarkKernel(repeatTimes, online_softmax_kernel4, M, blockSize,
                                blockSize / 32 * 2 * sizeof(float), 0, &elapsedTime, inputGPU, outputGPU, M,
                                N);
                break;
            case 5:
                benchmarkKernel(repeatTimes, online_softmax_kernel5, ceilDiv(M * 32, blockSize), blockSize,
                                0, 0, &elapsedTime, inputGPU, outputGPU, M, N);
                break;
        }
        printf(
            "softmax_forward kernel: %i | matrixSize: %i x %i | Times: %f ms | "
            "blockSize: %i\n",
            kernel, M, N, elapsedTime, blockSize);
    }

    free(input);
    free(output);
    free(resFromGPU);
    cudaErrorCheck(cudaFree(inputGPU));
    cudaErrorCheck(cudaFree(outputGPU));
    return EXIT_SUCCESS;
}
