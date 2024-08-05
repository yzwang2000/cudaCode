// 一般都是使用 safe_sotmax
// Softmax: half -> float -> half, {M, N}, N ~ (256, 1024)
#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda.h>
#include <cuda_fp16.h>
#include <vector>
#include <xmmintrin.h>
#include <omp.h>
#include <mma.h>
#include <limits>

constexpr int M = 1024;
constexpr int N = 423;

constexpr int num_block_each_grid = M;
constexpr int num_thread_each_block = 256;

template<typename T>
__device__ T warpReduce(T value, T(*op)(T, T)){
    value = op(value, __shfl_xor_sync(0xffffffff, value, 16));
    value = op(value, __shfl_xor_sync(0xffffffff, value, 8));
    value = op(value, __shfl_xor_sync(0xffffffff, value, 4));
    value = op(value, __shfl_xor_sync(0xffffffff, value, 2));
    value = op(value, __shfl_xor_sync(0xffffffff, value, 1));
    return value;  // 注意这里的返回值
}

template<typename T>
__device__ T blockReduce(T value, T(*op)(T, T), T initValue){
    int tid = threadIdx.x; 
    int laneId = tid % warpSize;
    int warpId = tid / warpSize;

    // 每个 warp 内进行规约，将规约值存放到 warp 中的第一个线程
    value = warpReduce(value, op);

    // 将每个 warp 中第一个线程的元素存放到 SM 中(SM做为线程间通信的媒介)
    __shared__ T sms[32];
    if(laneId==0){
        sms[warpId] = value;
    }
    __syncthreads();

    // 将 SM 中的数据读取到第一个warp中
    bool isValid = (warpId==0) && (laneId < blockDim.x / warpSize);
    T reduceValue = isValid ? sms[laneId] : initValue;
    if(warpId==0)
    {
        reduceValue = warpReduce(reduceValue, op);
    }
    if(warpId==0 && laneId==0) value = reduceValue;

    return value;
}

template<typename T>
__device__ T getMax(T v1, T v2){
    return v1 > v2 ? v1 : v2;
}

template<typename T>
__device__ T getSum(T v1, T v2){
    return v1 + v2;
}

__global__ void softmax_kernel(half* __restrict__ d_s, half* __restrict__ d_o){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    half* hd_s = d_s + N*bid;
    half* hd_o = d_o + N*bid;

    // 存放每一行的最大值, 存放每一行的累加和
    __shared__ float maxBlock = 0.f;
    __shared__ float sumExpBlock = 0.f;

    // 计算每个块的最大值，规约结果->SM->每个线程的寄存器
    float maxValue = -std::numeric_limits<float>::max();
    for(int i=tid; i<N; i+=blockDim.x)
    {
        maxValue = getMax<float>(maxValue, __half2float(hd_s[i]));
    }
    maxValue = blockReduce<float>(maxValue, getMax, -std::numeric_limits<float>::max());
    if(tid==0) maxBlock = maxValue;
    __syncthreads();
    maxValue = maxBlock;
    __syncthreads();

    float sumExpValue = 0.f;
    for(int i=tid; i<N; i+=blockDim.x)
    {
        sumExpValue = getSum<float>(sumExpValue, __expf(__half2float(hd_s[i])-maxValue));
    }
    sumExpValue = blockReduce<float>(sumExpValue, getSum, 0.f);
    if(tid==0) sumExpBlock = sumExpValue;
    __syncthreads();
    sumExpValue = sumExpBlock;
    __syncthreads();

    for(int i=tid; i<N; i+=blockDim.x)
    {
        hd_o[i] = __float2half(__expf(__half2float(hd_s[i])-maxValue) / sumExpBlock);
    }
}

void launch_softmax(half* __restrict__ d_s, half* __restrict__ d_o){
    dim3 grid(num_block_each_grid, 1, 1);
    dim3 block(num_thread_each_block, 1, 1);

    softmax_kernel<<<grid, block>>> (d_s, d_o);

    cudaError_t err = cudaGetLastError();
    if(err!=cudaSuccess)
    {
        std::cout << "Cuda Error" << std::endl;
    }
}

int main() {
    return 0;
}