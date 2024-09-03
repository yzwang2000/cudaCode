// 这个是 GEMM 的伪代码版本
#include <iostream>
#include <cuda.h>

// A is M*K, B is K*N, C is M*N
constexpr int M = 1024;
constexpr int K = 256;
constexpr int N = 1024;

// split C into many block, size of each block
constexpr int block_m = 128;
constexpr int block_n = 128;
constexpr int block_k = 8;
constexpr int maxIter = K/block_k;
constexpr int num_block_each_grid = M/block_m * N/block_n;

// split each block int many block, size of small block
constexpr int thread_m = 8;
constexpr int thread_n = 8;
constexpr int thread_k = 1;
constexpr int minIter = block_k/thread_k;
constexpr int num_thread_each_block = block_m/thread_m * block_n/ thread_n;  // 256

__global__ void gemm(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C){
    // 将每一块的数据从全局内存中读入到共享内存中
    __shared__ float smA[block_k][block_m];
    __shared__ float smB[block_k][block_n];

    // 从共享内存读取到每个线程的寄存器中
    float regA[thread_m];
    float regB[thread_n];

    // 最后直接存取到 C 对应的位置
    float accum[thread_m][thread_n] = {0.f};  // 注意这样初始化为 0 的方式

    // 大迭代
    #pragma unroll
    for(int i=0; i<maxIter; i+=block_k)
    {
        // 依据 blockIdx.x 和 blockIdx.y 将全局内存中的数据读取到共享内存 smA 和 smB 
        // 对 A, B 均是逐行存取，对 A 逐列写入，对 B 仍然是逐行写入
        // 装载完成 smA 和 smB
        __syncthreads();

        #pragma unroll
        for(int j=0; j<minIter; j+=thread_k)
        {
            // 每次将 smA 的 j 行读取到 regA 中
            // 每次将 smB 的 j 行读取到 regB 中
            __syncthreads();
            for(int k=0; k<thread_m; ++k)
            {
                for(int g=0; g<thread_n; ++g)
                {
                    // 将结果累加到 accum
                    accum[k][g]+=regA[k]*regB[g];
                }
            }
        }
    }

    // accum 的结果直接写入全局内存中
}

int main(){
    dim3 grid(N/block_n, M/block_m, 1);
    dim3 block(block_n/thread_n, block_m/thread_m);

    return 0;
}