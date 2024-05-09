#include <cuda.h>
#include <iostream>

constexpr int N = 512000;
constexpr int thread_per_block = 256;  // 每个块中包含的线程个数
constexpr int num_per_block = 512;  // 每个处理的数据个数
constexpr int num_block_per_grid = N/num_per_block;


// 在不使用 shuffle 的情况下, 进行规约操作, 需要 shared_memory
// 因为除了 global memory 就剩下 shared_meory 可以做为线程之间通信的中介
// 除了做为线程之间通信介质, 还得需要同步的操作. global_memory 也能进行同步操作 (__threadfence)
__global__ void Kernel_A(int *d_s, int *d_o)
{
    int tid = threadIdx.x;
    int *b_s = d_s + blockIdx.x * num_per_block;

    // 每个线程先将负责的部分的数据规约到自己的寄存器中
    int sum = 0;
    #pragma unroll
    for(int i=tid; i<num_per_block; i+=blockDim.x)
    {
        sum += b_s[i];
    }
    
    // 每个块中都分配共享内存, 共享内存负责存储每个线程的变量的值
    __shared__ int tmp_sum[thread_per_block];
    tmp_sum[tid] = sum;
    __syncthreads();

    // 这样规约的好处, 1) 避免了线程束的分歧 2) 不存在 blank 冲突 3) 最后一个 warp 内不需要同步, 避免了同步造成的影响
    #pragma unroll
    for(int s=blockDim.x/2; s>16; s>>=1)
    {
        if(tid<s)
        {
            tmp_sum[tid] += tmp_sum[tid+s];
        }
        __syncthreads();  // 这个不能放到 if 里面
    }

    // 一个 warp 中的所有线程无论什么时候, 都是处在同一种状态, SIMD 的特点
    if(tid<16)  // 因为是同一个 warp 所以不需要同步
    {
        tmp_sum[tid] += tmp_sum[tid+16];
        tmp_sum[tid] += tmp_sum[tid+8];
        tmp_sum[tid] += tmp_sum[tid+4];
        tmp_sum[tid] += tmp_sum[tid+2];
        tmp_sum[tid] += tmp_sum[tid+1];
    }

    if(tid==0) d_o[blockIdx.x] = tmp_sum[0];
}


// 每个 warp 进行规约, 规约后的值在 warp 中的 0 号线程
// __shfl_down_sync 相比于 __shfl_down 允许开发者指定一个线程掩码, 确保只有在指定的线程都完成数据交换后, 才继续执行后续的操作
__device__ int warpReduceSum(int sum)
{
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
    
    // sum += __shfl_xor_sync(0xffffffff, sum, 16);
    // sum += __shfl_xor_sync(0xffffffff, sum, 8);
    // sum += __shfl_xor_sync(0xffffffff, sum, 4);
    // sum += __shfl_xor_sync(0xffffffff, sum, 2);
    // sum += __shfl_xor_sync(0xffffffff, sum, 1);

    return sum;
}

__device__ int blockReduceSum(int sum)
{
    int tid = threadIdx.x;
    int laneId = threadIdx.x % warpSize;  // 每个 warp 中第几个线程
    int warpId = threadIdx.x / warpSize;  // 属于第几个 warp

    // 填充进去
    sum = warpReduceSum(sum);
    __shared__ int reduceSum[32];         // 最多也就 32 个 warp
    if(laneId==0)
    {
        reduceSum[warpId] = sum;
    }
    __syncthreads();

    bool pred = (warpId==0) && (laneId < blockDim.x / warpSize);
    int value =  (int)pred * reduceSum[laneId];  // 在范围内的为正常的数字, 否则为 0
    if(warpId==0) value = warpReduceSum(value);
    if(warpId==0 && laneId==0) sum=value;

    // if(tid<4)  // 因为是 256 个线程, 所以是 256/32 = 8 个 warp, 所以共享内存中的有效值也只有前 8 个
    // {
    //     reduceSum[tid] += reduceSum[tid+4];
    //     reduceSum[tid] += reduceSum[tid+2];
    //     reduceSum[tid] += reduceSum[tid+1];
    // }
    // if(tid==0) sum=reduceSum[0];

    return sum;
}


__global__ void Kernel_B(int *d_s, int *d_o)
{
    int tid = threadIdx.x;  // 每个块内线程的 id
    int *b_s = d_s + blockIdx.x * num_per_block;

    // 每个线程先将负责的部分的数据规约到自己的寄存器中
    int sum = 0;
    #pragma unroll
    for(int i=tid; i<num_per_block; i+=blockDim.x)
    {
        sum += b_s[i];
    }
    sum = blockReduceSum(sum);

    if(tid==0) d_o[blockIdx.x] = sum;
}


int main(){
    int *h_s = (int*)malloc(N*sizeof(int));
    int sum = 0;
    for(int i=0; i<N; ++i)
    {
        h_s[i] = 1;
    }

    for(int i=0; i<N; ++i)
    {
        sum += h_s[i];
    }
    std::cout << "cpu 上的结果为: " << sum << std::endl;

    int *d_s = nullptr;
    cudaMalloc(&d_s, N*sizeof(int));
    cudaMemcpy(d_s, h_s, N*sizeof(int), cudaMemcpyHostToDevice);

    int *d_o = nullptr;
    cudaMalloc(&d_o, num_block_per_grid*sizeof(int));

    // 分配网格和网格中线程的个数, 这里是一维网格模型
    dim3 grid(num_block_per_grid);
    dim3 block(thread_per_block);
    // Kernel_A<<<grid, block>>>(d_s, d_o);
    Kernel_B<<<grid, block>>>(d_s, d_o);

    int *h_o = (int*)malloc(num_block_per_grid*sizeof(int));
    cudaMemcpy(h_o, d_o, num_block_per_grid*sizeof(int), cudaMemcpyDeviceToHost);
    int sum_gpu = 0;
    for(int i=0; i<num_block_per_grid; ++i)
    {
        sum_gpu += h_o[i];
    }
    std::cout << "gpu 上的结果是：" << sum_gpu << std::endl;

    std::cout << "success!" << std::endl;
    free(h_o);
    free(h_s);
    cudaFree(d_s);
    cudaFree(d_o);

    return 0;
}