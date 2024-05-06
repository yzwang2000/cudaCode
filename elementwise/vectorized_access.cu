#include <cuda.h>
#include <iostream>

constexpr int N = 512000;
constexpr int thread_per_block = 256;                  // 每个块中包含的线程个数
constexpr int num_per_block = 512;                     // 每个块处理的数据个数
constexpr int num_block_per_grid = N/num_per_block;    // 网格中包含的线程块的个数

__global__ void kernel_A(int *d_s, int *d_o){
    int tid = threadIdx.x;
    int *bd_s = d_s + blockIdx.x * num_per_block;
    int *bd_o = d_o + blockIdx.x * num_per_block;

    for(int i=tid; i<num_per_block; i+=blockDim.x)
    {
        bd_o[i] = bd_s[i];
    }
}

// 这个是默认每个线程负责的元素个数是 2
__global__ void kernel_B(int *d_s, int *d_o){
    int tid = threadIdx.x;
    int2 *bd_s = reinterpret_cast<int2*>(d_s + blockIdx.x * num_per_block);
    int2 *bd_o = reinterpret_cast<int2*>(d_o + blockIdx.x * num_per_block);

    bd_o[tid] = bd_s[tid];
}

// 每个线程负责的元素个数不能被线程数整除
__global__ void kernel_C(int *d_s, int *d_o){
    int tid = threadIdx.x;  // 每个块内的线程 id
    
    // 网格跨步法处理能够向量化读取的部分
    for(int i=tid; i < num_per_block/2; i+=blockDim.x)
    {
        reinterpret_cast<int2*>(d_o)[i] = reinterpret_cast<int2*>(d_s)[i];
    }

    // 处理不够 2 的部分
    int remainder = num_per_block % 2;  // 余数
    int quotient = num_per_block / 2;  // 商
    if(tid < remainder)
    {
        d_o[2*quotient+tid] = d_s[2*quotient+tid];
    }

}

int main(){
    // host 分配内存
    int *h_s = (int*)malloc(N*sizeof(int));
    int *h_o = (int*)malloc(N*sizeof(int));
    for(int i=0; i<N; ++i)
    {
        h_s[i] = 1;
        h_o[i] = 0;
    }

    // gpu 分配显存
    int *d_s = nullptr;
    int *d_o = nullptr;
    cudaMalloc(&d_s, N*sizeof(int));
    cudaMalloc(&d_o, N*sizeof(int));

    // 显存拷贝
    cudaMemcpy(d_s, h_s, N*sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid(num_block_per_grid);
    dim3 block(thread_per_block);
    kernel_A<<<grid, block>>>(d_s, d_o);
    kernel_B<<<grid, block>>>(d_s, d_o);
    kernel_C<<<grid, block>>>(d_s, d_o);

    cudaMemcpy(h_o, d_o, N*sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    int sum = 0;
    for(int i=0; i<N; i++)
    {
        sum += (h_s[i]-h_o[i]);
    }
    if(sum)
    {
        std::cout << "fail : " << sum << std::endl;
    }
    else{
        std::cout << "sucess" << std::endl;
    }

    free(h_s);
    free(h_o);
    cudaFree(d_s);
    cudaFree(d_o);

    return 0;
}