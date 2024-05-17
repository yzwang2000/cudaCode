#include <stdio.h>
#include <cuda.h>
#include <iostream>


// 32*16 大小的线程块要操作 block_size_m*block_size_n 大小的数据块
template<int block_size_m, int block_size_n> 
__global__ void Kernel_A(
    float* __restrict__ d_i,   // 输入矩阵大小为 M*N
    float* __restrict__ d_o,   // 转置输入矩阵大小为 N*M
    const int M,
    const int N
){
    /*
        第一步将全局内存中的数据读入到共享内存中
    */
    __shared__ float sm_data[block_size_m][block_size_n+1];  // 每个共享内存块都要读入这么多的数据

    int tx = threadIdx.x;  // laneId
    int ty = threadIdx.y;  // warpId
    int bx = blockIdx.x;   // x 方向是第几个块
    int by = blockIdx.y;   // y 方向是第几个块

    #pragma unroll
    for(int i=0; i<block_size_m; i+=blockDim.y)
    {
        int idx_glo = (by*block_size_m+i+ty)*N+bx*block_size_n;  // 定位到第几行, 第几列的起始位置
        reinterpret_cast<float2*>(&sm_data[i+ty][tx*2])[0] = reinterpret_cast<float2*>(&d_i[idx_glo])[tx];
    }

    __syncthreads();  // 不同的线程要操作相同的共享内存, 所以要先进行同步

    /*
        将共享内存中的数据逐列读取, 逐行写入到全局内存中
    */
    #pragma unroll
    for(int i=0; i<block_size_n; i+=blockDim.y)
    {
        int idx_glo = (bx*block_size_n+ty+i)*M + by*block_size_m+tx;
        d_o[idx_glo] = sm_data[tx][ty+i];
    }
}


int main(){
    // 要转置的矩阵的大小为 M*N
    constexpr int M = 512;
    constexpr int N = 1024;
    
    // 这里不能设置太大了, block_size_M * block_size_N 是共享内存的大小 32*64*4/1024 = 8KB
    constexpr int block_size_M = 32;
    constexpr int block_size_N = 64;

    // h_i 的形状为 M*N, h_o 的形状为 N*M
    int matrix_bytes = sizeof(float) * M * N;
    float *h_i = (float*)malloc(matrix_bytes);
    float *h_o = (float*)malloc(matrix_bytes);

    for(int i=0; i<M*N; ++i)
    {
        h_i[i] = i;
    }

    for(int i=0; i<M*N; ++i)
    {
        h_o[i] = 0;
    }

    float *d_i, *d_o;
    cudaMalloc(&d_i, matrix_bytes);
    cudaMalloc(&d_o, matrix_bytes);

    // Host->Device
    cudaMemcpy(d_i, h_i, matrix_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_o, h_o, matrix_bytes, cudaMemcpyHostToDevice);

    dim3 grid(N/block_size_N, M/block_size_M, 1);  // (512/32, 1024/64) = (16, 16)
    dim3 block(32, 16, 1);  // 使用 16 个 warp

    Kernel_A<block_size_M, block_size_N><<<grid, block>>>(d_i, d_o, M, N);

    // Device->Host
    cudaMemcpy(h_o, d_o, matrix_bytes, cudaMemcpyDeviceToHost);

    float eps = 0.;
    for(int i=0; i<M; ++i)
    {
        for(int j=0; j<N; ++j)
        {
            eps+=(h_i[i*N+j]-h_o[j*M+i]);
        }
    }

    std::cout << "误差为: " << eps << std::endl;

    return 0;
}