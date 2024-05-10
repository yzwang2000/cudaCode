#include <cuda.h>
#include <stdio.h>
#include <iostream>

__device__ int warpReduce(int val){
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

__global__ void kernel_A(
    float* __restrict__ A,
    float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int N){
    
    int tx = threadIdx.x;  // laneId
    int ty = threadIdx.y;  // warpId
    int bx = blockIdx.x;

    int per_thread_iter= N/(warpSize*4);  // 每个线程迭代的次数, 每个 warp 每次处理 warpSize * 4 的数据(向量化访存)
    A = &A[N*blockDim.y*bx + N*ty];  // 当前 block 中的 thread 处理 A 的行数
    int res=0; // 每个线程先将多次迭代的结果归约到自己的寄存器中

    for(int i=0; i<per_thread_iter; ++i)
    {
        float4 current_A = reinterpret_cast<float4*>(A)[tx*per_thread_iter+i];  // 这里不要混淆了, 指针再用 [] 索引就是数值了
        float4 current_x = reinterpret_cast<float4*>(x)[tx*per_thread_iter+i];  // 这里不要混淆了, 指针再用 [] 索引就是数值了
        res += current_A.x * current_x.x;
        res += current_A.y * current_x.y;
        res += current_A.z * current_x.z;
        res += current_A.w * current_x.w;
    }

    res = warpReduce(res);
    if(tx==0) y[bx*blockDim.y + ty] = res;
}


int main(){

    // A 的大小是 M*N, x 的大小是 N*1, y 的大小是 M*1
    constexpr int M = 16384;
    constexpr int N = 128;

    int bytes_A = M*N*sizeof(float);
    int bytes_x = N*sizeof(float);
    int bytes_y = M*sizeof(float);

    float *h_A = (float*)malloc(bytes_A);
    float *h_x = (float*)malloc(bytes_x);
    float *h_y = (float*)malloc(bytes_y);

    float *d_A, *d_x, *d_y;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_x, bytes_x);
    cudaMalloc(&d_y, bytes_y);

    // 给 A 随机值
    for(int i = 0; i < M * N; i++){
        h_A[i] = i % 13;
    }
    // 给 x 随机值
    for(int i = 0; i < N; i++){
        h_x[i] = i % 13;
    }
    // 给 y 随机值
    for(int i = 0; i < M; i++){
        h_y[i] = 0; 
    }

    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, bytes_x, cudaMemcpyHostToDevice);

    int warpNum = 8;  // 每个线程块有多少个warp, 也就是一个线程块负责多少行
    dim3 grid(M/warpNum);
    dim3 block(32, warpNum);
    kernel_A<<<grid, block>>>(d_A, d_x, d_y, M, N);

    cudaMemcpy(h_y, d_y, bytes_y, cudaMemcpyDeviceToHost);

    double eps = 0.;
    for(int i=0; i<M; ++i)
    {
        double val = 0.;
        for(int j=0; j<N; ++j)
        {
            val += h_A[i*N+j] * h_x[j];
        }
        eps += (val-h_y[i]);
    }

    std::cout << "误差为: " << eps << std::endl;

    free(h_A);
    free(h_x);
    free(h_y);
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}