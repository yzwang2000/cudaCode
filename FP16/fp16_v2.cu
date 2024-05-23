#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <iostream>

// host 端调用, 依据传入的 cudaError_t 类型的变量来打印错误所在位置和错误信息
// #var 是一种预处理器操作，被称为字符串化操作符。将宏参数 val 转换为一个字符串。也就是说，它会把 val 所代表的实际值转换为字符串形式。
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

// host 端调用, 主要是打印 kernel 内发生的错误信息。因为 kernel 并不能使用 checkCudaErrors。但是有几点注意事项
// 1) 其主要是用于 kernel 的检查。因为核函数是异步的, 也没有任何返回值。所以必须在核函数启动之后调用 cudaGetLastError 来检索核函数是否启动成功
// 2) 我们要确保核函数启动之前 cudaError_t 类型的变量是 cudaSuccess, 排除核函数以外的错误信息。
// 3) 由于核函数的启动是异步的, 所以必须在调用 cudaGetLastError() 前同步核函数(其实也好理解, 只有核函数执行完, 才能得到 cudaError_t 类型的变量)。
#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), cudaGetErrorName(result), func);
    exit(EXIT_FAILURE);
  }
}

inline void __getLastCudaError(const char *errorMessage, const char *file,
                               const int line) {
  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err) {
    fprintf(stderr,
            "%s(%i) : getLastCudaError() CUDA error :"
            " %s : (%d) %s.\n",
            file, line, errorMessage, static_cast<int>(err),
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

__global__ void kernel_scalarProduct(half2 *vec1, half2 *vec2, float* result, int num_ele){

    extern __shared__ half2 sm_value[];
    const int tid = threadIdx.x;
    const int id = tid + blockIdx.x * blockDim.x;

    // 每个线程先将自己负责的数据归约到共享内存中
    for(int i=0; i<(num_ele>>1); i+=gridDim.x*blockDim.x)
    {
        sm_value[tid] = __hadd2(sm_value[tid], __hmul2(vec1[i+id], vec2[i+id]));
    }
    __syncthreads();

    for(int s=(blockDim.x>>1); s>=32; s>>=1)
    {
        if(tid<s)
        {
            sm_value[tid] = __hadd2(sm_value[tid], sm_value[tid+s]);
        }
        __syncthreads();
    }

    if(tid<32)
    {
        sm_value[tid] = __hadd2(sm_value[tid], sm_value[tid+16]);
        sm_value[tid] = __hadd2(sm_value[tid], sm_value[tid+8]);
        sm_value[tid] = __hadd2(sm_value[tid], sm_value[tid+4]);
        sm_value[tid] = __hadd2(sm_value[tid], sm_value[tid+2]);
        sm_value[tid] = __hadd2(sm_value[tid], sm_value[tid+1]);
    }
    __syncthreads();

    if(tid==0)
    {
        result[blockIdx.x] = __high2float(sm_value[0]) + __low2float(sm_value[0]);
    }
}

int main(){
    constexpr int num_blocks_per_grid = 128;
    constexpr int num_threads_per_block = 256;
    constexpr int num_ele = num_blocks_per_grid * num_threads_per_block * 32;    // 数组的长度

    // half2 是用的很多的, 四个字节, 一次处理两个数
    float* vecf = (float*)malloc(sizeof(float)*num_ele);
    half2* vec = (half2*)malloc(sizeof(float)*(num_ele>>1));
    half2* devVec;
    checkCudaErrors(cudaMalloc(&devVec, sizeof(float)*(num_ele>>1)));

    // 存放 host 计算的结果和 gpu 上计算的结果
    float *result = (float*)malloc(sizeof(float)*num_blocks_per_grid);
    float *devResult;
    checkCudaErrors(cudaMalloc(&devResult, sizeof(float)*num_blocks_per_grid));

    // 在 host 端生成输入数组
    for(int i=0; i<num_ele; ++i)
    {
        vecf[i] = i % 4;
    }

    // 将 host 端 float 数组转换为 half2, 两个一组
    for(int i=0; i<(num_ele>>1); ++i)
    {
        half2 tmp = {0.f, 0.f};
        tmp.x = vecf[2*i];
        tmp.y = vecf[2*i+1];
        vec[i] = tmp;
    }
    
    // 从 host 端拷贝到 gpu 端
    checkCudaErrors(cudaMemcpy(devVec, vec, sizeof(half2)*(num_ele>>1), cudaMemcpyHostToDevice));
    kernel_scalarProduct<<<num_blocks_per_grid, num_threads_per_block, num_threads_per_block*sizeof(half2)>>>(devVec, devVec, devResult, num_ele);
    cudaDeviceSynchronize();
    getLastCudaError("scalarProduct");  // 检查核函数是否执行正确
    checkCudaErrors(cudaMemcpy(result, devResult, sizeof(float)*num_blocks_per_grid, cudaMemcpyDeviceToHost));
    
    // 验证计算结果的准确性
    float sum_host {0.f};
    float sum_gpu {0.f};
    for(int i=0; i<num_ele; ++i)
    {
        sum_host += vecf[i]*vecf[i];
    }

    for(int i=0; i<num_blocks_per_grid; ++i)
    {
        sum_gpu += result[i];
    }

    if(std::abs(sum_host-sum_gpu)<1e-5){
        std::cout << "Sucess!" << std::endl;
    }else{
        std::cout << "Failed! " << std::endl;
    }

    cudaFree(devResult);
    cudaFree(devVec);
    free(vecf);
    free(vec);
    free(result);

    return 0;
}