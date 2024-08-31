#include <cuda.h>
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

__global__ void kernel_A(float*__restrict__ d_i, float*__restrict__ d_o, int offset)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float a = sinf(d_i[idx]+offset);
    float b = sinf(d_i[idx]+offset);
    float c = sqrtf(a+b);
    d_o[idx] = a+b+c;
}

int main(){
    int numStream = 4, blockSize = 256;             // 流的个数, 每个 block 处理的元素个数
    int numEle = 1024 * 256;                        // 元素的总个数
    int streamSize = numEle / numStream;            // 每个流处理的元素个数
    int eleBytes = numEle * sizeof(float);          // 元素的总字节数
    int streamBytes = streamSize * sizeof(float);   // 每个流处理的总字节数

    // 分配主机内存和显存, 内存分配在主机端才有花样, 全局内存没有什么花样的
    float *h_i, *h_o, *d_i, *d_o;
    checkCudaErrors(cudaMallocHost(&h_i, eleBytes));
    checkCudaErrors(cudaMallocHost(&h_o, eleBytes));
    checkCudaErrors(cudaMalloc(&d_i, eleBytes));
    checkCudaErrors(cudaMalloc(&d_o, eleBytes));

    // 创建 event 和 stream
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaStream_t streamVec[numStream];
    for(int i=0; i<numStream; ++i)
    {
        cudaStreamCreate(&streamVec[i]);
    }

    float letancy = 0.f;
    memset(h_i, 0, eleBytes);

    // 使用默认流, 默认流就是 copy -> kernel -> copy
    checkCudaErrors(cudaEventRecord(start_event, 0));
    checkCudaErrors(cudaMemcpy(d_i, h_i, eleBytes, cudaMemcpyHostToDevice));
    kernel_A<<<numEle/blockSize,blockSize>>>(d_i, d_o, 0);
    // cudaDeviceSynchronize();
    // getLastCudaError("default stream");
    checkCudaErrors(cudaMemcpy(h_o, d_o, eleBytes, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaEventRecord(stop_event, 0));
    checkCudaErrors(cudaEventSynchronize(stop_event));
    checkCudaErrors(cudaEventElapsedTime(&letancy, start_event, stop_event));
    std::cout << "default stream letancy: " << letancy << " ms" << std::endl;

    // 使用多个流, 实现 copy 与 kernel 的并行操作
    checkCudaErrors(cudaEventRecord(start_event, 0));  // default 上这个执行完了, 才会执行非默认流中的操作
    for(int i=0; i<numStream; ++i)
    {
      int cur = i * streamSize;  // 当前处理的元素个数
      checkCudaErrors(cudaMemcpyAsync(&d_i[cur], &h_i[cur], streamBytes, cudaMemcpyHostToDevice, streamVec[i]));
      kernel_A<<<streamSize/blockSize, blockSize, 0, streamVec[i]>>>(&d_i[cur], &d_o[cur], 0);
      checkCudaErrors(cudaMemcpyAsync(&h_o[cur], &d_o[cur], streamBytes, cudaMemcpyDeviceToHost, streamVec[i]));
    }
    checkCudaErrors(cudaEventRecord(stop_event, 0));
    checkCudaErrors(cudaEventSynchronize(stop_event));
    checkCudaErrors(cudaEventElapsedTime(&letancy, start_event, stop_event));
    std::cout << "multiple stream letancy: " << letancy << " ms" << std::endl;

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    for(int i=0; i<numStream; ++i)
    {
        cudaStreamDestroy(streamVec[i]);
    }
    cudaFreeHost(h_i);
    cudaFreeHost(h_o);
    cudaFree(d_i);
    cudaFree(d_o);

    return 0;
}