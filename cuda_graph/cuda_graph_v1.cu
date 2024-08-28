#include <cuda_runtime.h>
#include <iostream>

#define N 500000     // tuned such that kernel takes a few microseconds
#define NSTEP 1000
#define NKERNEL 20

constexpr int numVec = 200000;               // 数组中元素的个数
constexpr int num_thread_each_block = 512;   // 设置每个 block 的线程总数

__global__ void shortKernel(float * out_d, float * in_d){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx<N) out_d[idx] = 1.23*in_d[idx];
}

int main(){
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    dim3 threads(num_thread_each_block, 1, 1);
    dim3 blocks((numVec+num_thread_each_block-1)/num_thread_each_block, 1, 1);

    float* out_d, *in_d;
    cudaMallocHost(&out_d, sizeof(float)*numVec, cudaHostAllocDefault);
    cudaMallocHost(&in_d, sizeof(float)*numVec, cudaHostAllocDefault);

    // start CPU wallclock timer
    for(int istep=0; istep<NSTEP; istep++){
        for(int ikrnl=0; ikrnl<NKERNEL; ikrnl++){
            shortKernel<<<blocks, threads, 0, stream>>>(out_d, in_d);
            // 这里的 cudaStreamSynchronize 能够保证后序内核在前一个内核完成之前不会启动
            cudaStreamSynchronize(stream);
        }
    }

    for(int istep=0; istep<NSTEP; istep++){
        for(int ikrnl=0; ikrnl<NKERNEL; ikrnl++){
            shortKernel<<<blocks, threads, 0, stream>>>(out_d, in_d);
        }
        // 其实允许内核在前一个内核完成之前启动(内核启动与前一个内核执行的 overlap), 从而允许启动开销隐藏在内核执行之后。
        cudaStreamSynchronize(stream);
    }

    bool graphCreated=false;
    cudaGraph_t graph;           // 类的实列, 包含定义图的结构和内容的信息
    cudaGraphExec_t instance;    // 类的实列, 是一个`可执行图`, 以一种可以类似单个内核启动的方式启动和执行的形式表示图。
    for(int istep=0; istep<NSTEP; istep++){
        if(!graphCreated){  // 只需要创建一个次就可以了
            // 在流中开始捕获 graph。当流处于捕获模式时, 所有推入流的操作都不会被执行, 而是捕获到图形中, 该图形通过 cudaStreamEndCapture 返回。
            // 只有未开启捕获的 stream, 再能开启捕获流。
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
            for(int ikrnl=0; ikrnl<NKERNEL; ikrnl++){
                shortKernel<<<blocks, threads, 0, stream>>>(out_d, in_d);
            }
            // 在流上结束捕获, 通过 cudaGraph_t 返回捕获的图形。
            cudaStreamEndCapture(stream, &graph);
            // 从一个图创建一个可执行的图, 后三个参数都是用于捕获函数调用的错误信息。
            cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
            graphCreated=true;
        }
        // 在流中启动可执行图, 其实就是将可执行图加入到 stream 的任务队列中了。
        cudaGraphLaunch(instance, stream);
        cudaStreamSynchronize(stream);
    }

    cudaFreeHost(out_d);
    cudaFreeHost(in_d);
    cudaStreamDestroy(stream);

    return 0;
}