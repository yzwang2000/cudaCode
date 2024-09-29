#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

__device__ __inline__ int float_atomic_add(float *dst, float src){
    int old = __float_as_int(*dst), expect;  // 因为 atomicCAS 只能实现 int 的比较与写入
    do {
        expect = old;
        old = atomicCAS((int *)dst, expect,
                    __float_as_int(__int_as_float(expect) + src));
    } while (expect != old);

    return old;
}

__global__ void parallel_sum(float *sum, float const *arr, int n) {
    // 每个线程先将自己负责的数据归约到寄存器中
    float local_sum = 0;
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        local_sum += arr[i];
    }

    // 将数据归约到全局内存中
    float_atomic_add(&sum[0], local_sum);
}

int main() {
    return 0;
}