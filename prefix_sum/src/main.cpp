#include "scan.cuh"
#include "utils.h"
#include <cstdio>
#include <cstdlib>
#include <tuple>


int main(int argc, char **argv)
{
    warm_up();  // 先进行 warm_up 的操作
    int nums[] = {1000, 2048, 100000, 10000000};
    int len = sizeof(nums) / sizeof(int);  // 数组的长度, 有几个长度的大小

    for (int i = 0; i < len; i++)
    {
        int N = nums[i];  // 数据的总个数
        size_t arr_size = N * sizeof(int);  // 数据的字节数
        int *data = (int *)malloc(arr_size);  // cpu 上的原始数据
        int *prefix_sum_cpu = (int *)malloc(arr_size);  // cpu 上计算的前缀和的结果
        int *prefix_sum_gpu = (int *)malloc(arr_size);  // gpu 上计算的前缀和的结果
        float total_cost, kernel_cost;
        data_init(data, N);  // 初始化数据

        printf("-------------------------- N = %d --------------------------\n", N);

        // 1) cpu 上进行计算
        total_cost = scan_cpu(data, prefix_sum_cpu, N);
        printf("%35s - total: %10.5f ms\n", "scan_cpu", total_cost);
        
        // 2) gpu 上只使用一个线程进行计算
        // std::tie 函数返回一个 tuple, 是将另一个 tuple 拷贝到这个 tuple 的过程
        std::tie(total_cost, kernel_cost) = sequential_scan_gpu(data, prefix_sum_gpu, N);
        results_check(prefix_sum_cpu, prefix_sum_gpu, N);
        printf("%35s - total: %10.5f ms    kernel: %10.5f ms\n", "sequential_scan_gpu", total_cost, kernel_cost);

        // 3) gpu 上并行计算, 但是数组个数必须小于等于一个 block 能处理的个数 N <= 1024*2, 分为解决 bank 冲突和不解决
        if (N <= MAX_ELEMENTS_PER_BLOCK)  // 元素个数必须小于等于 2048, 因为一个线程处理两个元素
        {
            std::tie(total_cost, kernel_cost) = parallel_block_scan_gpu(data, prefix_sum_gpu, N, false);
            results_check(prefix_sum_cpu, prefix_sum_gpu, N);
            printf("%35s - total: %10.5f ms    kernel: %10.5f ms\n", "parallel_block_scan_gpu", total_cost,
                   kernel_cost);

            std::tie(total_cost, kernel_cost) = parallel_block_scan_gpu(data, prefix_sum_gpu, N, true);
            results_check(prefix_sum_cpu, prefix_sum_gpu, N);
            printf("%35s - total: %10.5f ms    kernel: %10.5f ms\n", "parallel_block_scan_gpu with bcao", total_cost,
                   kernel_cost);
        }

        // 4) gpu 上并行计算, 数组个数大于一个 block 能够处理的个数, 分成多个 block 来处理。分为解决 bank 冲突和不解决
        std::tie(total_cost, kernel_cost) = parallel_large_scan_gpu(data, prefix_sum_gpu, N, false);
        results_check(prefix_sum_cpu, prefix_sum_gpu, N);
        printf("%35s - total: %10.5f ms    kernel: %10.5f ms\n", "parallel_large_scan_gpu", total_cost, kernel_cost);

        std::tie(total_cost, kernel_cost) = parallel_large_scan_gpu(data, prefix_sum_gpu, N, true);
        results_check(prefix_sum_cpu, prefix_sum_gpu, N);
        printf("%35s - total: %10.5f ms    kernel: %10.5f ms\n", "parallel_large_scan_gpu with bcao", total_cost,
               kernel_cost);

        free(data);
        free(prefix_sum_cpu);
        free(prefix_sum_gpu);
        printf("\n");
    }
}