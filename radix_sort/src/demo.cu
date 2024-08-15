#include <cuda.h>
#include <iostream>

constexpr int intVecLen = 1 << 20;          // 输入数组的长度
constexpr int max_thread_each_block = 128;  // 每个 block 中使用的最大线程个数

void launch_radix_sort(unsigned int* d_in, unsigned int* d_out){
    int num_thread_each_block = max_thread_each_block;
    int num_block_each_grid = (intVecLen + num_thread_each_block-1) / num_thread_each_block;  // block 向上取整

    unsigned int * d_prefix_sums;      // inVecLen, d_prefix_sums[i] 表示 d_in[i] 在其所在 block 中, 所属 way 的前缀和。
    unsigned int * d_block_sums;       // 4 * num_block_each_grid, d_block_sums[i] 表示第 i%num_block_each_grid 个 block 的 i/num_block_each_grid way 的元素总个数。 
    unsigned int * d_scan_block_sums;  // 4 * num_block_each_grid,  d_blocks_sums[i] 表示第 i%num_block_each_grid 个 block 的 i/num_block_each_grid way 个元素总数的累加和。

    for(unsigned int shift_width = 0; shift_width<=30; shift_width+=2)
    {
        // 每个 block 内进行一次局部 radix sort, 得到 d_prefix_sums 和 d_block_sums
        radix_sort_local<<<num_block_each_grid, num_thread_each_block>>>();
        // 通过 d_scan_block_sums 得到 d_block_sums
        sum_scan();
        // 得到 d_in 中每个元素的新坐标, 然后进行变换过去
        global_shuffle<<<>>>(d_in, d_out);
    }
}

// d_prefix_sums
// d_block_sums
// d_scan_block_sums

int main(){

    return 0;
}