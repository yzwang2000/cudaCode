#include "sort.h"

#define MAX_BLOCK_SZ 128

__global__ void gpu_radix_sort_local(
    unsigned int* d_out_sorted,        // 存放打乱顺序后的数组
    unsigned int* d_prefix_sums,       // 存放排序后数组, 每个元素在其 way 中的顺序, 这个是每个 way 的前缀和 0 1 2 3 ... 0 1 2 3 ... 0 1 2 3 ... 0 1 2 3
    unsigned int* d_block_sums,        // 存放到全局内存中, 用于记录每个 block 负责的元素的每一路元素的个数
    unsigned int input_shift_width,    // 移位的宽度
    unsigned int* d_in,                // 输入元素的指针
    unsigned int d_in_len,             // 输入元素的长度
    unsigned int max_elems_per_block   // 每个块负责处理的元素个数
    )
{
    // need shared memory array for:
    // - block's share of the input data (local sort will be put here too)
    // - mask outputs
    // - scanned mask outputs
    // - merged scaned mask outputs ("local prefix sum")
    // - local sums of scanned mask outputs
    // - scanned local sums of scanned mask outputs

    // for all radix combinations:
    // build mask output for current radix combination
    // scan mask ouput
    // store needed value from current prefix sum array to merged prefix sum array
    // store total sum of mask output (obtained from scan) to global block sum array
    // calculate local sorted address from local prefix sum and scanned mask output's total sums
    // shuffle input block according to calculated local sorted addresses
    // shuffle local prefix sums according to calculated local sorted addresses
    // copy locally sorted array back to global memory
    // copy local prefix sum array back to global memory

    extern __shared__ unsigned int shmem[];
    unsigned int* s_data = shmem;  // 存放每个 block 负责的全局数据
    // s_mask_out[] will be scanned in place, 注意这个下面的写法, 是一个累加的写法
    unsigned int s_mask_out_len = max_elems_per_block + 1;
    unsigned int* s_mask_out = &s_data[max_elems_per_block];  // len + 1
    unsigned int* s_merged_scan_mask_out = &s_mask_out[s_mask_out_len];  // len, 将各个 way 的前缀后统计到一起
    unsigned int* s_mask_out_sums = &s_merged_scan_mask_out[max_elems_per_block];  // 4, 存放这个块每一个 way 元素个数
    unsigned int* s_scan_mask_out_sums = &s_mask_out_sums[4];  // 4, 是 s_mask_out_sums 的独占前缀和

    unsigned int thid = threadIdx.x;  // 一个 block 中的线程 id

    // Copy block's portion of global input data to shared memory, 全局内存数据拷贝到共享内存中, 超边界的用 0 填充
    unsigned int cpy_idx = max_elems_per_block * blockIdx.x + thid;
    if (cpy_idx < d_in_len)
        s_data[thid] = d_in[cpy_idx];
    else
        s_data[thid] = 0;
    __syncthreads();

    //  To extract the correct 2 bits, we first shift the number
    //  to the right until the correct 2 bits are in the 2 LSBs,
    //  then mask on the number with 11 (3) to remove the bits
    //  on the left
    // 此时数据在每个寄存器中, 并且提取到了最后两位
    unsigned int t_data = s_data[thid];  // 从共享内存取到自己的寄存器中
    unsigned int t_2bit_extract = (t_data >> input_shift_width) & 3;  // 右移取出最后两个2, 这个两个2有四种可能的情况

    // 统计这个 block 中各个 way 的元素数目; 得到包含各个 way 的前缀和
    for (unsigned int i = 0; i < 4; ++i)  // 遍历这些值, 哪里为 true, 那些为 false
    {
        // 将 s_mask_out 的所有元素都置为 0, 先初始化
        s_mask_out[thid] = 0;
        if (thid == 0)  // 这个数组的元素比线程个数多一个, 所以需要单独一个线程来索引最后的值
            s_mask_out[s_mask_out_len - 1] = 0;

        __syncthreads();

        // 判断每个元素的对应两位是否是 i, 主要是为了计算相对位置。经过这里的处理, 为只有对应的桶才为 1, 其余都为 0
        bool val_equals_i = false;
        if (cpy_idx < d_in_len)
        {
            val_equals_i = t_2bit_extract == i;
            s_mask_out[thid] = val_equals_i;
        }
        __syncthreads();

        // Scan mask outputs (Hillis-Steele), 将 s_mask_out 进行包含式前缀和
        int partner = 0;
        unsigned int sum = 0;
        unsigned int max_steps = (unsigned int) log2f(max_elems_per_block);
        for (unsigned int d = 0; d < max_steps; d++) {
            partner = thid - (1 << d);
            if (partner >= 0) {
                sum = s_mask_out[thid] + s_mask_out[partner];
            }
            else {
                sum = s_mask_out[thid];
            }
            __syncthreads();
            s_mask_out[thid] = sum;
            __syncthreads();
        }

        // 包含式前缀和转换为独占式前缀和, 这样转换完以后, s_mask_out 就变成了 [0, s_mask_out[0], ..., 所有元素之和] (长度为元素和+1)
        unsigned int cpy_val = 0;
        cpy_val = s_mask_out[thid];
        __syncthreads();
        s_mask_out[thid + 1] = cpy_val;
        __syncthreads();

        if (thid == 0)
        {
            // Zero out first element to produce the same effect as exclusive scan
            s_mask_out[0] = 0;
            unsigned int total_sum = s_mask_out[s_mask_out_len - 1];  // 这一个 block 中遍历这一 way 元素总个数
            s_mask_out_sums[i] = total_sum;  // 记录每一个 way, 元素的个数
            d_block_sums[i * gridDim.x + blockIdx.x] = total_sum;  // 存放到全局内存中, 用于记录每个 block 负责的元素的每一路元素的个数
        }
        __syncthreads();

        if (val_equals_i && (cpy_idx < d_in_len))
        {
            s_merged_scan_mask_out[thid] = s_mask_out[thid];  // 进行了一个 copy, 非包含式前缀和
        }

        __syncthreads();
    }

    // Scan mask output sums
    // Just do a naive scan since the array is really small, 统计每个 way 的元素前缀和
    if (thid == 0)
    {
        unsigned int run_sum = 0;
        for (unsigned int i = 0; i < 4; ++i)
        {
            s_scan_mask_out_sums[i] = run_sum;  // 统计每个 way 的元素前缀和
            run_sum += s_mask_out_sums[i];
        }
    }

    __syncthreads();

    if (cpy_idx < d_in_len)
    {
        // Calculate the new indices of the input elements for sorting
        unsigned int t_prefix_sum = s_merged_scan_mask_out[thid];  // 取得这个元素在这个 way 中的相对位置
        unsigned int new_pos = t_prefix_sum + s_scan_mask_out_sums[t_2bit_extract];  // 取得绝对位置 (=偏移+相对位置)
        
        __syncthreads();

        // Shuffle the block's input elements to actually sort them
        // Do this step for greater global memory transfer coalescing
        //  in next step
        s_data[new_pos] = t_data;  // 调换数据顺序, 进行排序
        s_merged_scan_mask_out[new_pos] = t_prefix_sum;  // 每个元素对的在 way 中的顺序
 
        __syncthreads();

        // Copy block - wise prefix sum results to global memory
        // Copy block-wise sort results to global 
        d_prefix_sums[cpy_idx] = s_merged_scan_mask_out[thid];
        d_out_sorted[cpy_idx] = s_data[thid];  // 排序后的结果搬回到全局内存中
    }
}

__global__ void gpu_glbl_shuffle(unsigned int* d_out,  // 注意这里 d_in 和 d_out 颠倒了一下
    unsigned int* d_in,
    unsigned int* d_scan_block_sums,
    unsigned int* d_prefix_sums,
    unsigned int input_shift_width,
    unsigned int d_in_len,
    unsigned int max_elems_per_block)
{
    // get d = digit
    // get n = blockIdx
    // get m = local prefix sum array value
    // calculate global position = P_d[n] + m
    // copy input element to final position in d_out

    unsigned int thid = threadIdx.x;
    unsigned int cpy_idx = max_elems_per_block * blockIdx.x + thid;  // 线程的全程 id

    if (cpy_idx < d_in_len)
    {
        unsigned int t_data = d_in[cpy_idx];
        unsigned int t_2bit_extract = (t_data >> input_shift_width) & 3;  // 判断属于那个桶
        unsigned int t_prefix_sum = d_prefix_sums[cpy_idx];  // 这个桶位于那个 block 中 way 是多少
        unsigned int data_glbl_pos = d_scan_block_sums[t_2bit_extract * gridDim.x + blockIdx.x] + t_prefix_sum;
        __syncthreads();
        d_out[data_glbl_pos] = t_data;
    }
}

// An attempt at the gpu radix sort variant described in this paper:
// https://vgc.poly.edu/~csilva/papers/cgf.pdf
void radix_sort(unsigned int* const d_out,
    unsigned int* const d_in,
    unsigned int d_in_len)  // d_in_len 要排序是数据元素总个数
{
    unsigned int block_sz = MAX_BLOCK_SZ;  // 一个 block 最大线程个数
    unsigned int max_elems_per_block = block_sz;
    unsigned int grid_sz = d_in_len / max_elems_per_block;  // 每个线程对应一个元素,总共分成多少个 block
    // Take advantage of the fact that integer division drops the decimals
    if (d_in_len % max_elems_per_block != 0)  // 向上取整
        grid_sz += 1;

    unsigned int* d_prefix_sums;  // 存放相对位置索引
    unsigned int d_prefix_sums_len = d_in_len;
    checkCudaErrors(cudaMalloc(&d_prefix_sums, sizeof(unsigned int) * d_prefix_sums_len));
    checkCudaErrors(cudaMemset(d_prefix_sums, 0, sizeof(unsigned int) * d_prefix_sums_len));

    unsigned int* d_block_sums;  // 统计每个线程块的各个桶的数量
    unsigned int d_block_sums_len = 4 * grid_sz; // 4-way split
    checkCudaErrors(cudaMalloc(&d_block_sums, sizeof(unsigned int) * d_block_sums_len));
    checkCudaErrors(cudaMemset(d_block_sums, 0, sizeof(unsigned int) * d_block_sums_len));

    unsigned int* d_scan_block_sums;  // 每个线程块各个桶的数量的前缀和
    checkCudaErrors(cudaMalloc(&d_scan_block_sums, sizeof(unsigned int) * d_block_sums_len));
    checkCudaErrors(cudaMemset(d_scan_block_sums, 0, sizeof(unsigned int) * d_block_sums_len));

    // shared memory consists of 3 arrays the size of the block-wise input
    // and 2 arrays the size of n in the current n-way split (4)
    unsigned int s_data_len = max_elems_per_block;  // 存放数据, 每个线程对应一个数据
    unsigned int s_mask_out_len = max_elems_per_block + 1;  // 
    unsigned int s_merged_scan_mask_out_len = max_elems_per_block;
    unsigned int s_mask_out_sums_len = 4; // 4-way split
    unsigned int s_scan_mask_out_sums_len = 4;
    unsigned int shmem_sz = (s_data_len 
                            + s_mask_out_len
                            + s_merged_scan_mask_out_len
                            + s_mask_out_sums_len
                            + s_scan_mask_out_sums_len)
                            * sizeof(unsigned int);


    // for every 2 bits from LSB to MSB:
    //  block-wise radix sort (write blocks back to global memory)
    for (unsigned int shift_width = 0; shift_width <= 30; shift_width += 2)
    {
        // 每个 block 上进行排序得到最终的结果
        gpu_radix_sort_local<<<grid_sz, block_sz, shmem_sz>>>(d_out, 
                                                            d_prefix_sums, 
                                                            d_block_sums, 
                                                            shift_width, 
                                                            d_in, 
                                                            d_in_len, 
                                                            max_elems_per_block);


        // scan global block sum array, 四路分别进行规约
        sum_scan_blelloch(d_scan_block_sums, d_block_sums, d_block_sums_len);

        // scatter/shuffle block-wise sorted array to final positions
        gpu_glbl_shuffle<<<grid_sz, block_sz>>>(d_in,
                                                d_out, 
                                                d_scan_block_sums, 
                                                d_prefix_sums, 
                                                shift_width, 
                                                d_in_len, 
                                                max_elems_per_block);
    }

    checkCudaErrors(cudaMemcpy(d_out, d_in, sizeof(unsigned int) * d_in_len, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaFree(d_scan_block_sums));
    checkCudaErrors(cudaFree(d_block_sums));
    checkCudaErrors(cudaFree(d_prefix_sums));
}
