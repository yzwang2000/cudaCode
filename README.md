# cudaCode
此仓库是本人在秋招准备过程中关于 `CUDA` 编程技巧的总结

## NVIDIA Nsight Compute 和 NVIDIA Nsight Systems
* `Nsight Systems` 提供全局视图的性能分析, 包括整体应用的执行流程、资源使用和性能特性。`Nsight Systems` 不仅能够分析 GPU 性能, 也能够分析 CPU、内存和系统级的性能特性。使用时主要关注: 应用整个上各个核函数以及操作消耗的事情顺序, CPU 和 GPU 之间的数据传输耗时, 多个 Stream 之间的调度信息, SM warp occupancy。
* `Nsight Compute` 对核函数的性能特性和瓶颈进行详细的分析。使用时主要关注: SM 的吞吐量, 依据 roofline model 分析当前核函数是属于计算密集型, 还是访存密集型, 估算核函数不同线程配置对 warp occupancy 的影响。L1 cache 和 L2 cache 的吞吐量和命中率。

# Roofline Model 的介绍
* Roofline Model 其实是说明模型在一个计算平台的限制下, 到底能够达到多快的浮点计算速度。具体来说解决的问题是 `计算量为A且访存量为B的模型在算力为C且带宽为D的计算平台所能达到的理论性能上限E是多少`。Roofline 划分出了计算瓶颈取余和贷款瓶颈区域。模型是实际表现一定是越贴近于边界越好的, 最理想的情况, 是实际表现达到拐点处。
![Roofline Model](./fig/roofline.png)

# Reduce 优化
* reduce 算法也就是规约运算, 本质上是 $ x = x_0 \otimes x_1 \otimes x_2 \cdots \otimes x_n $。在并行计算中通常采用树形的及计算方式。比如计算长度为 $N$ 的数组的所有元素之和。首先将数组分成 $m$ 个小份, 开启 $m$ 个 block 计算出 $m$ 个小份的 reduce 的值。接下来再使用一个 block 将 $m$ 个小份再次进行 reduce, 得到最终的结果。
![reduce](./fig/reduce.jpg)
* 对于线程模型的分配, 线程模型的块的个数尽量给到 `SM` 个数的整数倍, 一个块内线程的个数通常是 `128, 256, 512, 1024`。在线程与元素一一对应不能满足的时候, 通常先满足块的个数是整数的要求, 然后再分配适当的线程, 使用 `块跨步` 的方法使得一个线程块的线程能够遍历完分配其的所有数据。进行一个 block 内规约的方法通常有两种, 第一种方法如下所示:
```C++
__global__ void Kernel_A(int *d_s, int *d_o)
{
    int tid = threadIdx.x;
    int *b_s = d_s + blockIdx.x * num_per_block;

    // 1) 每个线程先将负责的部分的数据规约到自己的寄存器中
    int sum = 0;
    for(int i=tid; i<num_per_block; i+=blockDim.x)
    {
        sum += b_s[i];
    }
    
    // 2) 每个块中都分配共享内存, 共享内存负责存储每个线程的变量的值
    __shared__ int tmp_sum[thread_per_block];
    tmp_sum[tid] = sum;
    __syncthreads();

    // 3) 这样规约的好处, 1) 避免了线程束的分歧 2) 不存在 blank 冲突 3) 最后一个 warp 内不需要同步, 避免了同步造成的影响
    for(int s=blockDim.x/2; s>16; s>>=1)
    {
        if(tid<s)
        {
            tmp_sum[tid] += tmp_sum[tid+s];
        }
        __syncthreads();  // 这个不能放到 if 里面
    }

    // 一个 warp 中的所有线程无论什么时候, 都是处在同一种状态, SIMD 的特点
    if(tid<16)  // 因为是同一个 warp 所以不需要同步
    {
        tmp_sum[tid] += tmp_sum[tid+16];
        tmp_sum[tid] += tmp_sum[tid+8];
        tmp_sum[tid] += tmp_sum[tid+4];
        tmp_sum[tid] += tmp_sum[tid+2];
        tmp_sum[tid] += tmp_sum[tid+1];
    }

    // 4) 将每个块规约之后的值赋值到输出
    if(tid==0) d_o[blockIdx.x] = tmp_sum[0];
}
```
* 对于代码中的 `1)`, 每个线程都将自己负责的部分首先进行归约到自己的寄存器变量中。尽可能每个块分配比较少的线程, 让一个线程负责多个数据, 不然每次进行一次规约都有很多的线程不干活。
* 对于代码中的 `2)`, 根据每个块中线程的个数分配共享内存的大小, 让后把每个线程中寄存器的值, 写入到共享内存中。这里的 `__syncthreads()` 是为了同步每个块中的线程, `同一warp中的线程一定是处于同一状态和进度的, 但是不同 warp 的线程所处状态是不确定的, 也就是不同 warp 执行到的地方可能不相同`。读入共享内存中, 是因为 `共享内存能够进行块间数据的共享和同步`(因为我们需要进行块内规约, 需要这一特点)。
* 对于代码中的 `3)`, 这里 `blockDim.x` 一定是能够被 `32` 整除的。这样的写法有很多优势
    * 避免了线程束的分歧, 因为最后一次规约的时候, 步长为 `32`, 所有 `warp` 都是同一状态, 要么执行, 要么不执行。
    * 避免了共享内存中的 blank 冲突, 以 `warp 0` 为例分析, 当 `s=128` 时, `warp 0` 中的 0 线程访问 0 和 128 元素, 都是位于 0 bank,  1 线程访问 1 和 129 元素, 都是位于 1 bank, ... 31 线程方位 31 和 159 元素, 都位于 31 bank, 不存在冲突。
    * 对于最后一个 warp 内值的规约, 这里并没有在循环中使用, 因为最后一个 `warp` 内的所有线程都是同步的, 对共享内存的访问也是同步的, 不需要同步。
* 对于代码中的 `4)`, 这里并没有做同步, 因为所有对共享内存的操作都是在一个 warp 内完成的, 减少同步损耗。
* 规约之后规约的值就位于每个块中共享内存中的第一个值。
* 规约注意的事项就是, 是共享内存块的规约, 也就是每个块规约出一个值, 如果想不同块之间再进行规约, 可以
    * 每个块的结果传回主机端, 然后 host 端进行规约(数据拷贝造成损耗)
    * 先把每个块规约的结果写到全局内存中, 然后利用 `一个块` 对这些数据进行规约(两步规约的办法)。
* 第二种方法如下所示:
```C++
// 每个 warp 进行规约, 规约后的值在 warp 中的 0 号线程
// __shfl_down_sync 相比于 __shfl_down 允许开发者指定一个线程掩码, 确保只有在指定的线程都完成数据交换后, 才继续执行后续的操作
__device__ int warpReduceSum(int sum)
{
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
    
    // sum += __shfl_xor_sync(0xffffffff, sum, 16);
    // sum += __shfl_xor_sync(0xffffffff, sum, 8);
    // sum += __shfl_xor_sync(0xffffffff, sum, 4);
    // sum += __shfl_xor_sync(0xffffffff, sum, 2);
    // sum += __shfl_xor_sync(0xffffffff, sum, 1);

    return sum;
}

__device__ int blockReduceSum(int sum)
{
    int laneId = threadIdx.x % warpSize;  // 每个 warp 中第几个线程
    int warpId = threadIdx.x / warpSize;  // 属于第几个 warp

    // 填充进去
    sum = warpReduceSum(sum);
    __shared__ int reduceSum[32];         // 最多也就 32 个 warp
    if(laneId==0)
    {
        reduceSum[warpId] = sum;
    }
    __syncthreads();

    bool pred = (warpId==0) && (laneId < blockDim.x / warpSize);
    int value =  (int)pred * reduceSum[laneId];  // 在范围内的为正常的数字, 否则为 0
    if(warpId==0) value = warpReduceSum(value);
    if(warpId==0 && laneId==0) sum=value;

    return sum;
}


__global__ void Kernel_B(int *d_s, int *d_o)
{
    int tid = threadIdx.x;  // 每个块内线程的 id
    int *b_s = d_s + blockIdx.x * num_per_block;

    // 1) 每个线程先将负责的部分的数据规约到自己的寄存器中
    int sum = 0;
    for(int i=tid; i<num_per_block; i+=blockDim.x)
    {
        sum += b_s[i];
    }
    sum = blockReduceSum(sum);

    if(tid==0) d_o[blockIdx.x] = sum;
}
```
* 对于第二种方法, 其实整体思路就是, 每个线程先将负责的那部分数据归约到总计的寄存器中, 然后 `warp` 内先进行一次规约, 规约的结果存储在每个 `warp` 的第一个线程中。然后再分配共享内存, 此时共享内存只需要再分配最多 32 个元素字节的大小(因为一个块内最多有 1024 个线程, 对应 32 个 `warp`)。但是这里有个问题, 就是得确定真正参与规约的每个块内有多少个 `warp`(确定个数来给共享内存32个无效的位置填充适当的数字)。最后再对这个 32 个共享内存数据进行规约, 这里使用的方法是, 每个块内 0 号 `warp` 读取这 32 个值, 然后再来一次 warp 内的规约。其实也可以像第一种方法一样, 直接操作共享内存(这里涉及到需要给共享内存写没用到的值, 或者依据线程个数只规约想要的部分)。