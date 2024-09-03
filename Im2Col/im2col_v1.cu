#include <cuda_runtime.h>
#include <iostream>


/*
    实现卷积的 im2col, 输入特征图 input [bs, C_in, height, width]
    卷积核 filter [C_out, C_in, kernel_h, kernel_w], stride_h, stride_w, padding_h, padding_w
    输出结果 output [bs, C_out, height_col, width_col]

    先计算卷积后输出特征图的大小 height_out = (height+2*padding_h-kernel_h)/stride_h + 1; 
    width_out = (width+2*padding_w-kernel_w)/stride_w + 1;

    整体的运算过程其实就是比较简单的思路, 
    1) 将 im矩阵先转换成 col矩阵, [bs, C_in, height, width] -> [bs, C_in*kernel_h*kernel_w, height_out*width_out]
    2) 展平 filter [C_out, C_in, kernel_h, kernel_w] -> [C_out, C_in*kernel_h*kernel_w] 其实这步展平操作, 只需要改变 shape 就可以了, 并不需要改变元素的排列顺序
    3) 将 filter 矩阵乘 col矩阵得到 ouput [C_out, height_out*width_out]

    在 CUDA 实现的时候, 线程组织分配的方式为 每个 thread 处理一个 kernel_h*kernel_w(一个通道的 col 中的一列), 一个 block 分配 256 个线程。
    那么 block 的分配方式就为  dim3 grid(C_in*height_out*width_out/blockDim.x, bs);
*/


#define BLOCK_SIZE 256

// 实现 [bs, C_in, H, W] -> [bs, C_in*kernel_h*kernel_w, with_col*height_col]
__global__ void im2col_h(const int n, const float *data_im, const int height,
                         const int width, const int kernel_h,
                         const int kernel_w, const int pad_h, const int pad_w,
                         const int stride_h, const int stride_w,
                         const int height_col, const int width_col,
                         float *data_col, int im_stride, int col_stride)
{
    // 每个 batch 中要处理的列的索引
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n)  // n 是一个 batch 需要的 列的个数, 超边界的都去除
    {
        const int batch_idx = blockIdx.y;    // 第几个 batch
        data_im += batch_idx * im_stride;    // data_im  先找到对应的 batch
        data_col += batch_idx * col_stride;  // data_col 先找到对应的 batch

        // 因为线程处理都是 data_col 中 batch0第0行第0列, 第0行第1列, 第0行, 第2列,....
        const int h_index = index / width_col;
        const int h_col = h_index % height_col;        // 对应输出的行数
        const int w_col = index % width_col;           // 对应输出的列数

        const int c_im = h_index / height_col;         // 在 im 哪个通道中
        const int c_col = c_im * kernel_h * kernel_w;  // 临时变量, 没有实际意义

        // 卷积核中左上角元素相对于 im 矩阵 (0,0) 元素的偏移, 这里是关键, 如何映射过去的
        const int h_offset = h_col * stride_h - pad_h;
        const int w_offset = w_col * stride_w - pad_w;

        // 依据当前 index 在 block 中的对应位置来进行索引
        float *data_col_ptr = data_col;
        // 在 col 矩阵中每个通道之间数据的跨度是：
        //     c_im * kernel_h * kernel_w * height_col * width_col
        // 前文所说 (h_col, w_col)的卷积对应的index是 h_col* width_col + w_col
        // 所以该线程处理的col矩阵在某一通道中的列为data_col_ptr即为下式
        data_col_ptr += c_col * height_col * width_col + h_col * width_col + w_col;

        const float *data_im_ptr = data_im;
        // 在im矩阵中不同通道的数据跨度是 c_im * height * width
        // 该卷积核(h_col, w_col)对应的im矩阵中第一个元素的线性索引是：
        //              h_offest * width + w_offset
        // 所以该卷积核对应的 im矩阵 中第一个数据索引为下式
        data_im_ptr += c_im * height * width + h_offset * width + w_offset;  // 找到在原始图像中的位置

        // 进行数据转换，将这个卷积核 kernel_h * kernel_w 中所对应的 im 数据存储到 data_col_ptr 中
        for (int i = 0; i < kernel_h; ++i)
        {
            for (int j = 0; j < kernel_w; ++j)
            {
                int h_im = h_offset + i;
                int w_im = w_offset + j; // (h_im, w_im)相对于卷积核左上角元素的偏移
                *data_col_ptr =
                    (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
                        ? data_im_ptr[i * width + j]
                        : 0;
                // data_im_ptr已经是相对于im矩阵中(0, 0)的偏移。(i,j)在im矩阵中的线性索引是 i * width + j
                data_col_ptr += height_col * width_col;  // 下一列
            }
        }
    }
}

// 实现 im2col 的卷积运算
// 其中 data_im 是原始 im 矩阵, data_col 是中间的 col 矩阵, channels 是 C_in
void im2col(const float *data_im, const int batch_size, const int channels,
            const int height, const int width,
            const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
            const int stride_h, const int stride_w,
            float *data_col)
{
    // 计算得到卷积输出的特征图的形状大小
    int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int width_col  = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    int size = channels * height_col * width_col;  // 每个线程处理一个通道的 col 中的一列, 这个 data_col 中一个 batch 中的总列数

    int im_stride = channels * height * width;  // data_im 中一个 batch 中的元素的个数
    int col_stride = channels * kernel_h * kernel_w * height_col * width_col;  // data_col 中一个 batch 中元素的个数

    dim3 dim_grid((size+BLOCK_SIZE-1) / BLOCK_SIZE, batch_size);  // 相当于每个 block 处理 blockDim.x 那么多的列

    im2col_h<<<dim_grid, BLOCK_SIZE>>>(size, data_im, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h,
        stride_w, height_col, width_col, data_col, im_stride, col_stride);

    // 最后实现矩阵乘法计算 filter * data_col 得到最终卷积的结果
}