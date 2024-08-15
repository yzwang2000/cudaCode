#include "cuda_runtime.h"
#include <iostream>
#include <opencv2/opencv.hpp>

// OpenCV 中 cv::Mat 的 data 有坑
// 这个坑一直有 opencv 读入图像的矩阵格式为 [B H W C] Pytorch 读入图像的矩阵格式为 [B C H W]
// 整体上都是从左到右从上到小的存储顺序
// 对于单通道矩阵来说, 数据元素按照顺序存储
// 对于多通道矩阵来说, 数据元素按照通道存储
/* 这是一个 3*3 RGB 矩阵的存储方式
    [R0, G0, B0]
    [R1, G1, B1]
    [R2, G2, B2]
    [R3, G3, B3]
    [R4, G4, B4]
    [R5, G5, B5]
    [R6, G6, B6]
    [R7, G7, B7]
    [R8, G8, B8]
*/

// __device__ int srcIdxFun(int x, int y, int srcW, int z_id)
// { 
//     return (y * srcW + x) * 3 + (2 - z_id); 
// };

__global__ void BGR2RGB_bilinear_normalize_transpose_kernel(
    float *d_tar, uint8_t *d_src,
    const int tarW, const int tarH,
    const int srcW, const int srcH,
    float scale_w, float scale_h)
{
    // 取出各个索引
    uint x_id = threadIdx.x + blockIdx.x * blockDim.x;
    uint y_id = threadIdx.y + blockIdx.y * blockDim.y;
    uint z_id = threadIdx.z + blockIdx.z * blockDim.z;

    // 超出范围的线程都不要
    if (x_id >= tarW || y_id >= tarH || z_id > 3) return;

    // 避免重复计算, 得到在原图的坐标
    float x_scale = x_id * scale_w;
    float y_scale = y_id * scale_h;

    // 双线性插值的公式
    // 左上  右下角就是 x 和 y 都加 1
    int src_x1 = floor(static_cast<float>(x_scale));
    int src_y1 = floor(static_cast<float>(y_scale));

    // 这个线程对应目标矩阵中的元素位置
    // 这里这样写就是从原图 [H W C] 到 目标图 [C H W]
    int tar_id = z_id * tarW * tarH + y_id * tarW + x_id;

    // 计算在原图坐标的匿名函数
    auto srcIdxFun = [srcW, z_id](int x, int y) -> int { return (y * srcW + x) * 3 + (2 - z_id); };

    // 简化索引计算
    int src_id1 = srcIdxFun(src_x1, src_y1);         // 左上
    int src_id2 = srcIdxFun(src_x1, src_y1 + 1);     // 左下
    int src_id3 = srcIdxFun(src_x1 + 1, src_y1);     // 右上
    int src_id4 = srcIdxFun(src_x1 + 1, src_y1 + 1); // 右下

    // 将除法操作移到循环外部
    float tw = x_scale - src_x1;  // 宽度增加的大小
    float th = y_scale - src_y1;  // 高度增加的大小

    // fmaf 一个用于执行单个指令中的乘法和加法的内建函数
    // fmaf(a, b, c) 的计算结果等价于 (a * b) + c，但 fmaf 可以在硬件级别上以更高效的方式执行这两个操作
    d_tar[tar_id] = fmaf(d_src[src_id1], (1.0f - tw) * (1.0f - th),
                        fmaf(d_src[src_id2], (1.0f - tw) * th,
                        fmaf(d_src[src_id3], tw * (1.0f - th),
                             d_src[src_id4] * tw * th))) / 255.0f;
}


__global__ void BGR2RGB_bilinear_normalize_transpose_letterbox_kernel(
    float *d_tar, uint8_t *d_src,
    const int tarW, const int tarH,
    const int srcW, const int srcH,
    float scale_w, float scale_h)
{
    // 取出各个索引
    uint x_id = threadIdx.x + blockIdx.x * blockDim.x;
    uint y_id = threadIdx.y + blockIdx.y * blockDim.y;
    uint z_id = threadIdx.z + blockIdx.z * blockDim.z;

    // 超出范围的线程都不要
    if(x_id >= tarW || y_id >= tarH || z_id > 3) return;

    // 双线性插值的公式
    // 左上  右下角就是 x 和 y 都加 1
    int src_x1 = floor(static_cast<float>(x_id * scale_w));
    int src_y1 = floor(static_cast<float>(y_id * scale_h));

    // 当使用相同的 scale 的时候，会产生填充不足的情况，这时候使用 0 来补充
    if (src_x1 < 0 || src_y1 < 0 || src_x1 >= srcW || src_y1 >= srcH) return;

    // x 和 y 方向的偏移量
    int shift_x = (tarW - ceil(srcW / scale_w)) / 2;
    int shift_y = (tarH - ceil(srcH / scale_h)) / 2;

    // 这个线程对应目标矩阵中的元素位置
    // 这里这样写就是从原图 [H W C] 到目标图 [C H W]
    int tar_id = z_id * tarW * tarH + (y_id + shift_y) * tarW + (x_id + shift_x);

    // 计算在原图坐标的匿名函数
    auto srcIdxFun = [srcW, z_id](int x, int y)->int{return (y * srcW + x)*3 + (2 - z_id);};

    // 找到的原来矩阵中的位置
    int src_id1 = srcIdxFun(src_x1, src_y1);       // 左上
    int src_id2 = srcIdxFun(src_x1, src_y1+1);     // 左下
    int src_id3 = srcIdxFun(src_x1+1, src_y1);     // 右上
    int src_id4 = srcIdxFun(src_x1+1, src_y1+1);   // 右下

    // 计算对应的面积
    float tw = (x_id * scale_w) - src_x1;  // 宽度增加的大小
    float th = (y_id * scale_h) - src_y1;  // 高度增加的大小

    // fmaf 一个用于执行单个指令中的乘法和加法的内建函数
    // fmaf(a, b, c) 的计算结果等价于 (a * b) + c，但 fmaf 可以在硬件级别上以更高效的方式执行这两个操作
    d_tar[tar_id] = fmaf(d_src[src_id1], (1.0f - tw) * (1.0f - th),
                        fmaf(d_src[src_id2], (1.0f - tw) * th,
                        fmaf(d_src[src_id3], tw * (1.0f - th),
                             d_src[src_id4] * tw * th))) / 255.0f;
}


void resize_image(
    float *d_tar, uint8_t *d_src,
    const int tarW, const int tarH,
    const int srcW, const int srcH,
    int tatics)
{
    // 定义线程块
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(tarW/dimBlock.x+1, tarH/dimBlock.y+1, 3);   // 加 1 的目的是起始就是 1, 相当于是每个线程对应一个元素

    // 缩放的 scale
    float scale_h = static_cast<float>(srcH) / tarH;
    float scale_w = static_cast<float>(srcW) / tarW;
    // (尽量不让图片尺寸过大的规则)
    float scale = scale_w > scale_h ? scale_w : scale_h;

    switch (tatics)
    {
    case 0:
        BGR2RGB_bilinear_normalize_transpose_kernel<<<dimGrid, dimBlock>>>(d_tar, d_src, tarW, tarH, srcW, srcH, scale_w, scale_h);
        break;
    case 1:
        scale_h = scale; scale_w = scale;  // 保持相同的 scale 就不会产生形变
        BGR2RGB_bilinear_normalize_transpose_letterbox_kernel<<<dimGrid, dimBlock>>>(d_tar, d_src, tarW, tarH, srcW, srcH, scale_w, scale_h);
        break;
    default:
        break;
    }
}

// 依据比例进行缩放 (GPU 版本)
/*
    原图是 unsigned char [H W C] --> 目标图 float [C H W]
    1) BGR --> RGB     2) 双线性插值或者 letterbox 双线性插值   3) 归一化   4) [H W C] --> [C H W]
*/
void preprocess_gpu(cv::Mat &h_src, float *d_tar, const int tar_h, const int tar_w, int tactis){
    // 获取输入图像的长和宽, 注意 h_src.dims 为 2, 表示 h_src 是一个二维矩阵(即图像)
    const int src_height = h_src.rows;
    const int src_weight = h_src.cols;
    const int channels = 3;

    // 输入输出图像的像素个数
    size_t src_size = src_height * src_weight * channels;

    uint8_t * d_src{nullptr};  // 指向设备上的原图数据

    cudaMalloc((void**)&d_src, sizeof(unsigned char)*src_size);

    // 将数据拷贝到 device 上
    cudaMemcpy(d_src, h_src.data, sizeof(unsigned char)*src_size, cudaMemcpyHostToDevice);

    // // 在这个函数里根据策略执行核函数
    resize_image(d_tar, d_src, tar_w, tar_h, src_weight, src_height, tactis);

    // cpu 和 gpu 进行同步处理
    cudaDeviceSynchronize();

    // 释放资源 (设备上原始图像 主机上目标图像)
    cudaFree(d_src);
}

// 依据比例进行缩放 (CPU 版本)
cv::Mat preprocess_cpu(cv::Mat &src, const int tar_h, const int tar_w)
{
    cv::Mat tar_image;

    // bgr 到 rgb
    cv::cvtColor(src, src, cv::COLOR_BGR2RGB);
    // resize
    cv::resize(src, tar_image, cv::Size(tar_w, tar_h), 0, 0, cv::InterpolationFlags::INTER_LINEAR);

    return tar_image;
}