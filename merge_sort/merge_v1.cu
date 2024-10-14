#include <iostream>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256  // 每个block中的线程数

// CUDA 核函数, 并行合并两个有序数组
__global__ void parallelMerge(int *A, int m, int *B, int n, int *C) {
    int L = m + n;
    int totalThreads = gridDim.x * blockDim.x;  // 总线程数
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;  // 全局线程ID
    int chunk_size = (L + totalThreads - 1) / totalThreads;  // 每个线程处理的元素数量
    int s = threadId * chunk_size;  // 当前线程负责的输出数组起始位置
    int e = min(s + chunk_size, L);  // 当前线程负责的输出数组结束位置

    // 在 A 和 B 中寻找划分点 i 和 j
    int i_lower = max(0, s - n);
    int i_upper = min(s, m);
    int i, j;

    // 二分法查找 i 和 j
    while (i_lower <= i_upper) {
        i = (i_lower + i_upper) / 2;
        j = s - i;

        int A_i_1 = (i > 0) ? A[i - 1] : INT_MIN;
        int A_i = (i < m) ? A[i] : INT_MAX;
        int B_j_1 = (j > 0) ? B[j - 1] : INT_MIN;
        int B_j = (j < n) ? B[j] : INT_MAX;

        if (A_i_1 <= B_j && B_j_1 <= A_i) {
            break; // 找到合适的划分点
        } else if (A_i_1 > B_j) {
            i_upper = i - 1;
        } else {
            i_lower = i + 1;
        }
    }

    // 现在 i 和 j 是输入数组的起始位置
    int i_start = i;
    int j_start = s - i_start;

    // 计算输入区间的结束位置
    int i_end = min(m, i_start + (e - s));
    int j_end = min(n, j_start + (e - s));

    // 合并操作，确保只处理当前线程的输出区间 [s, e)
    int idx = s;
    while (i_start < m && j_start < n && idx < e) {
        if (A[i_start] <= B[j_start]) {
            C[idx++] = A[i_start++];
        } else {
            C[idx++] = B[j_start++];
        }
    }
    while (i_start < m && idx < e) {
        C[idx++] = A[i_start++];
    }
    while (j_start < n && idx < e) {
        C[idx++] = B[j_start++];
    }
}

// CPU 端的归并函数：归并两个有序数组
void merge(int* A, int m, int* B, int n, int* C) {
    int i = 0, j = 0, k = 0;

    while (i < m && j < n) {
        if (A[i] <= B[j]) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }

    // 复制剩余元素
    while (i < m) {
        C[k++] = A[i++];
    }
    while (j < n) {
        C[k++] = B[j++];
    }
}

// 递归归并排序函数
void mergeSort(int* arr, int n) {
    if (n <= 1) {
        return; // 已经是有序的
    }

    int mid = n / 2;
    int* left = new int[mid];
    int* right = new int[n - mid];

    // 分割数组
    for (int i = 0; i < mid; i++) {
        left[i] = arr[i];
    }
    for (int i = mid; i < n; i++) {
        right[i - mid] = arr[i];
    }

    // 递归排序
    mergeSort(left, mid);
    mergeSort(right, n - mid);

    // 归并排序
    merge(left, mid, right, n - mid, arr);

    // 释放内存
    delete[] left;
    delete[] right;
}

// 主程序
int main() {
    const int n = 100000;  // 数据量10万
    int* h_A = new int[n];  // 数组A
    int* h_B = new int[n];  // 数组B
    int* h_C = new int[2 * n];  // 用于存放结果的数组C

    // 初始化数据
    for (int i = 0; i < n; i++) {
        h_A[i] = rand() % (n * 2);  // 随机数填充数组A
        h_B[i] = rand() % (n * 2);  // 随机数填充数组B
    }

    // CPU 归并排序
    mergeSort(h_A, n);
    mergeSort(h_B, n);

    // 打印归并后的结果（可以去掉，避免大数据量下打印影响性能）
    std::cout << "Array A sorted (first 10 elements): ";
    for (int i = 0; i < 20; i++) {
        std::cout << h_A[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Array B sorted (first 10 elements): ";
    for (int i = 0; i < 20; i++) {
        std::cout << h_B[i] << " ";
    }
    std::cout << std::endl;

    // 分配设备内存
    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, n * sizeof(int));
    cudaMalloc((void**)&d_B, n * sizeof(int));
    cudaMalloc((void**)&d_C, 2 * n * sizeof(int));

    // 复制数据到设备
    cudaMemcpy(d_A, h_A, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * sizeof(int), cudaMemcpyHostToDevice);

    // 启动 CUDA 核函数进行并行归并
    int numBlocks = (2 * n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;  // 计算需要的块数
    parallelMerge<<<numBlocks, THREADS_PER_BLOCK>>>(d_A, n, d_B, n, d_C);

    // 复制结果回主机
    cudaMemcpy(h_C, d_C, 2 * n * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果（可以去掉，避免大数据量下打印影响性能）
    std::cout << "Merged array (first 10 elements): ";
    for (int i = 0; i < 30; i++) {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // 释放主机内存
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
