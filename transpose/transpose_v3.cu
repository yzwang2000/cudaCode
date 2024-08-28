#include <stdio.h>
#include <stdlib.h>

#define TILE_DIM 32
#define BLOCK_SIZE 32
#define M 1000  // 输入矩阵的行
#define N 800  // 输入矩阵的列

// 普通的转置方法, 这里是每个线程对应一个数据。线程块的大小为 32*32, 处理的数据块的大小也为 32*32
// 其实这里是两个层面, block 层面的 bidx 和 bidy 是可以直接交换的。而 thread 层面的 x 和 y, 要看共享内存中的索引顺序。
__global__ void transpose(int* A, int* B)
{
    // 避免 bank 冲突, 通过填充的办法
    __shared__ int rafa[TILE_DIM][TILE_DIM + 1];
	
    // 每个元素对应一个线程
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    // 逐行的搬
    if (x < N && y < M)
    {
        int Aidx = y*N + x;
        rafa[threadIdx.y][threadIdx.x] = A[Aidx];
    }
    __syncthreads();
	
    int x2 = threadIdx.x + blockDim.y * blockIdx.y;
    int y2 = threadIdx.y + blockDim.x * blockIdx.x;
    if (x2 < M && y2 < N)
    {
        int Bidx = y2*M + x2;
        B[Bidx] = rafa[threadIdx.x][threadIdx.y];
    }
}

__managed__ int input_M[M*N];
int cpu_result[M*N];

// in-place matrix transpose, 其实只有一般的线程再干活(严谨来说是多一半)
__global__ void ip_transpose(int* data)
{
    __shared__ int tile_s[TILE_DIM][TILE_DIM+1];  // 存储原始的
    __shared__ int tile_d[TILE_DIM][TILE_DIM+1];  // 存储对称的

    int x = blockIdx.x * TILE_DIM + threadIdx.x;  // 处理的元素 x
    int y = blockIdx.y * TILE_DIM + threadIdx.y;  // 处理的元素 y

    // 处理对角线之下的线程块部分
    if (blockIdx.y > blockIdx.x) {  // 相当于是这块元素都在下面
        // 找到对称的线程块(注意这里的对称都是以线程块为单位的)
        int dx = blockIdx.y * TILE_DIM + threadIdx.x;  // 这里的 d 表示对称
        int dy = blockIdx.x * TILE_DIM + threadIdx.y;  // 这里的 d 表示对称
        if(x < N && y < M)  // 过滤下
        {
            tile_s[threadIdx.y][threadIdx.x] = data[y * N + x];
        }
        if(dx < N && dy < M)
        {
            tile_d[threadIdx.y][threadIdx.x] = data[dy * N + dx];
        }
        __syncthreads();

        // 开始互相交换
        if(dx < M && dy < N)
        {
            data[dy * M + dx] = tile_s[threadIdx.x][threadIdx.y];
        }
        if(x < M && y < N)
        {
            data[y * M + x] = tile_d[threadIdx.x][threadIdx.y];
        }
    }
    else if (blockIdx.y == blockIdx.x)  // 其实这款只需要实现一个 block 内数据的调换就可以了
    { 
        if(x < N && y < M)
        {
            tile_s[threadIdx.y][threadIdx.x] = data[y * N + x];
        }
        __syncthreads();
        if(x < N && y < M)
        {
            data[y * M + x] = tile_s[threadIdx.x][threadIdx.y];
        }
    }
}

void cpu_transpose(int* A, int* B)
{
    for (int j = 0; j < M; j++)
    {
        for (int i = 0; i < N; i++)
        {
            B[i * M + j] = A[j * N + i];
        }
    }
}

int main(int argc, char const *argv[])
{
    cudaEvent_t start, stop_gpu;
    cudaEventCreate(&start);
    cudaEventCreate(&stop_gpu);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            input_M[i * N + j] = rand() % 1000;
        }
    }
    cpu_transpose(input_M, cpu_result);

    cudaEventRecord(start);
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 每行对应多少个 block
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 每列对应多少个 block
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);  // 每个 block 是 32*32 个线程
    ip_transpose<<<dimGrid, dimBlock>>>(input_M);
    cudaDeviceSynchronize();
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    float elapsed_time_gpu;
    cudaEventElapsedTime(&elapsed_time_gpu, start, stop_gpu);
    printf("Time_GPU = %g ms.\n", elapsed_time_gpu);

    cudaEventDestroy(start);
    cudaEventDestroy(stop_gpu);

    int ok = 1;
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            if(fabs(input_M[i * N + j] - cpu_result[i * N + j]) > (1.0e-10))
            {
                ok = 0;
            }
        }
    }

    if(ok)
    {
        printf("Pass!!!\n");
    }
    else
    {
        printf("Error!!!\n");
    }

    return 0;
}