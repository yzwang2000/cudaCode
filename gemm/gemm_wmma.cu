// 调用 TensorCore API 实现矩阵乘法
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <iostream>

using namespace nvcuda;

constexpr int DEFAULT_BLOCK_DIM = 512;
constexpr int M_TILE = 16;
constexpr int K_TILE = 16;
constexpr int N_TILE = 16;
constexpr int warpSize = 32;

// 填充 0 对计算的结果是没有影响的
inline int padding(int x, int tile) {
    return x % tile ? (x / tile + 1) * tile : x;
}

// 第一个易错点是 A 和 B 均是 half* 的形式
__global__ void gemm_kernel(half *A, half *B, float *C, int M_PAD, int K_PAD, int N_PAD) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;  // 当前的 warp ID
    int mwarp = M_PAD / M_TILE;  // 竖直方向需要的 warp 数量
    int nwarp = N_PAD / N_TILE;  // 水平方向需要的 warp 数量

    if (idx >= (mwarp * nwarp)) return;  // 超出范围直接返回

    int nidx = idx % nwarp;
    int midx = idx / nwarp;

    // 第二个易错点, matrix_a 和 matrix_b 都是需要指定 row_major, 
    // 而 accumulator 是不需要指定的(因为这个只需要一个累加的操作, 不需要从全局内存中读取)
    wmma::fragment<wmma::matrix_a, M_TILE, N_TILE, K_TILE, half, wmma::row_major> afrag;
    wmma::fragment<wmma::matrix_b, M_TILE, N_TILE, K_TILE, half, wmma::row_major> bfrag;
    wmma::fragment<wmma::accumulator, M_TILE, N_TILE, K_TILE, float> abfrag;
    // 第三个易错点, abfrag 是不错需要指定 row_major, 但是需要填充为 0
    wmma::fill_fragment(abfrag, 0.0f);

    int niter = K_PAD / K_TILE;
    for (int k = 0; k < niter; ++k){
        wmma::load_matrix_sync(afrag, A + midx * M_TILE * K_PAD + k * K_TILE, K_PAD);
        wmma::load_matrix_sync(bfrag, B + k * K_TILE * N_PAD + nidx * N_TILE, N_PAD);
        wmma::mma_sync(abfrag, afrag, bfrag, abfrag);
    }

    float *cptr = C + midx * M_TILE * N_PAD + nidx * N_TILE;
    // 第三个易错点, 当将 abfrag 的结果存储到其他存储器时, 需要指定 mem_row_major
    wmma::store_matrix_sync(cptr, abfrag, N_PAD, wmma::mem_row_major);  // 修正参数
}

// A 是 M*K, B 是 K*N, C 是 M*N
void launch_gemm(const int M, const int K, const int N) {
    int M_PAD = padding(M, M_TILE);
    int K_PAD = padding(K, K_TILE);
    int N_PAD = padding(N, N_TILE);

    half *A, *B;
    float *C, *C_cpu;
    cudaMallocManaged(&A, sizeof(half) * M_PAD * K_PAD);
    cudaMallocManaged(&B, sizeof(half) * K_PAD * N_PAD);
    cudaMallocManaged(&C, sizeof(float) * M_PAD * N_PAD);

    // 数据初始化(对于查出的部分，应该是用 0 来填充)
    for (int i = 0; i < M_PAD * K_PAD; ++i) A[i] = __float2half(1.0f);
    for (int i = 0; i < K_PAD * N_PAD; ++i) B[i] = __float2half(1.0f);
    for (int i = 0; i < M_PAD * N_PAD; ++i) C[i] = 0.0f;

    int nwarps = (M_PAD / M_TILE) * (N_PAD / N_TILE);

    dim3 grid, block;
    if (nwarps * warpSize < DEFAULT_BLOCK_DIM) {
        grid = {1, 1, 1};
        block = {nwarps * warpSize, 1, 1};
    } else {
        grid = {(nwarps * warpSize + DEFAULT_BLOCK_DIM - 1) / DEFAULT_BLOCK_DIM, 1, 1};
        block = {DEFAULT_BLOCK_DIM, 1, 1};
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    gemm_kernel<<<grid, block>>>(A, B, C, M_PAD, K_PAD, N_PAD);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU Time: %f ms\n", milliseconds);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}

int main() {
    int sizes[] = {512, 1024, 2048};
    for (int size : sizes) {
        launch_gemm(size, size, size);
    }
    return 0;
}