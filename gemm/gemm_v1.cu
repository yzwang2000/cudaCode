#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// cal offset from row col and ld , in row-major matrix, ld is the width of the matrix
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// transfer float4
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}

// 每一次大迭代都是对输出矩阵 C 的所有元素进行一次逐元素累加和
// 每一次小迭代是是在大迭代中对输出矩阵 C 的分块的逐元素的累加和 一个大迭代有多个小迭代
// K: ldA
// N: ldB
template <
    const int BLOCK_SIZE_M,   // 对矩阵 A 行分块的高度
    const int BLOCK_SIZE_K,   // 对矩阵 A 的列和矩阵 B 的行这一公共维度进行分块 (维度/分块数决定了大迭代的次数)
    const int BLOCK_SIZE_N,   // 对矩阵 B 列分块的宽度
    const int THREAD_SIZE_Y,  // 对每个 block 负责的共享矩阵块的行分块的高度
    const int THREAD_SIZE_X,  // 对每个 block 负责的共享矩阵块的列分块的宽度
    const bool ENABLE_DOUBLE_BUFFER // whether enable double buffering or not
    > 
__global__ void Sgemm( 
    float * __restrict__ A,
    float * __restrict__ B,
    float * __restrict__ C, 
    const int M,
    const int N,
    const int K) {
    
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // the threads number in Block of X,Y
    const int THREAD_X_PER_BLOCK = BLOCK_SIZE_N / THREAD_SIZE_X;  // 一个 block 的横向线程个数
    const int THREAD_Y_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE_Y;  // 一个 block 的纵向线程个数
    const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;  // 一个 block 中的线程总数

    // thread id in cur Block
    const int tid = ty * THREAD_X_PER_BLOCK + tx;  // 当前线程在 block 中的 id

    // shared memory 一轮迭代中需要使用的数据, 存放从 global mem 中读取的数据
    __shared__ float As[2][BLOCK_SIZE_K][BLOCK_SIZE_M];  // 这种写法是要对从 A 中读入到 As 中的元素进行转置 (A 中是 M*K, As 中是 K*M)
    __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];

    // registers for C, 临时存储 C 的计算结果, 这个是在寄存器上, 是正确的, 因为正好是每个线程负责一个块的计算, 相当于迭代这么多次都是往这个上面累加
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};  // 这个最后直接赋值到输出矩阵 C 中

    // registers for A and B, 加载数据, 这里也是开辟双倍的空间为了预取, 把数据从共享内存读取到寄存器中
    float frag_a[2][THREAD_SIZE_Y];
    float frag_b[2][THREAD_SIZE_X];

    // 每个线程都负责将一部分数据从 global mem 搬到 shared mem
    // registers load global memory 实现 global memory -> register -> shared memory 的过程, 在计算每个线程要搬运多少次 a 和 b 的数据
    const int ldg_num_a = BLOCK_SIZE_M * BLOCK_SIZE_K / (THREAD_NUM_PER_BLOCK * 4);  // 每个线程搬运 A 的次数
    const int ldg_num_b = BLOCK_SIZE_K * BLOCK_SIZE_N / (THREAD_NUM_PER_BLOCK * 4);  // 每个线程搬运 B 的次数
    float ldg_a_reg[4*ldg_num_a];  // 每个线程搬运的数据总数, 这个是声明在寄存器上的
    float ldg_b_reg[4*ldg_num_b];

    // threads number in one row
    const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;  // 搬运 A 一行要使用多少个线程
    const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;  // 搬运 B 一行要使用多少个线程

    // 依据当前线程 tid 来分配每个线程搬运这个块开始的第几行, 第几列数据, 这个从一个线程所能负责的数据块开始的相对位置
    const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;  // 当前线程搬运的是第几行的数据
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;
    const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4;    // 当前线程搬运的是第列的数据
    const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;

    // 记录一个线程块的线程, 每次能够搬运 A 和 B 多少行的数据
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;  // 一个 block 的线程一次能够搬运 A 的行数总和
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;  // 一个 block 的线程一次能够搬运 B 的行数总合

    // 根据 bid 来确定当前的线程块要搬运的块的起始位置, by 决定当前搬运 A 的位置, bx 决定当前搬运 B 的位置
    A = &A[(BLOCK_SIZE_M * by) * K];  // 因为每次都是从一整行的最起始开始的
    B = &B[BLOCK_SIZE_N * bx];        // 因为每次都是从一整列的最起始开始的

    // 这里的 __syncthreads() 是对两个共享矩阵 A 和 B 都赋值完以后才使用的函数
    // transfer first tile from global mem to shared mem 第一轮先填充了数据
    // load A from global memory to shared memory, 实现了存取并对A进行了转置
    // 每个线程先把共享内存的数据读取到线程的寄存器中, 然后再将寄存器中的数据读取到共享内存中
    #pragma unroll
    for (int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {  // 已经能够确定了每个线程搬运的次数
        int ldg_index = i / A_TILE_ROW_STRIDE * 4;  // 根据搬运的次数来决定存放寄存器的位置
        FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
            A_TILE_ROW_START + i, // row
            A_TILE_COL, // col
            K)]);
        As[0][A_TILE_COL][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index];  // 因为每次搬运是四个元素
        As[0][A_TILE_COL+1][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+1];
        As[0][A_TILE_COL+2][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+2];
        As[0][A_TILE_COL+3][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+3];
    }

    // load B from global memory to shared memory, 这里这么简单是因为列数相同, 直接读取就可以了
    #pragma unroll
    for (int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
        FETCH_FLOAT4(Bs[0][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
                B_TILE_ROW_START + i, // row
                B_TILE_COL, // col
                N)]);
    }
    __syncthreads();

    // 每个线程负责把自己的那部分从共享内存搬运到寄存器中
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
        FETCH_FLOAT4(frag_a[0][thread_y]) = FETCH_FLOAT4(As[0][0][THREAD_SIZE_Y * ty + thread_y]);
    }

    // load B from shared memory to register
    #pragma unroll
    for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
        FETCH_FLOAT4(frag_b[0][thread_x]) = FETCH_FLOAT4(Bs[0][0][THREAD_SIZE_X * tx + thread_x]);
    }

    int write_stage_idx = 1;  // 为 1 时, 对 As[1] 空间进行写操作, 对 As[0] 空间进行读操作
    int tile_idx = 0;         // 当 tile_idx=K 时停止

    do{
        tile_idx += BLOCK_SIZE_K;  // 控制迭代次数, 先直接加到下一步
        // 如果还有下一个迭代, 则将下一个迭代的数据块(全局内存) 搬运到临时寄存器上, 注意 tile_idx 这个变量
        // 预取到寄存器中
        if(tile_idx< K){
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
                int ldg_index = i / A_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
                    A_TILE_ROW_START + i, // row
                    A_TILE_COL + tile_idx, // col
                    K )]);
            }

            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
                int ldg_index = i / B_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(ldg_b_reg[ldg_index]) = FETCH_FLOAT4(B[OFFSET(
                    tile_idx + B_TILE_ROW_START + i, // row
                    B_TILE_COL, // col
                    N )]);
            }
        }

        int load_stage_idx = write_stage_idx ^ 1;  // 与 write_stage_idx 对应, 两者保持二进制位相反 ^ 是按位异或的操作

        // 双缓冲的方式, 将下一轮小迭代提前写到寄存器中, 同时更新缓存的 C 的元素的结果
        #pragma unroll
        for(int j=0; j<BLOCK_SIZE_K-1; ++j){
            // 计算当前小迭代之前, 先将下一轮的数据写到 write SM 中
            // load next tile from shared mem to register 
            // load A from shared memory to register
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
                FETCH_FLOAT4(frag_a[(j+1)%2][thread_y]) = FETCH_FLOAT4(As[load_stage_idx][j+1][THREAD_SIZE_Y * ty + thread_y]);
            }

            // load B from shared memory to register
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
                FETCH_FLOAT4(frag_b[(j+1)%2][thread_x]) = FETCH_FLOAT4(Bs[load_stage_idx][j+1][THREAD_SIZE_X * tx + thread_x]);
            }

            // compute C THREAD_SIZE_X x THREAD_SIZE_Y
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
                #pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                    accum[thread_y][thread_x] += frag_a[j%2][thread_y] * frag_b[j%2][thread_x];
                }
            }
        }

        // 临时寄存器上的数据搬运到共享内存中
        if(tile_idx < K){
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
                int ldg_index = i / A_TILE_ROW_STRIDE * 4;
                As[write_stage_idx][A_TILE_COL][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index];
                As[write_stage_idx][A_TILE_COL+1][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+1];
                As[write_stage_idx][A_TILE_COL+2][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+2];
                As[write_stage_idx][A_TILE_COL+3][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+3];
            }
            // load B from global memory to shared memory
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
                int ldg_index = i / B_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(Bs[write_stage_idx][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(ldg_b_reg[ldg_index]);
            }
            // use double buffer, only need one sync
            __syncthreads();
            // switch
            write_stage_idx ^= 1;
        }

        // load first tile from shared mem to register of next iter 完成寄存器的预取, 并将最后一个小迭代完成
        // load A from shared memory to register
        #pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
            FETCH_FLOAT4(frag_a[0][thread_y]) = FETCH_FLOAT4(As[load_stage_idx^1][0][THREAD_SIZE_Y * ty + thread_y]);
        }

        // load B from shared memory to register
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
            FETCH_FLOAT4(frag_b[0][thread_x]) = FETCH_FLOAT4(Bs[load_stage_idx^1][0][THREAD_SIZE_X * tx + thread_x]);
        }

        //compute last tile mma THREAD_SIZE_X x THREAD_SIZE_Y
        #pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                accum[thread_y][thread_x] += frag_a[1][thread_y] * frag_b[1][thread_x];
            }
        }
    }while(tile_idx< K);
    

    // 每个线程的寄存器中都保存了小块的结果, 迭代完大循环和小循环后, 将结果写回到全局内存中
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x+=4) {
            FETCH_FLOAT4(C[OFFSET(
                BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y,
                BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x,
                N)]) = FETCH_FLOAT4(accum[thread_y][thread_x]);
        }
    }
}

int main(int argc, char** argv) {
    // if (argc != 4) {
    //     printf("usage: ./main [M] [K] [N]\n");
    //     exit(0);
    // }
    // size_t M = atoi(argv[1]);
    // size_t K = atoi(argv[2]);
    // size_t N = atoi(argv[3]);

    // 矩阵 A 的大小为 M*K, 矩阵 B 的大小为 K*N, 矩阵 C 的大小为 M*N
    size_t M = 1024;
    size_t K = 256;
    size_t N = 1024;

    assert( M%8 == 0); 
    assert( N%8 == 0); 
    assert( K%8 == 0);

    // 在 Host 上分配空间
    size_t bytes_A = sizeof(float) * M * K;
    size_t bytes_B = sizeof(float) * K * N;
    size_t bytes_C = sizeof(float) * M * N;
    float* h_A = (float*)malloc(bytes_A);
    float* h_B = (float*)malloc(bytes_B);
    float* h_C = (float*)malloc(bytes_C);
    float* h_C1 = (float*)malloc(bytes_C);

    // 在 GPU 上分配空间
    float* d_A;
    float* d_B;
    float* d_C;

    checkCudaErrors(cudaMalloc(&d_A, bytes_A));
    checkCudaErrors(cudaMalloc(&d_B, bytes_B));
    checkCudaErrors(cudaMalloc(&d_C, bytes_C));
    double msecPerMatrixMul[2] = {0, 0};  // 记录 MyGEMM 与 cublas 所需要花费的时间 (msec)
    double gigaFlops[2] = {0, 0};  // 记录运算的 FLOPs (G)
    double flopsPerMatrixMul = 2.0 * M * N * K;

    // 这里的两步分块起始理解成对输出的分块更合适些
    const int BLOCK_SIZE_M = 128;  // A 矩阵的行分块的每块大小
    const int BLOCK_SIZE_K = 8;    // A 和 B 矩阵的公共维度分块每块的大小
    const int BLOCK_SIZE_N = 128;  // B 矩阵的列分块的每块大小
    const int THREAD_SIZE_X = 8;   // 对分块的数据再分块的行每块的大小 (X 和 Y　凑一起就是每个线程负责的大小)
    const int THREAD_SIZE_Y = 8;   // 对分块的数据再分块的列每块的大小
    const bool ENABLE_DOUBLE_BUFFER = false;

    // generate A
    for( int i = 0; i < M * K; i++ ){
        h_A[i] = i / 13;
    }

    // generate B
    for( int i = 0; i < K * N; i++ ) {
        h_B[i] = i % 13;
    }

    checkCudaErrors(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msecTotal = 0;
    int nIter = 1000;

    checkCudaErrors(cudaMemcpy(d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(start));
    
    for (int run = 0 ; run < nIter; run ++ ) {
        dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);  // 每个方向的块数设置网格的线程模型 bx 是横向是第几个块 by 是纵向是第几个块
        dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);  // 再分块时每个方向的块数设置线程块的线程模型 tx 是横向的第几个块 ty 是纵向的第几个块
        Sgemm<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X, ENABLE_DOUBLE_BUFFER> 
        <<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    }

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    checkCudaErrors(cudaMemcpy( h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    msecPerMatrixMul[0] = msecTotal / nIter;
    gigaFlops[0] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[0] / 1000.0f);
    printf( "My gemm Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        gigaFlops[0],
        msecPerMatrixMul[0],
        flopsPerMatrixMul);

    // cublas
    cublasHandle_t blas_handle;  
    cublasCreate(&blas_handle);
    float alpha = 1.0;
    float beta = 0;
    checkCudaErrors(cudaMemcpy( d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0 ; run < nIter; run ++ ) {
        cublasSgemm (blas_handle, CUBLAS_OP_T, CUBLAS_OP_T, 
            M, N, K, &alpha, 
            d_A, K, d_B, N, &beta, d_C, N
        );
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    checkCudaErrors(cudaMemcpy(h_C1, d_C, bytes_C, cudaMemcpyDeviceToHost));

    msecPerMatrixMul[1] = msecTotal / nIter;
    gigaFlops[1] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[1] / 1000.0f);
    printf( "CuBlas Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        gigaFlops[1],
        msecPerMatrixMul[1],
        flopsPerMatrixMul);

    cublasDestroy(blas_handle); 
    
    double eps = 1.e-6;  // machine zero
    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        int row = i / N;
        int col = i % N;
        double abs_err = fabs(h_C[i] - h_C1[col * M + row]);
        double dot_length = M;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err / abs_val / dot_length;
        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                    i, h_C[i], h_C1[col * M + row], eps);
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
    printf("ratio= %f\n", gigaFlops[0] / gigaFlops[1]);
    
    // Free Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C1);
}