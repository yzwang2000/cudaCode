#include <cuda.h>
#include <cuda_fp16.h>
#include <iostream>
#include <limits>

constexpr int batch_size = 16;
constexpr int num_head = 12;
constexpr int seq_len = 64;
constexpr int hidden_dim = 64;

// Q, K, V, O 均为 (bs, nh, seq_len, d)
__global__ void kernel_flashAttention(float* Q, float* K, float* V, float *O, float* l, float *m, int tile_size, int niter){
    int tid = threadIdx.x;
    int bidx = blockIdx.x; int bidy = blockIdx.y;  // nhead 和 batchsize

    // 寻找当前的 block 要处理到哪里的数据了
    int qkv_offset = bidy * num_head * seq_len * hidden_dim + bidx * seq_len * hidden_dim;
    int lm_offset = bidy * num_head * seq_len + bidx * seq_len;

    extern __shared__ float sms[];
    float* Qi = sms;
    float* Ki = &sms[tile_size*hidden_dim];
    float* Vi = &sms[2*tile_size*hidden_dim];
    float* Si = &sms[3*tile_size*hidden_dim];

    for(int i=0; i<niter; ++i)  // 外圈用来循环 K 和 V
    {
        for(int k=0; k<hidden_dim; ++k)
        {
            // 依据 qkv_offset 将 K, V 全局内存中的数据读取到 Ki, Vi 中
        }
        __syncthreads();

        for(int j=0; j<niter; ++j)  // 内层用来循环 Q
        {
            for(int k=0; k<hidden_dim; ++k)
            {
                // 依据 qkv_offset 将 Q 全局内存中的数据读取到 Qi 中
            }

            // 取出 m 和 l 每一行的最大值 
            int m_pre = m[lm_offset+j*tile_size+tid], l_pre = l[lm_offset+j*tile_size+tid];

            int m_row = -std::numeric_limits<float>::max();
            for(int y=0; y<tile_size; ++y)  // 每行都要与任何一列做点乘操作
            {
                float sumValue {0.f};
                for(int x=0; x<hidden_dim; ++x)
                {
                    // 得到 sumValue
                }
                // 将 Sum 写入到 S 中
                m_row = fmaxf(sumValue, m_row);  // 获得每行的最大值
            }

            float l_row = 0.f;
            for(int y=0; y<tile_size; ++y)  // 得到每行的EXP规约和
            {

            }

            // 得到新的 m_row 和 l_row
            float m_row_new = fmaxf(m_pre, m_row);
            float l_row_new = expf(m_pre-m_row_new) * l_pre + expf(m_row-m_row_new) * l_row;

            // 计算 S*V, 但是需要更新O的值
            for(int y=0; y<hidden_dim; ++y)
            {
                float pv {0.f};
                for(int x=0; x<tile_size; ++x) 
                {
                    // 得到 pv 的值
                }
                // 将原始的值更新，并加上如今的值
            }

            // 更新当前最大值和现如今的最大值
            m[lm_offset+j*tile_size+tid]=m_row_new, l[lm_offset+j*tile_size+tid]=l_row_new;
        }
        __syncthreads();
    }
}

cudaError_t launch_flashAttention(float* Q, float *K, float* V, float* O){
    int tile_size = 32;  // 设置滑块大小
    int niter = seq_len / tile_size;  // 迭代多少次

    // 每个 block 处理一个矩阵块
    dim3 grid(num_head, batch_size);
    dim3 block(tile_size);
    float * m;  // bs * nh * seq_len, 记录每一行的最大值
    float * l;  // bs * nh * seq_len, 记录每一行的前缀和

    int sms = 3*tile_size*hidden_dim + tile_size*tile_size;
    
    kernel_flashAttention<<<grid, block, sms*sizeof(float)>>>(Q, K, V, O, l, m, tile_size, niter);
}

// batch_size, num_head, seq_len, hidden_dim
dim3 grid(batch_size, num_head);
dim3 block(32);

int main(){
    return 0;
}