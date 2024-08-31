#include<cuda.h>

// NV 倡导的一些精度 fp32 TF32 FP16 BF16
// type    sign    exponent    mantissa   
// fp32     1         8          23 (IEEE754)
// tf32     1         8          10
// fp16     1         5          10
// bf16     1         8          7

// fp32 -> bf16
// unsigned int 转换为 unsigned short 时候会截断高位而保留低位
// 这里使用 union 节省了空间, 而且非常自然的让其公用一个空间 
unsigned short float32_to_bfloat16(float value){
    union {
        unsigned int u;
        float f;
    } tmp;
    tmp.f = value;
    return tmp.u >> 16;
}

// bf16 -> fp32 
float bfloat16_to_float32(unsigned short value){
    union
    {
        unsigned int u;
        float f;
    } tmp;

    tmp.u = value;
    tmp.u <<= 16;
    return tmp.f;
}

// fp32 -> fp16
unsigned short float32_to_fp16(float value){
    union
    {
        unsigned int u;
        float f;
    }tmp;
    
    tmp.f = value;
    tmp.u = ((tmp.u & 0x80000000) >> 16) | ((tmp.u & 0x7fc00000-(unsigned int)127+(unsigned int)15)>>13) | ((tmp.u & 0x007fffff)>>13);
    return tmp.u;
}

// fp16 -> fp32
float fp16_to_float32(unsigned short value)
{
    union
    {
        unsigned int u;
        float f;
    } tmp;
    
    tmp.u = value;
    tmp.u = ((tmp.u & 0x8000) << 16) | (((tmp.u & 0x7c00)-(unsigned short)15+(unsigned short)127) << 13) | ((tmp.u & 0x03ff) << 13);
    return tmp.f;
}

// 这个转换过程中要注意一个基础知识点。
// 对于整数类型来说, 高位转为低位时, 1) 高位截断: 转换过程中, 只保留低位部分, 高位部分会被舍弃
// 2) 类型解释: 保留的低位部分将按照目标类型的表示方式进行解释。
// int int_val = 0x12345678; // 305419896 in decimal
// short short_val = static_cast<short>(int_val); // Expected to keep lower 16 bits: 0x5678 (22136 in decimal)

// 对于整数类型来说, 低位转为高位时, 
// 1) 低位保留：如果—>左侧是无符号数(即被转类型), 高位用0填充。
// 2) 高位填充部分由被转类型是有符号还是无符号决定：如果->左侧是有符号类型(即被转类型)，高位用符号位填充(正数填充0, 负数填充1)

int main(){
    
    return 0;
}