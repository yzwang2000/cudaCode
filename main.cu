#include<cuda.h>
#include <stdio.h>

__global__ void hello_world()
{
    printf("hello_world!\n");
}

int main(){
    dim3 grid(1, 1);
    dim3 block(10);
    hello_world<<<grid, block>>>();

    cudaDeviceSynchronize();

    return 0;
}