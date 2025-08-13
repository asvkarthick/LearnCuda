/* Program to demonstrate multi-dimensional blocks and multi-dimensional threads
 * nvcc program-009.cu
 */
#include <iostream>
#include <cuda_runtime.h>

__global__ void kernel() {
    printf("Block idx : %4d Thread idx : %4d, Block Dim : %4d, Grid Dim : %4d\n", blockIdx.x, threadIdx.x, blockDim.x, gridDim.x);
}

int main(void) {
    dim3 blocks(6, 6);
    dim3 threads(5, 5);
    kernel<<<blocks, threads>>>();
    cudaDeviceSynchronize();
}
