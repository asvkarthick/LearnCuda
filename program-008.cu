/* Program to demonstrate multiple blocks and multiple threads
 * nvcc program-008.cu
 */
#include <iostream>
#include <cuda_runtime.h>

__global__ void kernel() {
    printf("Block idx : %4d Thread idx : %4d, Block Dim : %4d, Grid Dim : %4d\n", blockIdx.x, threadIdx.x, blockDim.x, gridDim.x);
}

int main(void) {
    kernel<<<4,4>>>();
    cudaDeviceSynchronize();
}
