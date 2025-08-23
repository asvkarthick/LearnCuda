#include <iostream>
#include <cuda_runtime.h>

__global__ void kernel()
{
    printf("blockDim.x : %d, blockDim.y : %d, gridDim.x : %d, gridDim.y : %d, tIdx.x : %d, tIdx.y : %d, bIdx.x : %d, bIdx.y : %d\n",
		    blockDim.x, blockDim.y, gridDim.x, gridDim.y, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
}

int main(int argc, char* argv[])
{
    int numXblocks = 1, numYblocks = 1, numXthreads = 1, numYthreads = 1;
    std::cout << "Start of CPU execution" << std::endl;
    if (argc >= 5) {
        numXblocks = atoi(argv[1]);
        numYblocks = atoi(argv[2]);
        numXthreads = atoi(argv[3]);
        numYthreads = atoi(argv[4]);
    } else {
        std::cout << "Usage: <exe> <numXblocks> <numYblocks> <numXthreads> <numYthreads>" << std::endl;
    }
    dim3 dimBlock(numXthreads, numYthreads);
    dim3 dimGrid(numXblocks, numYblocks);
    kernel<<<dimGrid, dimBlock>>>();
    cudaDeviceSynchronize();
    return 0;
}
