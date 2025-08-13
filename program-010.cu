/* Program to create a simple circle graphics using GPU
 * nvcc program-010.cu
 */
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

#define DIM 512

__global__ void kernel(unsigned char *ptr, int ticks) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float fx = x - DIM / 2;
    float fy = y - DIM / 2;
    float d  = sqrtf(fx * fx + fy * fy);

    unsigned char grey = (unsigned char) (128.0f + 127.0f *
		    cos(d / 10.0f - ticks / 7.0f) / (d / 10.0f + 1.0f));
    ptr[offset * 3 + 0] = grey;
    ptr[offset * 3 + 1] = grey;
    ptr[offset * 3 + 2] = grey;
}

int main() {
    dim3 blocks(DIM / 16, DIM / 16);
    dim3 threads(16, 16);

    unsigned char *gpuMemory;
    if (cudaMalloc((void**) &gpuMemory, DIM * DIM * 3) != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory" << std::endl;
	return -1;
    }

    unsigned char *cpuMemory = (unsigned char*) malloc(DIM * DIM * 3);
    if (cpuMemory == nullptr) {
        std::cerr << "Failed to allocate CPU memory" << std::endl;
	cudaFree(gpuMemory);
	return -1;
    }

    std::ofstream fout("frame.rgb", std::ios::binary);
    if (!fout) {
        std::cerr << "Failed to open output file" << std::endl;
	cudaFree(gpuMemory);
	free(cpuMemory);
	return -1;
    }

    kernel<<<blocks, threads>>>(gpuMemory, 0);

    if (cudaMemcpy(cpuMemory, gpuMemory, DIM * DIM * 3, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cerr << "Failed to copy memory from GPU to CPU" << std::endl;
    }

    fout.write((char*)cpuMemory, DIM * DIM * 3);

    fout.close();
    cudaFree(gpuMemory);
    free(cpuMemory);
}
