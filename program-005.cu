/* Program to add two arrays in GPU using multiple blocks
   nvcc program-005.cu
 */
#include <iostream>
#include <cuda_runtime.h>

// Number of elements to add is 10
#define N 10

__global__ void add(int* a, int* b, int* c) {
    int index = blockIdx.x;
    if (index < N) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    int a[N], b[N], c[N];
    int *gpuMem1, *gpuMem2, *gpuResult;

    // Allocate the GPU input and output memories
    if (cudaMalloc((void**)&gpuMem1, N * sizeof(int)) != cudaSuccess) {
        std::cerr << "Failed to allocate GPU mem1" << std::endl;
	return -1;
    }
    if (cudaMalloc((void**)&gpuMem2, N * sizeof(int)) != cudaSuccess) {
        std::cerr << "Failed to allocate GPU mem2" << std::endl;
	cudaFree(gpuMem1);
	return -1;
    }
    if (cudaMalloc((void**)&gpuResult, N * sizeof(int)) != cudaSuccess) {
        std::cerr << "Failed to allocate GPU result memory" << std::endl;
	cudaFree(gpuMem1);
	cudaFree(gpuMem2);
	return -1;
    }

    // Fill the input array
    for (int i = 0; i < N; i++) {
        a[i] = i + 1;
	b[i] = N + i + 1;
    }

    if (cudaMemcpy(gpuMem1, a, N * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cerr << "Failed to copy a to gpuMem1" << std::endl;
	goto exit;
    }
    if (cudaMemcpy(gpuMem2, b, N * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cerr << "Failed to copy b to gpuMem2" << std::endl;
	goto exit;
    }

    // Add the arrays in GPU
    add<<<N, 1>>>(gpuMem1, gpuMem2, gpuResult);

    // Copy the output from GPU memory to CPU memory
    if (cudaMemcpy(c, gpuResult, N * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cerr << "Failed to copy gpuResult to c" << std::endl;
	goto exit;
    }

    // Print the output
    for (int i = 0; i < N; i++) {
        std::cout << a[i] << " + " << b[i] << " = " << c[i] << std::endl;
    }

exit:
    cudaFree(gpuMem1);
    cudaFree(gpuMem2);
    cudaFree(gpuResult);
    
    return 0;
}
