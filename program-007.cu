/* Program to add two arrays in GPU using multiple grids, multiple blocks and multiple threads
   nvcc program-007.cu
 */
#include <iostream>
#include <cuda_runtime.h>

// Number of elements to add is 20000
#define N 20000

__global__ void add(int* a, int* b, int* c) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < N) {
        c[index] = a[index] + b[index];
	index += blockDim.x * gridDim.x;
    }
}

int main() {
    int a[N], b[N], c[N];
    int *gpuMem1, *gpuMem2, *gpuResult;

    std::cout << "Program to demonstrate parallel programming in GPU with multiple grids, multiple blocks and multiple threads" << std::endl;

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
    // add<<<(N + 127)/128, 128>>>(gpuMem1, gpuMem2, gpuResult);
    add<<<128, 128>>>(gpuMem1, gpuMem2, gpuResult);

    // Copy the output from GPU memory to CPU memory
    if (cudaMemcpy(c, gpuResult, N * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cerr << "Failed to copy gpuResult to c" << std::endl;
	goto exit;
    }

    // Print the output
    for (int i = 0; i < N; i++) {
        std::cout << a[i] << " + " << b[i] << " = " << c[i] << std::endl;
	if (c[i] != a[i] + b[i]) {
            std::cerr << "Error in computation : " << a[i] << " + " << b[i] << " = " << c[i] << std::endl;
	    goto exit;
        }
    }

exit:
    cudaFree(gpuMem1);
    cudaFree(gpuMem2);
    cudaFree(gpuResult);
    
    return 0;
}
