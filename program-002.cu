/* Program to add 2 numbers in GPU
   nvcc program-002.cu
 */

#include <iostream>
#include <cuda_runtime.h>

__global__ void add (int a, int b, int *result) {
    *result = a + b;
}

int main() {
    int result;
    int *gpuMemory;
    int num1, num2;

    std::cout << "This is the demonstration of adding two numbers in GPU" << std::endl;

    if (cudaMalloc((void**)&gpuMemory, sizeof(int)) != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory" << std::endl;
	return -1;
    }

    std::cout << "Enter number-1 : ";
    std::cin >> num1;
    std::cout << "Enter number-2 : ";
    std::cin >> num2;

    add<<<1, 1>>>(num1, num2, gpuMemory);

    if (cudaMemcpy(&result, gpuMemory, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cerr << "Failed to copy memory from GPU to CPU" << std::endl;
	cudaFree(gpuMemory);
	return -1;
    }

    std::cout << num1 << " + " << num2 << " = " << result << std::endl;
    cudaFree(gpuMemory);
    return 0;
}
