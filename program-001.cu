/* Hello World CUDA program
   nvcc program-001.cu -o program-001
 */
#include <iostream>
#include <cuda_runtime.h>

__global__ void kernel(void) {
    printf("Hello World from GPU!\n");
}

int main() {
    std::cout << "Hello World from CPU" << std::endl;
    kernel<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}
