#include <iostream>
#include <cuda_runtime.h>

#define N 3

// Program to multiply two square matrices
__global__ void gpuMultiply(float *src1, float *src2, float *dst, int width)
{
    int idx = threadIdx.x;
    int idy = threadIdx.y;

    float val = 0;
    for (int k = 0; k < width; k++)
    {
        val += src1[idy * width + k] * src2[k * width + idx];
    }
    dst[idy * width + idx] = val;
}

void cpuMultiply(float *src1, float *src2, float *dst, int width)
{
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < width; j++)
	{
            float val = 0;
	    for (int k = 0; k < width; k++)
	    {
                val += src1[i * width + k] * src2[k * width + j];
	    }
	    dst[i * width + j] = val;
	}
    }
}

void printMatrix(float *m, int width)
{
    std::cout << "*************************" << std::endl;
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < width; j++)
        {
            std::cout << m[i * width + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main(void)
{
    std::cout << "In main function" << std::endl;
    float cpu_a[N][N] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    float cpu_b[N][N] = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
    float cpu_c[N][N];
    float *gpu_a, *gpu_b, *gpu_c;

    if (cudaMalloc((void**)&gpu_a, N * N * sizeof(float)) != cudaSuccess) {
        std::cerr << "Failed to allocate gpu_a memory" << std::endl;
        return -1;
    }
    if (cudaMalloc((void**)&gpu_b, N * N * sizeof(float)) != cudaSuccess) {
        std::cerr << "Failed to allocate gpu_b memory" << std::endl;
        cudaFree(gpu_a);
        return -1;
    }
    if (cudaMalloc((void**)&gpu_c, N * N * sizeof(float)) != cudaSuccess) {
        std::cerr << "Failed to allocate gpu_c memory" << std::endl;
        cudaFree(gpu_a);
        cudaFree(gpu_b);
        return -1;
    }

    cpuMultiply((float*)cpu_a, (float*)cpu_b, (float*)cpu_c, N);
    printMatrix((float*)cpu_a, N);
    printMatrix((float*)cpu_b, N);
    printMatrix((float*)cpu_c, N);

    // Copy to GPU memory
    if (cudaMemcpy(gpu_a, cpu_a, N * N * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cerr << "Failed to copy cpu_a to gpu_a" << std::endl;
        return -1;
    }
    if (cudaMemcpy(gpu_b, cpu_b, N * N * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cerr << "Failed to copy cpu_b to gpu_b" << std::endl;
        return -1;
    }

    // Perform multiplication in the GPU
    dim3 x(N, N);
    gpuMultiply<<<1, x>>>(gpu_a, gpu_b, gpu_c, N);

    if (cudaMemcpy(cpu_c, gpu_c, N * N * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cerr << "Failed to copy gpu_c to cpu_c" << std::endl;
        return -1;
    }
    printMatrix((float*)cpu_c, N);

    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_c);
    return 0;
}
