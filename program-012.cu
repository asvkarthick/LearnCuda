/* Program to demonstrate Ray Tracing in GPU
 * nvcc program-012.cu
 */
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <cuda_runtime.h>

#define DIM 1024

#define rnd( x ) (x * rand() / RAND_MAX)
#define SPHERES 20

#define INF 2e10f

struct Sphere {
    float r,b,g;
    float radius;
    float x,y,z;

    __device__ float hit( float ox, float oy, float *n ) {
        float dx = ox - x;
        float dy = oy - y;

        if (dx*dx + dy*dy < radius*radius) {
            float dz = sqrtf( radius*radius - dx*dx - dy*dy );
            *n = dz / sqrtf( radius * radius );
            return dz + z;
        }
        return -INF;
    }
};

Sphere *s;

__global__ void kernel(unsigned char *ptr, Sphere *s) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    float ox = (x - DIM / 2);
    float oy = (y - DIM / 2);

    float r = 0, g = 0, b = 0;
    float maxz = -INF;

    for (int i = 0; i < SPHERES; i++) {
        float n;
        float t = s[i].hit(ox, oy, &n);

	if (t > maxz) {
            float fscale = n;
            r = s[i].r * fscale;
	    g = s[i].g * fscale;
	    b = s[i].b * fscale;
        }
    }

    ptr[offset * 3 + 0] = (int)(r * 255);
    ptr[offset * 3 + 1] = (int)(g * 255);
    ptr[offset * 3 + 2] = (int)(b * 255);
}

int main() {
    std::cout << "Welcome to CUDA Programming" << std::endl;
    cudaEvent_t start, stop;

    if (cudaEventCreate(&start) != cudaSuccess) {
        std::cerr << "Failed to create event start" << std::endl;
        return -1;
    }
    if (cudaEventCreate(&stop) != cudaSuccess) {
        std::cerr << "Failed to create event stop" << std::endl;
    }
    cudaEventRecord(start, 0);

    unsigned char *devBitmap;

    if (cudaMalloc((void**)&devBitmap, DIM * DIM * 3) != cudaSuccess) {
        std::cerr << "Failed to allocate bitmap GPU memory" << std::endl;
	return -1;
    }
    if (cudaMalloc((void**)&s, sizeof(Sphere) * SPHERES) != cudaSuccess) {
        std::cerr << "Failed to allocate sphers GPU memory" << std::endl;
	cudaFree(devBitmap);
	return -1;
    }

    Sphere *tmpS = (Sphere*) malloc(sizeof(Sphere) * SPHERES);
    for (int i = 0; i < SPHERES; i++) {
        tmpS[i].r = rnd(1.0f);
	tmpS[i].g = rnd(1.0f);
	tmpS[i].b = rnd(1.0f);
	tmpS[i].x = rnd(1000.f) - 500;
	tmpS[i].y = rnd(1000.f) - 500;
	tmpS[i].z = rnd(1000.f) - 500;
	tmpS[i].radius = rnd(100.0f) + 20;
    }

    if (cudaMemcpy(s, tmpS, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cerr << "Failed to copy spheres memory from CPU to GPU" << std::endl;
        cudaFree(devBitmap);
	cudaFree(s);
	free(tmpS);
	return -1;
    }

    dim3 blocks(DIM / 16, DIM / 16);
    dim3 threads(16, 16);
    kernel<<<blocks, threads>>>(devBitmap, s);

    unsigned char *bitmap = (unsigned char*) malloc(DIM * DIM * 3);
    if (cudaMemcpy(bitmap, devBitmap, DIM * DIM * 3, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cerr << "Failed to copy bitmap from GPU to CPU" << std::endl;
	cudaFree(devBitmap);
	cudaFree(s);
	free(tmpS);
	return -1;
    }

    std::ofstream fout("sphere.rgb", std::ios::binary);
    fout.write((char*)bitmap, DIM * DIM * 3);
    fout.close();

    cudaFree(s);
    cudaFree(devBitmap);
    free(tmpS);
    return 0;
}
