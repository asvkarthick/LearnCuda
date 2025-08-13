/* Program to get the GPU Device properties
   nvcc program-003.cu
 */
#include <iostream>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    int count;

    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        std::cerr << "Failed to get the GPU device count" << std::endl;
	return -1;
    }

    std::cout << "Number of GPU devices : " << count << std::endl;
    for (int i = 0; i < count; i++) {
        if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) {
            std::cerr << "Failed to get the GPU device(" << i << ") properties" << std::endl;
	    return -1;
	}
	std::cout << "---------------------------------------------------" << std::endl;
	std::cout << "GPU: " << i << std::endl;
	std::cout << "Name : " << prop.name << std::endl;
	std::cout << "Total mem : " << prop.totalGlobalMem << std::endl;
	std::cout << "Multiprocessor count : " << prop.multiProcessorCount << std::endl;
	std::cout << "Shared mem per mp : " << prop.sharedMemPerBlock << std::endl;
	std::cout << "Registers per mp : " << prop.regsPerBlock << std::endl;
	std::cout << "Threads in warp : " << prop.warpSize << std::endl;
	std::cout << "Max threads per block : " << prop.maxThreadsPerBlock << std::endl;
	std::cout << "Max thread dimensions : ("
		<< prop.maxThreadsDim[0] << ","
		<< prop.maxThreadsDim[1] << ","
		<< prop.maxThreadsDim[2] << ")" << std::endl;
	std::cout << "Max grid dimensions : ("
		<< prop.maxGridSize[0] << ","
		<< prop.maxGridSize[1] << ","
		<< prop.maxGridSize[2] << ")" << std::endl;
    }

    return 0;
}
