#include <iostream>
#include <cuda_runtime.h>

int main()
{
    cudaDeviceProp prop;
    int count;

    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        std::cerr << "Failed to get the GPU device count" << std::endl;
        return -1;
    }
    std::cout << "Number of GPU Device : " << count << std::endl;

    for (int i = 0; i < count; i++) {
        if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) {
            std::cerr << "Failed to get the GPU device " << i << " properties" << std::endl;
            return -1;
        }
        std::cout << "*****************************************" << std::endl;
        std::cout << "GPU: " << i << std::endl;
        std::cout << "Name : " << prop.name << std::endl;
        std::cout << "Total memory : " << prop.totalGlobalMem << std::endl;
        std::cout << "Multiprocessor count : " << prop.multiProcessorCount << std::endl;
        std::cout << "Shared-mem per block : " << prop.sharedMemPerBlock << std::endl;
        std::cout << "Registers per block  : " << prop.regsPerBlock << std::endl;
        std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    }
    return 0;
}
