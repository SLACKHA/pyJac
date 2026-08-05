#ifndef GPU_MEMORY_CUH
#define GPU_MEMORY_CUH

#include "header.cuh"
#include "gpu_macros.cuh"

void initialize_gpu_memory(int, mechanism_memory**, mechanism_memory**);
size_t required_mechanism_size();
void free_gpu_memory(mechanism_memory**, mechanism_memory**);

#endif
