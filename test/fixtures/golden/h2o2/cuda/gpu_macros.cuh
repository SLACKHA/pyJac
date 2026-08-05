#ifndef GPU_MACROS_CUH
#define GPU_MACROS_CUH
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#define GRID_DIM (blockDim.x * gridDim.x)
#define T_ID (threadIdx.x + blockIdx.x * blockDim.x)
#define INDEX(i) (T_ID + (i) * GRID_DIM)

#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#endif
