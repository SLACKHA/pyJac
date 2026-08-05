#ifndef JACOB_HEAD
#define JACOB_HEAD

#include "header.cuh"
#include "chem_utils.cuh"
#include "rates.cuh"
#include "gpu_memory.cuh"

__device__ void eval_jacob (const double, const double, const double * __restrict__, double * __restrict__, const mechanism_memory * __restrict__);

#endif
