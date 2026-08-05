#ifndef DYDT_HEAD
#define DYDT_HEAD

#include "header.cuh"

__device__ void dydt (const double, const double, const double * __restrict__, double * __restrict__, const mechanism_memory * __restrict__);

#endif
