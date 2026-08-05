#ifndef SPARSE_HEAD
#define SPARSE_HEAD

#define N_A 81
#include "header.cuh"

__device__
void sparse_multiplier (const double *, const double *, double*);

#ifdef COMPILE_TESTING_METHODS
  int test_sparse_multiplier();
#endif

#endif
