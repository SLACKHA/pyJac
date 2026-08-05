#include <math.h>
#include "header.cuh"
#include "rates.cuh"

__device__ void get_rxn_pres_mod (const double T, const double pres, const double * __restrict__ C, double * __restrict__ pres_mod) {
  extern volatile __shared__ double shared_temp[];
  // third body variable declaration
  register double thd;

  // pressure dependence variable declarations
  register double k0;
  register double kinf;
  register double Pr;

  // troe variable declarations
  register double logFcent;
  register double A;
  register double B;

  register double logT = log(T);
  register double m = pres / (8.31446210e+03 * T);

  // reaction 0;
  shared_temp[threadIdx.x + 3 * blockDim.x] = C[INDEX(0)];
  shared_temp[threadIdx.x + 2 * blockDim.x] = C[INDEX(5)];
  shared_temp[threadIdx.x + 1 * blockDim.x] = C[INDEX(8)];
  pres_mod[INDEX(0)] = m + 1.4 * shared_temp[threadIdx.x + 3 * blockDim.x] + 14.4 * shared_temp[threadIdx.x + 2 * blockDim.x] - 0.17000000000000004 * shared_temp[threadIdx.x + 1 * blockDim.x];

  // reaction 1;
  pres_mod[INDEX(1)] = m + 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 2 * blockDim.x] - 0.30000000000000004 * shared_temp[threadIdx.x + 1 * blockDim.x];

  // reaction 5;
  shared_temp[threadIdx.x] = C[INDEX(3)];
  pres_mod[INDEX(2)] = m - 1.0 * shared_temp[threadIdx.x] - 1.0 * shared_temp[threadIdx.x + 2 * blockDim.x] - 1.0 * shared_temp[threadIdx.x + 1 * blockDim.x];

  // reaction 10;
  pres_mod[INDEX(3)] = m - 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] - 1.0 * shared_temp[threadIdx.x + 2 * blockDim.x] - 0.37 * shared_temp[threadIdx.x + 1 * blockDim.x];

  // reaction 13;
  pres_mod[INDEX(4)] = m - 0.27 * shared_temp[threadIdx.x + 3 * blockDim.x] + 2.65 * shared_temp[threadIdx.x + 2 * blockDim.x] - 0.62 * shared_temp[threadIdx.x + 1 * blockDim.x];

  // reaction 20;
  thd = m + 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 2 * blockDim.x] - 0.30000000000000004 * shared_temp[threadIdx.x + 1 * blockDim.x];
  k0 = exp(2.8463930238863654e+01 - 0.9 * logT - (-8.5547326026057669e+02 / T));
  kinf = exp(2.5027330930150580e+01 - 0.37 * logT);
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(2.65400000e-01 * exp(-T / 9.40000000e+01) + 7.34600000e-01 * exp(-T / 1.75600000e+03) + exp(-5.18200000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(5)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

} // end get_rxn_pres_mod

