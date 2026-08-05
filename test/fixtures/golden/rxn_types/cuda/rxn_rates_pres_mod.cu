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

  // sri variable declarations
  register double X;

  register double logT = log(T);
  register double m = pres / (8.31446210e+03 * T);

  // reaction 4;
  shared_temp[threadIdx.x + 3 * blockDim.x] = C[INDEX(0)];
  shared_temp[threadIdx.x + 2 * blockDim.x] = C[INDEX(5)];
  shared_temp[threadIdx.x + 1 * blockDim.x] = C[INDEX(8)];
  pres_mod[INDEX(0)] = m + 1.5 * shared_temp[threadIdx.x + 3 * blockDim.x] + 11.0 * shared_temp[threadIdx.x + 2 * blockDim.x] - 0.25 * shared_temp[threadIdx.x + 1 * blockDim.x];

  // reaction 5;
  thd = m + 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] + 13.0 * shared_temp[threadIdx.x + 2 * blockDim.x] - 0.32999999999999996 * shared_temp[threadIdx.x + 1 * blockDim.x];
  k0 = exp(3.1778590439387948e+01 - 1.4 * logT);
  kinf = exp(2.2355638720663009e+01 + 0.44 * logT);
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(5.00000000e-01 * exp(-T / 1.00000000e-30) + 5.00000000e-01 * exp(-T / 1.00000000e+30) + exp(-1.00000000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(1)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 6;
  thd = m + 1.5 * shared_temp[threadIdx.x + 3 * blockDim.x] + 11.0 * shared_temp[threadIdx.x + 2 * blockDim.x] - 0.36 * shared_temp[threadIdx.x + 1 * blockDim.x];
  k0 = exp(3.2420178138029655e+01 - (2.2896490201091907e+04 / T));
  kinf = exp(3.3318335397877441e+01 - (2.4370923526129252e+04 / T));
  Pr = k0 * thd / kinf;
  X = 1.0 / (1.0 + log10(fmax(Pr, 1.0e-300)) * log10(fmax(Pr, 1.0e-300)));
  pres_mod[INDEX(2)] = pow(0.54 * exp(-201.0 / T) + exp(-T / 1024.0), X) * Pr / (1.0 + Pr);

} // end get_rxn_pres_mod

