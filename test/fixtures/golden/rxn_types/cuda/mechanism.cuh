#ifndef MECHANISM_cuh
#define MECHANISM_cuh

#ifdef __GNUG__
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "launch_bounds.cuh"
#include "gpu_macros.cuh"
#endif

struct mechanism_memory {
  double * y;
  double * dy;
  double * conc;
  double * fwd_rates;
  double * rev_rates;
  double * spec_rates;
  double * cp;
  double * h;
  double * dBdT;
  double * jac;
  double * var;
  double * pres_mod;
  double * dot_prod;
};

//last_spec 8
/* Species Indexes
0  H2
1  H
2  O
3  O2
4  OH
5  H2O
6  HO2
7  H2O2
8  AR
*/

//Number of species
#define NSP 9
//Number of variables. NN = NSP + 1 (temperature)
#define NN 10
//Number of forward reactions
#define FWD_RATES 11
//Number of reversible reactions
#define REV_RATES 8
//Number of reactions with pressure modified rates
#define PRES_MOD_RATES 3

//Must be implemented by user on a per mechanism basis in mechanism.cu
void set_same_initial_conditions(int, double**, double**);

#if defined (RATES_TEST) || defined (PROFILER)
    void write_jacobian_and_rates_output(int NUM);
#endif
//apply masking of ICs for cache optimized mechanisms
void apply_mask(double*);
void apply_reverse_mask(double*);
#endif

