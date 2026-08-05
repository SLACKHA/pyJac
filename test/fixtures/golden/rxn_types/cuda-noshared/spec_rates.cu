#include "header.cuh"
#include "rates.cuh"

__device__ void eval_spec_rates (const double * __restrict__ fwd_rates, const double * __restrict__ rev_rates, const double * __restrict__ pres_mod, double * __restrict__ sp_rates, double * __restrict__ dy_N) {
  //rxn 0
  //sp 1
  sp_rates[INDEX(1)] = -(fwd_rates[INDEX(0)] - rev_rates[INDEX(0)]);
  //sp 2
  sp_rates[INDEX(2)] = (fwd_rates[INDEX(0)] - rev_rates[INDEX(0)]);
  //sp 3
  sp_rates[INDEX(3)] = -(fwd_rates[INDEX(0)] - rev_rates[INDEX(0)]);
  //sp 4
  sp_rates[INDEX(4)] = (fwd_rates[INDEX(0)] - rev_rates[INDEX(0)]);

  //rxn 1
  //sp 0
  sp_rates[INDEX(0)] = -fwd_rates[INDEX(1)];
  //sp 1
  sp_rates[INDEX(1)] += fwd_rates[INDEX(1)];
  //sp 2
  sp_rates[INDEX(2)] -= fwd_rates[INDEX(1)];
  //sp 4
  sp_rates[INDEX(4)] += fwd_rates[INDEX(1)];

  //rxn 2
  //sp 0
  sp_rates[INDEX(0)] -= fwd_rates[INDEX(2)];
  //sp 1
  sp_rates[INDEX(1)] += fwd_rates[INDEX(2)];
  //sp 4
  sp_rates[INDEX(4)] -= fwd_rates[INDEX(2)];
  //sp 5
  sp_rates[INDEX(5)] = fwd_rates[INDEX(2)];

  //rxn 3
  //sp 0
  sp_rates[INDEX(0)] += fwd_rates[INDEX(3)];
  //sp 1
  sp_rates[INDEX(1)] -= fwd_rates[INDEX(3)];
  //sp 4
  sp_rates[INDEX(4)] += fwd_rates[INDEX(3)];
  //sp 5
  sp_rates[INDEX(5)] -= fwd_rates[INDEX(3)];

  //rxn 4
  //sp 1
  sp_rates[INDEX(1)] -= (fwd_rates[INDEX(4)] - rev_rates[INDEX(1)]) * pres_mod[INDEX(0)];
  //sp 2
  sp_rates[INDEX(2)] -= (fwd_rates[INDEX(4)] - rev_rates[INDEX(1)]) * pres_mod[INDEX(0)];
  //sp 4
  sp_rates[INDEX(4)] += (fwd_rates[INDEX(4)] - rev_rates[INDEX(1)]) * pres_mod[INDEX(0)];

  //rxn 5
  //sp 1
  sp_rates[INDEX(1)] -= (fwd_rates[INDEX(5)] - rev_rates[INDEX(2)]) * pres_mod[INDEX(1)];
  //sp 3
  sp_rates[INDEX(3)] -= (fwd_rates[INDEX(5)] - rev_rates[INDEX(2)]) * pres_mod[INDEX(1)];
  //sp 6
  sp_rates[INDEX(6)] = (fwd_rates[INDEX(5)] - rev_rates[INDEX(2)]) * pres_mod[INDEX(1)];

  //rxn 6
  //sp 4
  sp_rates[INDEX(4)] += 2.0 * (fwd_rates[INDEX(6)] - rev_rates[INDEX(3)]) * pres_mod[INDEX(2)];
  //sp 7
  sp_rates[INDEX(7)] = -(fwd_rates[INDEX(6)] - rev_rates[INDEX(3)]) * pres_mod[INDEX(2)];

  //rxn 7
  //sp 1
  sp_rates[INDEX(1)] -= (fwd_rates[INDEX(7)] - rev_rates[INDEX(4)]);
  //sp 4
  sp_rates[INDEX(4)] += 2.0 * (fwd_rates[INDEX(7)] - rev_rates[INDEX(4)]);
  //sp 6
  sp_rates[INDEX(6)] -= (fwd_rates[INDEX(7)] - rev_rates[INDEX(4)]);

  //rxn 8
  //sp 3
  sp_rates[INDEX(3)] += (fwd_rates[INDEX(8)] - rev_rates[INDEX(5)]);
  //sp 4
  sp_rates[INDEX(4)] -= (fwd_rates[INDEX(8)] - rev_rates[INDEX(5)]);
  //sp 5
  sp_rates[INDEX(5)] += (fwd_rates[INDEX(8)] - rev_rates[INDEX(5)]);
  //sp 6
  sp_rates[INDEX(6)] -= (fwd_rates[INDEX(8)] - rev_rates[INDEX(5)]);

  //rxn 9
  //sp 3
  sp_rates[INDEX(3)] += (fwd_rates[INDEX(9)] - rev_rates[INDEX(6)]);
  //sp 6
  sp_rates[INDEX(6)] -= 2.0 * (fwd_rates[INDEX(9)] - rev_rates[INDEX(6)]);
  //sp 7
  sp_rates[INDEX(7)] += (fwd_rates[INDEX(9)] - rev_rates[INDEX(6)]);

  //rxn 10
  //sp 3
  sp_rates[INDEX(3)] += (fwd_rates[INDEX(10)] - rev_rates[INDEX(7)]);
  //sp 6
  sp_rates[INDEX(6)] -= 2.0 * (fwd_rates[INDEX(10)] - rev_rates[INDEX(7)]);
  //sp 7
  sp_rates[INDEX(7)] += (fwd_rates[INDEX(10)] - rev_rates[INDEX(7)]);

  //sp 8
  (*dy_N) = 0.0;
} // end eval_spec_rates

