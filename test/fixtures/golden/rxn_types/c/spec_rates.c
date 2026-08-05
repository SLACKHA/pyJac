#include "header.h"
#include "rates.h"

void eval_spec_rates (const double * __restrict__ fwd_rates, const double * __restrict__ rev_rates, const double * __restrict__ pres_mod, double * __restrict__ sp_rates, double * __restrict__ dy_N) {
  //rxn 0
  //sp 1
  sp_rates[1] = -(fwd_rates[0] - rev_rates[0]);
  //sp 2
  sp_rates[2] = (fwd_rates[0] - rev_rates[0]);
  //sp 3
  sp_rates[3] = -(fwd_rates[0] - rev_rates[0]);
  //sp 4
  sp_rates[4] = (fwd_rates[0] - rev_rates[0]);

  //rxn 1
  //sp 0
  sp_rates[0] = -fwd_rates[1];
  //sp 1
  sp_rates[1] += fwd_rates[1];
  //sp 2
  sp_rates[2] -= fwd_rates[1];
  //sp 4
  sp_rates[4] += fwd_rates[1];

  //rxn 2
  //sp 0
  sp_rates[0] -= fwd_rates[2];
  //sp 1
  sp_rates[1] += fwd_rates[2];
  //sp 4
  sp_rates[4] -= fwd_rates[2];
  //sp 5
  sp_rates[5] = fwd_rates[2];

  //rxn 3
  //sp 0
  sp_rates[0] += fwd_rates[3];
  //sp 1
  sp_rates[1] -= fwd_rates[3];
  //sp 4
  sp_rates[4] += fwd_rates[3];
  //sp 5
  sp_rates[5] -= fwd_rates[3];

  //rxn 4
  //sp 1
  sp_rates[1] -= (fwd_rates[4] - rev_rates[1]) * pres_mod[0];
  //sp 2
  sp_rates[2] -= (fwd_rates[4] - rev_rates[1]) * pres_mod[0];
  //sp 4
  sp_rates[4] += (fwd_rates[4] - rev_rates[1]) * pres_mod[0];

  //rxn 5
  //sp 1
  sp_rates[1] -= (fwd_rates[5] - rev_rates[2]) * pres_mod[1];
  //sp 3
  sp_rates[3] -= (fwd_rates[5] - rev_rates[2]) * pres_mod[1];
  //sp 6
  sp_rates[6] = (fwd_rates[5] - rev_rates[2]) * pres_mod[1];

  //rxn 6
  //sp 4
  sp_rates[4] += 2.0 * (fwd_rates[6] - rev_rates[3]) * pres_mod[2];
  //sp 7
  sp_rates[7] = -(fwd_rates[6] - rev_rates[3]) * pres_mod[2];

  //rxn 7
  //sp 1
  sp_rates[1] -= (fwd_rates[7] - rev_rates[4]);
  //sp 4
  sp_rates[4] += 2.0 * (fwd_rates[7] - rev_rates[4]);
  //sp 6
  sp_rates[6] -= (fwd_rates[7] - rev_rates[4]);

  //rxn 8
  //sp 3
  sp_rates[3] += (fwd_rates[8] - rev_rates[5]);
  //sp 4
  sp_rates[4] -= (fwd_rates[8] - rev_rates[5]);
  //sp 5
  sp_rates[5] += (fwd_rates[8] - rev_rates[5]);
  //sp 6
  sp_rates[6] -= (fwd_rates[8] - rev_rates[5]);

  //rxn 9
  //sp 3
  sp_rates[3] += (fwd_rates[9] - rev_rates[6]);
  //sp 6
  sp_rates[6] -= 2.0 * (fwd_rates[9] - rev_rates[6]);
  //sp 7
  sp_rates[7] += (fwd_rates[9] - rev_rates[6]);

  //rxn 10
  //sp 3
  sp_rates[3] += (fwd_rates[10] - rev_rates[7]);
  //sp 6
  sp_rates[6] -= 2.0 * (fwd_rates[10] - rev_rates[7]);
  //sp 7
  sp_rates[7] += (fwd_rates[10] - rev_rates[7]);

  //sp 8
  (*dy_N) = 0.0;
} // end eval_spec_rates

