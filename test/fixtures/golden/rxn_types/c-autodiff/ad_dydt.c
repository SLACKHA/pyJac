#include "header.h"
#include "ad_chem_utils.h"
#include "ad_rates.h"

#include "adept.h"
using adept::adouble;
#if defined(CONP)

void dydt (const double t, const adouble& pres, const adouble * __restrict__ y, adouble * __restrict__ dy) {

  // species molar concentrations
  adouble conc[9];
  adouble y_N;
  adouble mw_avg;
  adouble rho;
  eval_conc (y[0], pres, &y[1], &y_N, &mw_avg, &rho, conc);

  // local arrays holding reaction rates
  adouble fwd_rates[11];
  adouble rev_rates[8];
  eval_rxn_rates (y[0], pres, conc, fwd_rates, rev_rates);

  // get pressure modifications to reaction rates
  adouble pres_mod[3];
  get_rxn_pres_mod (y[0], pres, conc, pres_mod);

  // evaluate species molar net production rates
  adouble dy_N;
  eval_spec_rates (fwd_rates, rev_rates, pres_mod, &dy[1], &dy_N);
  // local array holding constant pressure specific heat
  adouble cp[9];
  eval_cp (y[0], cp);

  // constant pressure mass-average specific heat
  adouble cp_avg = (cp[0] * y[1]) + (cp[1] * y[2]) + (cp[2] * y[3]) + (cp[3] * y[4])
              + (cp[4] * y[5]) + (cp[5] * y[6]) + (cp[6] * y[7]) + (cp[7] * y[8]) + (cp[8] * y_N);

  // local array for species enthalpies
  adouble h[9];
  eval_h(y[0], h);
  // rate of change of temperature
  dy[0] = (-1.0 / (rho * cp_avg)) * ((dy[1] * h[0] * 2.0158800000000001e+00)
        + (dy[2] * h[1] * 1.0079400000000001e+00) + (dy[3] * h[2] * 1.5999400000000000e+01)
        + (dy[4] * h[3] * 3.1998799999999999e+01) + (dy[5] * h[4] * 1.7007339999999999e+01)
        + (dy[6] * h[5] * 1.8015280000000001e+01) + (dy[7] * h[6] * 3.3006740000000001e+01)
        + (dy[8] * h[7] * 3.4014679999999998e+01));

  // calculate rate of change of species mass fractions
  dy[1] *= (2.0158800000000001e+00 / rho);
  dy[2] *= (1.0079400000000001e+00 / rho);
  dy[3] *= (1.5999400000000000e+01 / rho);
  dy[4] *= (3.1998799999999999e+01 / rho);
  dy[5] *= (1.7007339999999999e+01 / rho);
  dy[6] *= (1.8015280000000001e+01 / rho);
  dy[7] *= (3.3006740000000001e+01 / rho);
  dy[8] *= (3.4014679999999998e+01 / rho);

} // end dydt

#elif defined(CONV)

void dydt (const double t, const adouble rho, const adouble * __restrict__ y, adouble * __restrict__ dy) {

  // species molar concentrations
  adouble conc[9];
  adouble y_N;
  adouble mw_avg;
  adouble pres;
  eval_conc_rho (y[0]rho, &y[1], &y_N, &mw_avg, &pres, conc);

  // local arrays holding reaction rates
  adouble fwd_rates[11];
  adouble rev_rates[8];
  eval_rxn_rates (y[0], pres, conc, fwd_rates, rev_rates);

  // get pressure modifications to reaction rates
  adouble pres_mod[3];
  get_rxn_pres_mod (y[0], pres, conc, pres_mod);

  // evaluate species molar net production rates
  adouble dy_N;  eval_spec_rates (fwd_rates, rev_rates, pres_mod, &dy[1], &dy_N);

  adouble cv[9];
  eval_cv(y[0], cv);

  // constant volume mass-average specific heat
  adouble cv_avg = (cv[0] * y[1]) + (cv[1] * y[2]) + (cv[2] * y[3]) + (cv[3] * y[4])
              + (cv[4] * y[5]) + (cv[5] * y[6]) + (cv[6] * y[7]) + (cv[7] * y[8])(cv[8] * y_N);

  // local array for species internal energies
  adouble u[9];
  eval_u (y[0], u);

  // rate of change of temperature
  dy[0] = (-1.0 / (rho * cv_avg)) * ((dy[1] * u[0] * 2.0158800000000001e+00)
        + (dy[2] * u[1] * 1.0079400000000001e+00) + (dy[3] * u[2] * 1.5999400000000000e+01)
        + (dy[4] * u[3] * 3.1998799999999999e+01) + (dy[5] * u[4] * 1.7007339999999999e+01)
        + (dy[6] * u[5] * 1.8015280000000001e+01) + (dy[7] * u[6] * 3.3006740000000001e+01)
        + (dy[8] * u[7] * 3.4014679999999998e+01));

  // calculate rate of change of species mass fractions
  dy[1] *= (2.0158800000000001e+00 / rho);
  dy[2] *= (1.0079400000000001e+00 / rho);
  dy[3] *= (1.5999400000000000e+01 / rho);
  dy[4] *= (3.1998799999999999e+01 / rho);
  dy[5] *= (1.7007339999999999e+01 / rho);
  dy[6] *= (1.8015280000000001e+01 / rho);
  dy[7] *= (3.3006740000000001e+01 / rho);
  dy[8] *= (3.4014679999999998e+01 / rho);

} // end dydt

#endif
