#include "header.cuh"
#include "chem_utils.cuh"
#include "rates.cuh"
#include "gpu_memory.cuh"

#if defined(CONP)

__device__ void dydt (const double t, const double pres, const double * __restrict__ y, double * __restrict__ dy, const mechanism_memory * __restrict__ d_mem) {

  // species molar concentrations
  double * __restrict__ conc = d_mem->conc;
  double y_N;
  double mw_avg;
  double rho;
  eval_conc (y[INDEX(0)], pres, &y[GRID_DIM], &y_N, &mw_avg, &rho, conc);

  double * __restrict__ fwd_rates = d_mem->fwd_rates;
  double * __restrict__ rev_rates = d_mem->rev_rates;
  eval_rxn_rates (y[INDEX(0)], pres, conc, fwd_rates, rev_rates);

  // get pressure modifications to reaction rates
  double * __restrict__ pres_mod = d_mem->pres_mod;
  get_rxn_pres_mod (y[INDEX(0)], pres, conc, pres_mod);

  double * __restrict__ spec_rates = d_mem->spec_rates;
  // evaluate species molar net production rates
  eval_spec_rates (fwd_rates, rev_rates, pres_mod, spec_rates, &spec_rates[INDEX(8)]);
  // local array holding constant pressure specific heat
  double * __restrict__ cp = d_mem->cp;
  eval_cp (y[INDEX(0)], cp);

  // constant pressure mass-average specific heat
  double cp_avg = (cp[INDEX(0)] * y[INDEX(1)]) + (cp[INDEX(1)] * y[INDEX(2)])
              + (cp[INDEX(2)] * y[INDEX(3)]) + (cp[INDEX(3)] * y[INDEX(4)])
              + (cp[INDEX(4)] * y[INDEX(5)]) + (cp[INDEX(5)] * y[INDEX(6)])
              + (cp[INDEX(6)] * y[INDEX(7)]) + (cp[INDEX(7)] * y[INDEX(8)]) + (cp[INDEX(8)] * y_N);

  // local array for species enthalpies
  double * __restrict__ h = d_mem->h;
  eval_h(y[INDEX(0)], h);
  // rate of change of temperature
  dy[INDEX(0)] = (-1.0 / (rho * cp_avg)) * ((spec_rates[INDEX(0)] * h[INDEX(0)] * 2.0158800000000001e+00)
        + (spec_rates[INDEX(1)] * h[INDEX(1)] * 1.0079400000000001e+00)
        + (spec_rates[INDEX(2)] * h[INDEX(2)] * 1.5999400000000000e+01)
        + (spec_rates[INDEX(3)] * h[INDEX(3)] * 3.1998799999999999e+01)
        + (spec_rates[INDEX(4)] * h[INDEX(4)] * 1.7007339999999999e+01)
        + (spec_rates[INDEX(5)] * h[INDEX(5)] * 1.8015280000000001e+01)
        + (spec_rates[INDEX(6)] * h[INDEX(6)] * 3.3006740000000001e+01)
        + (spec_rates[INDEX(7)] * h[INDEX(7)] * 3.4014679999999998e+01));

  // calculate rate of change of species mass fractions
  dy[INDEX(1)] = spec_rates[INDEX(0)] * (2.0158800000000001e+00 / rho);
  dy[INDEX(2)] = spec_rates[INDEX(1)] * (1.0079400000000001e+00 / rho);
  dy[INDEX(3)] = spec_rates[INDEX(2)] * (1.5999400000000000e+01 / rho);
  dy[INDEX(4)] = spec_rates[INDEX(3)] * (3.1998799999999999e+01 / rho);
  dy[INDEX(5)] = spec_rates[INDEX(4)] * (1.7007339999999999e+01 / rho);
  dy[INDEX(6)] = spec_rates[INDEX(5)] * (1.8015280000000001e+01 / rho);
  dy[INDEX(7)] = spec_rates[INDEX(6)] * (3.3006740000000001e+01 / rho);
  dy[INDEX(8)] = spec_rates[INDEX(7)] * (3.4014679999999998e+01 / rho);

} // end dydt

#elif defined(CONV)

__device__ void dydt (const double t, const double rho, const double * __restrict__ y, double * __restrict__ dy, mechanism_memory * __restrict__ d_mem) {

  // species molar concentrations
  double * __restrict__ conc = d_mem->conc;
  double y_N;
  double mw_avg;
  double pres;
  eval_conc_rho (y[INDEX(0)]rho, &y[GRID_DIM], &y_N, &mw_avg, &pres, conc);

  double * __restrict__ fwd_rates = d_mem->fwd_rates;
  double * __restrict__ rev_rates = d_mem->rev_rates;
  eval_rxn_rates (y[INDEX(0)], pres, conc, fwd_rates, rev_rates);

  // get pressure modifications to reaction rates
  double * __restrict__ pres_mod = d_mem->pres_mod;
  get_rxn_pres_mod (y[INDEX(0)], pres, conc, pres_mod);

  // evaluate species molar net production rates
  double dy_N;  eval_spec_rates (fwd_rates, rev_rates, pres_mod, &dy[GRID_DIM], &dy_N);

  double * __restrict__ cv = d_mem->cp;
  eval_cv(y[INDEX(0)], cv);

  // constant volume mass-average specific heat
  double cv_avg = (cv[INDEX(0)] * y[INDEX(1)]) + (cv[INDEX(1)] * y[INDEX(2)])
              + (cv[INDEX(2)] * y[INDEX(3)]) + (cv[INDEX(3)] * y[INDEX(4)])
              + (cv[INDEX(4)] * y[INDEX(5)]) + (cv[INDEX(5)] * y[INDEX(6)])
              + (cv[INDEX(6)] * y[INDEX(7)]) + (cv[INDEX(7)] * y[INDEX(8)])(cv[INDEX(8)] * y_N);

  // local array for species internal energies
  double * __restrict__ u = d_mem->h;
  eval_u (y[INDEX(0)], u);

  // rate of change of temperature
  dy[INDEX(0)] = (-1.0 / (rho * cv_avg)) * ((spec_rates[INDEX(0)] * u[INDEX(0)] * 2.0158800000000001e+00)
        + (spec_rates[INDEX(1)] * u[INDEX(1)] * 1.0079400000000001e+00)
        + (spec_rates[INDEX(2)] * u[INDEX(2)] * 1.5999400000000000e+01)
        + (spec_rates[INDEX(3)] * u[INDEX(3)] * 3.1998799999999999e+01)
        + (spec_rates[INDEX(4)] * u[INDEX(4)] * 1.7007339999999999e+01)
        + (spec_rates[INDEX(5)] * u[INDEX(5)] * 1.8015280000000001e+01)
        + (spec_rates[INDEX(6)] * u[INDEX(6)] * 3.3006740000000001e+01)
        + (spec_rates[INDEX(7)] * u[INDEX(7)] * 3.4014679999999998e+01));

  // calculate rate of change of species mass fractions
  dy[INDEX(1)] = spec_rates[INDEX(0)] * (2.0158800000000001e+00 / rho);
  dy[INDEX(2)] = spec_rates[INDEX(1)] * (1.0079400000000001e+00 / rho);
  dy[INDEX(3)] = spec_rates[INDEX(2)] * (1.5999400000000000e+01 / rho);
  dy[INDEX(4)] = spec_rates[INDEX(3)] * (3.1998799999999999e+01 / rho);
  dy[INDEX(5)] = spec_rates[INDEX(4)] * (1.7007339999999999e+01 / rho);
  dy[INDEX(6)] = spec_rates[INDEX(5)] * (1.8015280000000001e+01 / rho);
  dy[INDEX(7)] = spec_rates[INDEX(6)] * (3.3006740000000001e+01 / rho);
  dy[INDEX(8)] = spec_rates[INDEX(7)] * (3.4014679999999998e+01 / rho);

} // end dydt

#endif
