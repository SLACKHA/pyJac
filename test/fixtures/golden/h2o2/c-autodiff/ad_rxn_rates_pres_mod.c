#include <math.h>
#include "header.h"
#include "ad_rates.h"
#include "adept.h"
using adept::adouble;
#define fmax(a, b) (a.value() > b ? a : adouble(b))

void get_rxn_pres_mod (const adouble T, const adouble& pres, const adouble * __restrict__ C, adouble * __restrict__ pres_mod) {
  // third body variable declaration
  adouble thd;

  // pressure dependence variable declarations
  adouble k0;
  adouble kinf;
  adouble Pr;

  // troe variable declarations
  adouble logFcent;
  adouble A;
  adouble B;

  adouble logT = log(T);
  adouble m = pres / (8.31446210e+03 * T);

  // reaction 0;
  pres_mod[0] = m + 1.4 * C[0] + 14.4 * C[5] - 0.17000000000000004 * C[8];

  // reaction 1;
  pres_mod[1] = m + 1.0 * C[0] + 5.0 * C[5] - 0.30000000000000004 * C[8];

  // reaction 5;
  pres_mod[2] = m - 1.0 * C[3] - 1.0 * C[5] - 1.0 * C[8];

  // reaction 10;
  pres_mod[3] = m - 1.0 * C[0] - 1.0 * C[5] - 0.37 * C[8];

  // reaction 13;
  pres_mod[4] = m - 0.27 * C[0] + 2.65 * C[5] - 0.62 * C[8];

  // reaction 20;
  thd = m + 1.0 * C[0] + 5.0 * C[5] - 0.30000000000000004 * C[8];
  k0 = exp(2.8463930238863654e+01 - 0.9 * logT - (-8.5547326026057669e+02 / T));
  kinf = exp(2.5027330930150580e+01 - 0.37 * logT);
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(2.65400000e-01 * exp(-T / 9.40000000e+01) + 7.34600000e-01 * exp(-T / 1.75600000e+03) + exp(-5.18200000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[5] = pow(10.0, logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

} // end get_rxn_pres_mod

