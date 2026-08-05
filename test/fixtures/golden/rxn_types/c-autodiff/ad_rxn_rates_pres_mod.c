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

  // sri variable declarations
  adouble X;

  adouble logT = log(T);
  adouble m = pres / (8.31446210e+03 * T);

  // reaction 4;
  pres_mod[0] = m + 1.5 * C[0] + 11.0 * C[5] - 0.25 * C[8];

  // reaction 5;
  thd = m + 1.0 * C[0] + 13.0 * C[5] - 0.32999999999999996 * C[8];
  k0 = exp(3.1778590439387948e+01 - 1.4 * logT);
  kinf = exp(2.2355638720663009e+01 + 0.44 * logT);
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(5.00000000e-01 * exp(-T / 1.00000000e-30) + 5.00000000e-01 * exp(-T / 1.00000000e+30) + exp(-1.00000000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[1] = pow(10.0, logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 6;
  thd = m + 1.5 * C[0] + 11.0 * C[5] - 0.36 * C[8];
  k0 = exp(3.2420178138029655e+01 - (2.2896490201091907e+04 / T));
  kinf = exp(3.3318335397877441e+01 - (2.4370923526129252e+04 / T));
  Pr = k0 * thd / kinf;
  X = 1.0 / (1.0 + log10(fmax(Pr, 1.0e-300)) * log10(fmax(Pr, 1.0e-300)));
  pres_mod[2] = pow(0.54 * exp(-201.0 / T) + exp(-T / 1024.0), X) * Pr / (1.0 + Pr);

} // end get_rxn_pres_mod

