#ifndef RATES_HEAD
#define RATES_HEAD

#include "header.h"

#include "adept.h"
using adept::adouble;
void eval_rxn_rates (const adouble, const adouble&, const adouble * __restrict__, adouble * __restrict__, adouble * __restrict__);
void eval_spec_rates (const adouble * __restrict__, const adouble * __restrict__, const adouble * __restrict__, adouble * __restrict__, adouble * __restrict__);
void get_rxn_pres_mod (const adouble, const adouble&, const adouble * __restrict__, adouble * __restrict__);

#endif
