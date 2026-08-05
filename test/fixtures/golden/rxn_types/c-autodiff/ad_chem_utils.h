#ifndef CHEM_UTILS_HEAD
#define CHEM_UTILS_HEAD

#include "header.h"

#include "adept.h"
using adept::adouble;
void eval_conc (const adouble&, const adouble&, const adouble * __restrict__, adouble * __restrict__, adouble * __restrict__, adouble * __restrict__, adouble * __restrict__);
void eval_conc_rho (const adouble&, const adouble&, const adouble * __restrict__, adouble * __restrict__, adouble * __restrict__, adouble * __restrict__, adouble * __restrict__);
void eval_h (const adouble&, adouble * __restrict__);
void eval_u (const adouble&, adouble * __restrict__);
void eval_cv (const adouble&, adouble * __restrict__);
void eval_cp (const adouble&, adouble * __restrict__);

#endif
