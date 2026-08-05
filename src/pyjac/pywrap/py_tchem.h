/* wrapper to TChem interface */

#ifndef PY_TCHEM_HEAD
#define PY_TCHEM_HEAD

void tc_eval_jacob (char* mech, char* thermo, const int num,
                    const double* pres, double* y, double* conc,
                    double* fwd_rates, double* rev_rates,
                    double* spec_rates, double* dydt, double* jac);

#endif
