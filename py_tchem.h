/* wrapper to TChem interface */

#ifndef PY_TCHEM_HEAD
#define PY_TCHEM_HEAD

void tc_eval_jacob (const char* mech, const char* thermo,
                    const int num, const double t, const double* pres,
                    const double* y, double* jac);

#endif
