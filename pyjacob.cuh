/* wrapper to translate to cuda arrays */

#ifndef CU_PYJAC_HEAD
#define CU_PYJAC_HEAD

void cu_eval_conc (const int num, const double* T, const double* pres, const double* mass_frac, double* mw_avg, double* rho, double* conc);
void cu_eval_rxn_rates (const int num, const double* T, const double* pres, const double * C, double * fwd_rxn_rates, double * rev_rxn_rates);
void cu_get_rxn_pres_mod (const int num, const double* T, const double* pres, const double * C, double * pres_mod);
void cu_eval_spec_rates (const int num, const double* fwd_rates, const double* rev_rates, const double* pres_mod, double* spec_rates);
void cu_dydt (const int num, const double* pres, const double* y, double* dy);
void cu_eval_jacob (const int num, const double* pres, const double* y, double* jac);

#endif