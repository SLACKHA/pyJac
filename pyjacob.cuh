/* wrapper to translate to cuda arrays */

#ifndef CU_PYJAC_HEAD
#define CU_PYJAC_HEAD

void run(int num, const double* pres, const double* mass_frac,
			double* conc, double* fwd_rxn_rates, double* rev_rxn_rates,
			double* pres_mod, double* spec_rates, double* dy, double* jac);

#endif